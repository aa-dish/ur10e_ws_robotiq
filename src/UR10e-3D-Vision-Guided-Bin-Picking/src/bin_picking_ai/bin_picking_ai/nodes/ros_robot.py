"""ROS 2 implementation of RobotInterface (controllers, camera topics, Gazebo grasp topics)."""
from __future__ import annotations

import threading
import time
from typing import Callable, Dict, Optional

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Empty, String
from tf2_msgs.msg import TFMessage
from trajectory_msgs.msg import JointTrajectoryPoint

from ..geometry import make_T, pose_to_T
from ..kinematics import JOINT_NAMES
from ..motion.plan import Step
from ..robot_interface import RobotInterface

GRIPPER_JOINTS = ['gripper_joint']      # the only actuated joint of the Robotiq 2F-85 (others are mimic joints)


class RosRobot(RobotInterface):
    def __init__(self, node: Node, entities, base_world=(0.842175, -0.500138, 0.6)):
        self.n = node
        self.base_world = np.array(base_world)
        self.q = None
        self.grip = {}
        self.lock = threading.Lock()
        self._rgb = None
        self._depth = None
        self._K = None
        self._stamp_rgb = 0.0
        self._stamp_depth = 0.0
        self._poses: Dict[str, np.ndarray] = {}
        self.entities = list(entities)
        p = node.declare_parameter
        self.camera_frame = p('camera_frame', 'camera_color_optical_frame').value
        self.base_frame = p('base_frame', 'base_link').value
        self.cam_xyz = p('camera_xyz', [0.006, -0.62, 1.25]).value           # fallback if TF is missing
        self.arm_action = p('arm_action', '/arm_controller/follow_joint_trajectory').value
        self.grip_action = p('gripper_action', '/gripper_controller/follow_joint_trajectory').value
        self.image_topic = p('image_topic', '/d435/image').value
        self.depth_topic = p('depth_topic', '/d435/depth_image').value
        self.info_topic = p('camera_info_topic', '/d435/camera_info').value
        self.pose_topic = p('pose_topic', '/world/empty/pose/info').value
        node.create_subscription(JointState, '/joint_states', self._on_js, 10)
        node.create_subscription(Image, self.image_topic, self._on_rgb, 1)
        node.create_subscription(Image, self.depth_topic, self._on_depth, 1)
        node.create_subscription(CameraInfo, self.info_topic, self._on_info, 1)
        node.create_subscription(TFMessage, self.pose_topic, self._on_poses, 5)
        self.attach_pubs = {e: node.create_publisher(Empty, f'/bin_picking/attach/{e}', 10) for e in self.entities}
        self.detach_pubs = {e: node.create_publisher(Empty, f'/bin_picking/detach/{e}', 10) for e in self.entities}
        self.status_pub = node.create_publisher(String, '/bin_picking/status', 10)
        self.arm = ActionClient(node, FollowJointTrajectory, self.arm_action)
        self.gripper = ActionClient(node, FollowJointTrajectory, self.grip_action)
        import tf2_ros
        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf, node)
        self._T_cam = None
        self.last_error = ''

    # ------------------------------------------------------------ callbacks
    def _on_js(self, m: JointState):
        with self.lock:
            d = dict(zip(m.name, m.position))
            if all(j in d for j in JOINT_NAMES):
                self.q = np.array([d[j] for j in JOINT_NAMES])
            self.grip = {j: d[j] for j in GRIPPER_JOINTS if j in d}

    def _on_rgb(self, m: Image):
        if m.encoding not in ('rgb8', 'bgr8'):
            return
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
        with self.lock:
            self._rgb = a[..., ::-1].copy() if m.encoding == 'bgr8' else a.copy()
            self._stamp_rgb = time.monotonic()

    def _on_depth(self, m: Image):
        if m.encoding == '32FC1':
            a = np.frombuffer(m.data, np.float32).reshape(m.height, m.width)
        elif m.encoding == '16UC1':
            a = np.frombuffer(m.data, np.uint16).reshape(m.height, m.width).astype(np.float32) / 1000.0
        else:
            return
        with self.lock:
            self._depth = a.copy()
            self._stamp_depth = time.monotonic()

    def _on_info(self, m: CameraInfo):
        with self.lock:
            self._K = np.array(m.k, float).reshape(3, 3)

    def _on_poses(self, m: TFMessage):
        with self.lock:
            for t in m.transforms:
                name = t.child_frame_id.split('::')[-1]
                if name in self.entities:
                    tr = t.transform.translation
                    self._poses[name] = np.array([tr.x, tr.y, tr.z]) - self.base_world

    # ------------------------------------------------------------ RobotInterface
    def wait_ready(self, timeout=60.0) -> bool:
        t0 = time.monotonic()
        ok = False
        while time.monotonic() - t0 < timeout:
            ok = (self.arm.wait_for_server(timeout_sec=0.5) and self.gripper.wait_for_server(timeout_sec=0.5)
                  and self.q is not None)
            if ok:
                break
            self.sleep(0.2)
        return ok

    def get_arm_q(self):
        t0 = time.monotonic()
        while self.q is None and time.monotonic() - t0 < 5:
            time.sleep(0.05)
        with self.lock:
            if self.q is None:
                raise RuntimeError('no /joint_states received - the joint_state_broadcaster is not running. '
                                   'Check: ros2 control list_controllers')
            return self.q.copy()

    def _sim_now(self) -> float:
        """Seconds on the node clock = Gazebo time when use_sim_time is true.  Gazebo under WSL often runs at
        ~20 % real time, so every timeout must be measured in simulated, not wall-clock, seconds."""
        return self.n.get_clock().now().nanoseconds * 1e-9

    def _send(self, client: ActionClient, joints, points, should_abort, timeout_extra=10.0) -> bool:
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = joints
        goal.trajectory.points = points
        total = points[-1].time_from_start.sec + points[-1].time_from_start.nanosec * 1e-9
        fut = client.send_goal_async(goal)
        t0 = time.monotonic()
        while not fut.done():
            if time.monotonic() - t0 > 30:
                self.n.get_logger().error('trajectory goal was not accepted in time')
                self.last_error = 'goal not accepted in time'
                return False
            time.sleep(0.01)
        gh = fut.result()
        if not gh.accepted:
            self.n.get_logger().error('trajectory goal rejected')
            self.last_error = 'goal rejected by the controller'
            return False
        rf = gh.get_result_async()
        t0 = time.monotonic()
        s0 = self._sim_now()
        while not rf.done():
            if should_abort():
                gh.cancel_goal_async()
                return False
            if (self._sim_now() - s0 > total + timeout_extra + 8.0) or (time.monotonic() - t0 > 20 * total + 120):
                gh.cancel_goal_async()
                self.n.get_logger().error('trajectory timed out')
                self.last_error = 'trajectory timed out'
                return False
            time.sleep(0.01)
        res = rf.result().result
        ok = res.error_code == FollowJointTrajectory.Result.SUCCESSFUL
        if not ok:
            names = {-1: 'INVALID_GOAL', -2: 'INVALID_JOINTS', -3: 'OLD_HEADER_TIMESTAMP',
                     -4: 'PATH_TOLERANCE_VIOLATED', -5: 'GOAL_TOLERANCE_VIOLATED'}
            self.last_error = f'controller error {names.get(res.error_code, res.error_code)} {res.error_string}'
            self.n.get_logger().warn(f'trajectory finished with error {self.last_error}')
        return ok

    @staticmethod
    def _dur(t: float) -> Duration:
        return Duration(sec=int(t), nanosec=int((t - int(t)) * 1e9))

    def execute_arm(self, step: Step, should_abort: Callable[[], bool]) -> bool:
        q0 = self.get_arm_q()
        times = [0.0] + [t for t, _ in step.points]
        qs = np.vstack([q0] + [q for _, q in step.points])
        vel = np.gradient(qs, np.array(times), axis=0)
        vel[0] = 0.0
        vel[-1] = 0.0
        pts = []
        for i in range(1, len(times)):
            p = JointTrajectoryPoint()
            p.positions = [float(v) for v in qs[i]]
            p.velocities = [float(v) for v in vel[i]]
            p.time_from_start = self._dur(times[i] + 0.05)
            pts.append(p)
        ok = self._send(self.arm, JOINT_NAMES, pts, should_abort)
        if not ok:
            try:
                err = np.abs(self.get_arm_q() - qs[-1])
                j = int(np.argmax(err))
                self.last_error += f' | worst joint {JOINT_NAMES[j]} is {err[j]:.3f} rad from its target'
                self.n.get_logger().warn(f'arm step failed: {self.last_error}')
            except Exception:
                pass
        return ok

    def execute_gripper(self, theta, duration, should_abort) -> bool:
        from ..grasping.planner import GripperModel
        with self.lock:
            cur = [self.grip.get(j, 0.0) for j in GRIPPER_JOINTS]
        tgt = [float(theta)]
        pts = []
        for i, s in enumerate((0.5, 1.0)):
            p = JointTrajectoryPoint()
            p.positions = [c + (t - c) * s for c, t in zip(cur, tgt)]
            p.time_from_start = self._dur(duration * s)
            pts.append(p)
        self._send(self.gripper, GRIPPER_JOINTS, pts, should_abort, timeout_extra=3.0)
        return True                      # a closing gripper that stalls on the object is not an error

    def release_all(self):
        """The Gazebo plugin attaches every object when the robot spawns; make sure none is attached."""
        for pub in self.detach_pubs.values():
            pub.publish(Empty())

    def attach(self, entity: str):
        if entity in self.attach_pubs:
            self.attach_pubs[entity].publish(Empty())
        else:
            self.n.get_logger().error(f'unknown entity {entity!r} for attach')

    def _gz_detach(self, entity: str) -> None:
        """Same request through the gz command line (this route is known to work); runs in the background."""
        import subprocess
        try:
            subprocess.Popen(['gz', 'topic', '-t', f'/bin_picking/detach/{entity}', '-m', 'gz.msgs.Empty',
                              '-p', 'unused: true'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:                                       # gz not on PATH
            self.n.get_logger().warn(f'gz CLI detach unavailable: {e}')

    def detach(self, entity: str):
        if entity in self.detach_pubs:
            self.detach_pubs[entity].publish(Empty())
        self._gz_detach(entity)

    def sleep(self, seconds: float):
        s0 = self._sim_now()
        t0 = time.monotonic()
        while self._sim_now() - s0 < seconds and time.monotonic() - t0 < 20 * seconds + 5:
            time.sleep(0.02)

    def _camera_T(self) -> np.ndarray:
        if self._T_cam is not None:
            return self._T_cam
        try:
            from rclpy.time import Time
            tf = self.tf_buf.lookup_transform(self.base_frame, self.camera_frame, Time())
            t, r = tf.transform.translation, tf.transform.rotation
            self._T_cam = pose_to_T([t.x, t.y, t.z], [r.x, r.y, r.z, r.w])
            return self._T_cam
        except Exception as e:
            self.n.get_logger().warn(f'TF {self.base_frame}<-{self.camera_frame} unavailable ({e}); using camera_xyz parameter')
            R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
            return make_T(R, self.cam_xyz)

    def capture(self):
        t0 = time.monotonic()
        t_req = t0
        while True:
            with self.lock:
                fresh = (self._stamp_rgb > t_req and self._stamp_depth > t_req and self._K is not None)
                if fresh:
                    return self._rgb.copy(), self._depth.copy(), self._K.copy(), self._camera_T()
            if time.monotonic() - t0 > 40.0:
                raise RuntimeError('no camera images received - is the camera bridged? (ros2 topic list | grep d435)')
            time.sleep(0.02)

    def entity_poses(self) -> Optional[Dict[str, np.ndarray]]:
        with self.lock:
            return {k: v.copy() for k, v in self._poses.items()} if self._poses else None

    def say(self, message: str):
        self.n.get_logger().info(message)
        m = String()
        m.data = message
        self.status_pub.publish(m)
