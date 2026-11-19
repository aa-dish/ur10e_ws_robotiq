"""Offline emulator of the robot + camera (kinematic, no physics).  Used by the tests and by
``python -m bin_picking_ai.tools.offline_demo`` to try commands without ROS / Gazebo.

* the arm follows the planned joint waypoints exactly,
* ``attach`` rigidly attaches the named object to the tool, ``detach`` drops it onto the table
  (or into the bin when released above the bin opening),
* the camera image is rendered from the object meshes (tools/synthetic_scene.py).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .geometry import inv_T, make_T
from .kinematics import UR10e
from .motion.plan import Step
from .robot_interface import RobotInterface
from .tools.synthetic_scene import SyntheticScene, camera_intrinsics


class SimRobot(RobotInterface):
    def __init__(self, gazebo_dir: str, home_q, tcp_z: float = 0.15, bin_center=(0.7894, -0.6082),
                 bin_half=(0.17, 0.24), points_per_m2: float = 8e5, verbose: bool = False):
        self.scene = SyntheticScene(gazebo_dir, points_per_m2=points_per_m2)
        self.robot = UR10e(tcp_z)
        self.q = np.array(home_q, float)
        self.theta = 0.0          # Robotiq 2F-85: 0 rad = open
        self.moved: Dict[str, np.ndarray] = {}
        self.held: Optional[str] = None
        self.T_tcp_obj: Optional[np.ndarray] = None
        self.verbose = verbose
        self.bin_center = np.array(bin_center)
        self.bin_half = np.array(bin_half)
        self.time = 0.0
        self.table_z = 0.127
        self.trace = []                                   # (time, q, theta) for plotting / checks
        self.max_joint_step = 0.0
        self.min_tcp_z = 9.0

    # -- helpers
    def _obj(self, name):
        return next(o for o in self.scene.objects if o['name'] == name)

    def _T_obj(self, name):
        return self.moved.get(name, self.scene.object_pose_base(self._obj(name)))

    # -- RobotInterface
    def get_arm_q(self):
        return self.q.copy()

    def execute_arm(self, step: Step, should_abort):
        prev = self.q
        for t, q in step.points:
            self.max_joint_step = max(self.max_joint_step, float(np.max(np.abs(q - prev))))
            prev = q
            self.q = np.array(q)
            T = self.robot.fk(self.q)
            self.min_tcp_z = min(self.min_tcp_z, T[2, 3])
            if self.held:
                self.moved[self.held] = T @ self.T_tcp_obj
            if should_abort():
                return False
        self.time += step.duration
        return True

    def execute_gripper(self, theta, duration, should_abort):
        self.theta = theta
        self.time += duration
        return True

    def attach(self, entity):
        T_tcp = self.robot.fk(self.q)
        if entity not in [o['name'] for o in self.scene.objects]:
            raise RuntimeError(f'unknown entity {entity}')
        To = self._T_obj(entity)
        # object must be near the tool for a real grasp
        d = np.linalg.norm(To[:3, 3][:2] - T_tcp[:3, 3][:2])
        if d > 0.12:
            if self.verbose:
                print(f'[sim] attach {entity}: object is {d:.2f} m away from the tool - attaching anyway (like Gazebo)')
        self.held = entity
        self.T_tcp_obj = inv_T(T_tcp) @ To

    def detach(self, entity):
        if self.held != entity:
            return
        T = self.moved[entity].copy()
        xy = T[:2, 3]
        inbin = np.all(np.abs(xy - self.bin_center) < self.bin_half)
        # fall straight down, upright preserved
        T[2, 3] = (0.02 if inbin else self.table_z + (self._T_obj_rest_z(entity)))
        self.moved[entity] = T
        self.held = None

    def _T_obj_rest_z(self, entity) -> float:
        T0 = self.scene.object_pose_base(self._obj(entity))
        return float(T0[2, 3] - self.table_z)

    def sleep(self, seconds):
        self.time += seconds

    def capture(self):
        rgb, depth = self.scene.render(moved=self.moved)
        return rgb, depth, self.scene.K, self.scene.T_base_cam

    def entity_poses(self):
        return {o['name']: self._T_obj(o['name'])[:3, 3] for o in self.scene.objects}

    def say(self, message):
        print(message)
