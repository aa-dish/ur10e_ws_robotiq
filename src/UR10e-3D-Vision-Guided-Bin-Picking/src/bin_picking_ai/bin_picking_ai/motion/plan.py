"""Build a complete pick-and-place plan (list of steps) - pure python, no ROS.

Step kinds
  arm      : joint waypoints ``[(t, q6), ...]`` for the arm controller
  gripper  : target finger angle theta1 [rad] and duration
  attach / detach : virtual grasp (Gazebo DetachableJoint) of a named entity
  wait     : pause
Every arm segment is generated either as a Cartesian straight line (chained IK, used for
approach / lift / carry) or as a time-scaled joint interpolation, and is checked for joint
limits, jumps between IK solutions and a minimum tool height.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..geometry import make_T, top_down_orientation, wrap_half_pi
from ..grasping.planner import Grasp, GripperModel
from ..kinematics import UR10e, Q_PREF_MAX, Q_PREF_MIN


class BlockedError(RuntimeError):
    """Every grasp is blocked by neighbours; ``blockers`` = detection ids to move away first (best first)."""

    def __init__(self, message, blockers=()):
        super().__init__(message)
        self.blockers = list(blockers)


class PlanError(RuntimeError):
    pass


@dataclass
class Step:
    kind: str
    name: str
    points: List[Tuple[float, np.ndarray]] = field(default_factory=list)   # arm
    theta: float = 0.0                                                      # gripper
    duration: float = 1.0
    entity: str = ''

    def __repr__(self):
        if self.kind == 'arm':
            return f'Step(arm {self.name} {len(self.points)} pts {self.duration:.1f}s)'
        return f'Step({self.kind} {self.name})'


@dataclass
class Plan:
    steps: List[Step]
    q_end: np.ndarray
    info: Dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return sum(s.duration for s in self.steps)


def _min_jerk(s: float) -> float:
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


class PickPlacePlanner:
    def __init__(self, robot: UR10e, gripper: GripperModel, cfg: dict, table_z: float, rate: float = 20.0):
        self.r = robot
        self.g = gripper
        self.c = cfg
        self.m = cfg['motion']
        self.table_z = table_z
        self.rate = rate

    # ------------------------------------------------------------ segment builders
    def joint_segment(self, q0: np.ndarray, q1: np.ndarray, name: str, speed: Optional[float] = None,
                      min_z: Optional[float] = None) -> Step:
        speed = speed or self.m['joint_speed']
        dq = np.abs(q1 - q0)
        T = max(float(dq.max()) / speed * 1.875, self.m['joint_accel_time'] * 2, 0.8)   # min-jerk peak = 1.875 * mean
        n = max(int(T * self.rate), 4)
        pts = [(T * i / n, q0 + (q1 - q0) * _min_jerk(i / n)) for i in range(1, n + 1)]
        if min_z is not None:
            zmin = min(self.r.fk(q)[2, 3] for _, q in pts[::2])
            if zmin < min_z:
                raise PlanError(f'{name}: joint-space path dips to z={zmin:.2f} m (< {min_z:.2f}); needs a via point')
        return Step('arm', name, pts, duration=T)

    def line_segment(self, q0: np.ndarray, T_goal: np.ndarray, name: str, speed: Optional[float] = None,
                     max_step: float = 0.45) -> Tuple[Step, np.ndarray]:
        """Straight-line (and slerp-free: orientation is constant) move of the tcp via chained IK."""
        speed = speed or self.m['line_speed']
        T0 = self.r.fk(q0)
        dist = float(np.linalg.norm(T_goal[:3, 3] - T0[:3, 3]))
        rot_err = float(np.linalg.norm(self.r.pose_error(T0, T_goal)[3:]))
        n_ik = max(int(math.ceil(dist / 0.02)), int(math.ceil(rot_err / 0.05)), 2)
        qs = [q0]
        for i in range(1, n_ik + 1):
            s = i / n_ik
            Ti = T_goal.copy()
            Ti[:3, 3] = T0[:3, 3] + (T_goal[:3, 3] - T0[:3, 3]) * s
            if rot_err > 1e-3:                       # blend orientation (rotation vector interpolation)
                Ti[:3, :3] = _slerp(T0[:3, :3], T_goal[:3, :3], s)
            q = self.r.ik(Ti, qs[-1], max_iter=60)
            if q is None:
                raise PlanError(f'{name}: no IK along the straight line at {s * 100:.0f}% '
                                f'(target {np.round(Ti[:3, 3], 3)})')
            if np.max(np.abs(q - qs[-1])) > max_step:
                raise PlanError(f'{name}: IK solution jumps by {np.max(np.abs(q - qs[-1])):.2f} rad at {s * 100:.0f}%')
            qs.append(q)
        T = max(dist / speed * 1.875 if dist > 1e-4 else 0.0, rot_err / 1.0 * 1.875, 0.8)
        # time-scale the IK samples with min-jerk, then resample uniformly
        qs = np.array(qs)
        ss = np.linspace(0, 1, len(qs))
        n = max(int(T * self.rate), 4)
        pts = []
        for i in range(1, n + 1):
            u = _min_jerk(i / n)
            pts.append((T * i / n, np.array([np.interp(u, ss, qs[:, j]) for j in range(6)])))
        return Step('arm', name, pts, duration=T), qs[-1]

    # ------------------------------------------------------------ IK helpers
    def _solve_pose(self, T: np.ndarray, q_ref: np.ndarray, what: str, try_flip: bool = True) -> np.ndarray:
        """IK for a top-down pose; the 2-finger gripper is symmetric so yaw and yaw+pi are both valid."""
        best = None
        for flip in ([0.0, math.pi] if try_flip else [0.0]):
            Tf = T.copy()
            if flip:
                Rz = np.array([[math.cos(flip), -math.sin(flip), 0], [math.sin(flip), math.cos(flip), 0], [0, 0, 1]])
                Tf[:3, :3] = T[:3, :3] @ Rz                        # rotate about the tool z axis
            q = self.r.solve(Tf, q_ref, n_random=10)
            if q is not None:
                cost = float(np.sum((q - q_ref) ** 2))
                if best is None or cost < best[0]:
                    best = (cost, q)
        if best is None:
            raise PlanError(f'{what}: pose {np.round(T[:3, 3], 3)} is not reachable')
        return best[1]

    # ------------------------------------------------------------ the whole thing
    def plan_pick_place(self, q_start: np.ndarray, grasp: Grasp, obj_entity: str, obj_height: float,
                        place: Dict, travel_z: float, home_q: Sequence[float]) -> Plan:
        """place: {'kind': 'bin'|'table', 'xy': (x, y), 'yaw': closing yaw (table), 'z': tcp z (table)}"""
        g, m = self.g, self.m
        steps: List[Step] = []
        info: Dict = {}
        q = np.asarray(q_start, float)
        home = np.asarray(home_q, float)
        open_th = grasp.open_theta if getattr(grasp, 'open_theta', None) is not None else g.open_theta

        T_grasp = grasp.pose()
        T_pre = grasp.pose(dz=max(travel_z - grasp.tcp_z, m['approach_height']))
        # 1. open gripper, go above the object
        steps.append(Step('gripper', 'open', theta=open_th, duration=1.0))
        q_pre = self._solve_pose(T_pre, q, 'pre-grasp')
        steps.append(self.joint_segment(q, q_pre, 'to_pregrasp', min_z=min(self.r.fk(q)[2, 3], T_pre[2, 3]) - 0.12))
        q = q_pre
        R_hold = self.r.fk(q_pre)[:3, :3]                       # keep exactly this tool orientation (yaw or yaw+pi)
        T_grasp[:3, :3] = R_hold
        # 2. straight down to the grasp
        st, q = self.line_segment(q, T_grasp, 'descend')
        steps.append(st)
        # 3. close, attach, settle
        steps.append(Step('gripper', 'close', theta=grasp.close_theta, duration=1.5))
        steps.append(Step('attach', 'attach', entity=obj_entity, duration=0.3))
        steps.append(Step('wait', 'settle', duration=m['settle_time']))
        # 4. lift straight up
        T_lift = grasp.pose(dz=max(travel_z - grasp.tcp_z, m['lift_height']))
        T_lift[:3, :3] = R_hold
        st, q = self.line_segment(q, T_lift, 'lift')
        steps.append(st)
        # 5. carry to the drop pose, then lower for table placement
        if place['kind'] == 'bin':
            held_below = grasp.tcp_z - self.table_z                    # tcp height above the object's bottom
            z_drop = self.c['bin']['rim_z'] + self.c['bin']['release_clearance'] + max(held_below, 0.04)
            z_drop = max(z_drop, T_lift[2, 3] * 0.0 + z_drop)
            above = np.array([place['xy'][0], place['xy'][1], max(z_drop, travel_z) if travel_z > z_drop else z_drop])
            best = None
            for yaw in [grasp.closing_yaw, 0.0, math.pi / 2, -math.pi / 2, math.pi / 4, -math.pi / 4]:
                T_drop = make_T(top_down_orientation(yaw), [place['xy'][0], place['xy'][1], z_drop])
                try:
                    q_drop = self._solve_pose(T_drop, q, 'bin drop')
                except PlanError:
                    continue
                cost = float(np.sum((q_drop - q) ** 2))
                if best is None or cost < best[0]:
                    best = (cost, T_drop, q_drop)
            if best is None:
                raise PlanError('bin drop pose is not reachable (move the bin closer or lower the lift)')
            _, T_drop, q_drop = best
            # carry in joint space (the lifted tool is above everything), keep a floor check
            steps.append(self.joint_segment(q, q_drop, 'carry_to_bin', speed=0.7,
                                            min_z=min(T_lift[2, 3], z_drop) - 0.10))
            q = q_drop
            info['drop_z'] = z_drop
            T_retreat = self.r.fk(q_drop)
            T_retreat[2, 3] += 0.10
        else:
            z_place = place['z']
            T_above = make_T(top_down_orientation(place['yaw']), [place['xy'][0], place['xy'][1], T_lift[2, 3]])
            T_down = make_T(top_down_orientation(place['yaw']), [place['xy'][0], place['xy'][1], z_place])
            q_above = self._solve_pose(T_above, q, 'place above')
            steps.append(self.joint_segment(q, q_above, 'carry_to_place', speed=0.7, min_z=T_lift[2, 3] - 0.12))
            q = q_above
            T_down[:3, :3] = self.r.fk(q_above)[:3, :3]
            st, q = self.line_segment(q, T_down, 'lower_object', speed=m['line_speed'])
            steps.append(st)
            T_retreat = T_above.copy()
            T_retreat[:3, :3] = T_down[:3, :3]
        # 6. release
        steps.append(Step('detach', 'detach', entity=obj_entity, duration=0.3))
        steps.append(Step('gripper', 'open', theta=open_th, duration=1.0))
        steps.append(Step('wait', 'let_go', duration=0.5))
        # 7. retreat and go home
        st, q = self.line_segment(q, T_retreat, 'retreat')
        steps.append(st)
        steps.append(self.joint_segment(q, home, 'to_home', min_z=None))
        q = home
        return Plan(steps, q, info)


def _slerp(R0: np.ndarray, R1: np.ndarray, s: float) -> np.ndarray:
    dR = R0.T @ R1
    ang = math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2)))
    if ang < 1e-6:
        return R1.copy() if s >= 1 else R0.copy()
    axis = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2 * math.sin(ang))
    a = ang * s
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return R0 @ (np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K)
