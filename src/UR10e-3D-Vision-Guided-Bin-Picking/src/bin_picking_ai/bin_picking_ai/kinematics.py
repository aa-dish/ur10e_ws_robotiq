"""UR10e forward / inverse kinematics (numpy only).

Constants come from ``ur_description/config/ur10e/default_kinematics.yaml`` and the
URDF chain (base_link -> ... -> flange -> tool0 -> tcp).  All poses are expressed
in ``base_link`` (REP-103: x forward).  ``tcp`` is the grasp centre of the gripper:
``tcp_z`` metres along the approach (+z of tool0) axis.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from .geometry import inv_T, make_T, rot_z, rpy_to_matrix, xyz_rpy_to_T

JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

# (xyz, rpy) of each revolute joint origin; every joint rotates about its local z.
_CHAIN = [
    ((0, 0, 0.1807), (0, 0, 0)),
    ((0, 0, 0), (1.570796327, 0, 0)),
    ((-0.6127, 0, 0), (0, 0, 0)),
    ((-0.57155, 0, 0.17415), (0, 0, 0)),
    ((0, -0.11985, -2.458164590756244e-11), (1.570796327, 0, 0)),
    ((0, 0.11655, -2.390480459346185e-11), (1.570796326589793, math.pi, math.pi)),
]
_T_BASE = xyz_rpy_to_T(0, 0, 0, 0, 0, math.pi)                       # base_link -> base_link_inertia
_T_FLANGE = xyz_rpy_to_T(0, 0, 0, 0, -math.pi / 2, -math.pi / 2)       # wrist_3_link -> flange
_T_TOOL0 = xyz_rpy_to_T(0, 0, 0, math.pi / 2, 0, math.pi / 2)          # flange -> tool0
_FIXED = [xyz_rpy_to_T(*xyz, *rpy) for xyz, rpy in _CHAIN]

# Soft limits used by the planner (a bit inside the URDF limits).
Q_MIN = np.array([-2 * math.pi, -2 * math.pi, -math.pi, -2 * math.pi, -2 * math.pi, -2 * math.pi])
Q_MAX = -Q_MIN
# Keep the working range sane: no wild multi-turn wrist solutions.
Q_PREF_MIN = np.array([-math.pi, -math.pi, -math.pi, -math.pi * 1.25, -math.pi, -math.pi])
Q_PREF_MAX = -Q_PREF_MIN


class UR10e:
    def __init__(self, tcp_z: float = 0.15):
        self.tcp_z = tcp_z
        self.T_tcp = make_T(np.eye(3), (0, 0, tcp_z))

    def fk_tool0(self, q: Sequence[float]) -> np.ndarray:
        T = _T_BASE.copy()
        for Tf, qi in zip(_FIXED, q):
            T = T @ Tf @ make_T(rot_z(qi))
        return T @ _T_FLANGE @ _T_TOOL0

    def fk(self, q: Sequence[float]) -> np.ndarray:
        """Pose of the tcp frame in base_link."""
        return self.fk_tool0(q) @ self.T_tcp

    def jacobian(self, q: Sequence[float], eps: float = 1e-6) -> np.ndarray:
        """6x6 numeric geometric Jacobian of the tcp (v then w, base frame)."""
        q = np.asarray(q, float)
        T0 = self.fk(q)
        J = np.zeros((6, 6))
        for i in range(6):
            dq = q.copy()
            dq[i] += eps
            T1 = self.fk(dq)
            J[:3, i] = (T1[:3, 3] - T0[:3, 3]) / eps
            dR = T1[:3, :3] @ T0[:3, :3].T
            J[3:, i] = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2 * eps)
        return J

    @staticmethod
    def pose_error(T_cur: np.ndarray, T_goal: np.ndarray) -> np.ndarray:
        e_p = T_goal[:3, 3] - T_cur[:3, 3]
        dR = T_goal[:3, :3] @ T_cur[:3, :3].T
        e_w = 0.5 * np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
        return np.concatenate([e_p, e_w])

    def ik(self, T_goal: np.ndarray, q_seed: Sequence[float], max_iter: int = 200,
           pos_tol: float = 1e-4, rot_tol: float = 1e-3, lam: float = 0.05) -> Optional[np.ndarray]:
        """Damped least squares from one seed. Returns joints or None."""
        q = np.asarray(q_seed, float).copy()
        for _ in range(max_iter):
            T = self.fk(q)
            e = self.pose_error(T, T_goal)
            if np.linalg.norm(e[:3]) < pos_tol and np.linalg.norm(e[3:]) < rot_tol:
                if np.all(q >= Q_PREF_MIN - 1e-9) and np.all(q <= Q_PREF_MAX + 1e-9):
                    return q
                return None
            J = self.jacobian(q)
            dq = J.T @ np.linalg.solve(J @ J.T + (lam ** 2) * np.eye(6), e)
            n = np.max(np.abs(dq))
            if n > 0.5:
                dq *= 0.5 / n
            q = np.clip(q + dq, Q_PREF_MIN, Q_PREF_MAX)
        return None

    def solve(self, T_goal: np.ndarray, q_ref: Sequence[float], n_random: int = 24,
              rng: Optional[np.random.Generator] = None) -> Optional[np.ndarray]:
        """Best IK solution = the one closest (joint space) to ``q_ref``."""
        rng = rng or np.random.default_rng(0)
        q_ref = np.asarray(q_ref, float)
        seeds = [q_ref]
        for _ in range(n_random):
            seeds.append(np.clip(q_ref + rng.normal(0, 0.9, 6), Q_PREF_MIN, Q_PREF_MAX))
        sols = []
        for s in seeds:
            q = self.ik(T_goal, s)
            if q is not None and not any(np.allclose(q, o, atol=1e-3) for o in sols):
                sols.append(q)
                if len(sols) >= 4:
                    break
        if not sols:
            return None
        w = np.array([1.0, 1.0, 1.0, 1.0, 0.5, 0.3])
        return min(sols, key=lambda q: float(np.sum(w * (q - q_ref) ** 2)))
