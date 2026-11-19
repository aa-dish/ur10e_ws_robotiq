"""Small SE(3) helpers (numpy only, no ROS imports).

Conventions
-----------
* A pose is a 4x4 homogeneous matrix ``T_a_b`` that maps points expressed in
  frame ``b`` into frame ``a``  (``p_a = T_a_b @ p_b``).
* Quaternions are ``(x, y, z, w)`` like ROS.
* RPY is ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`` like URDF / SDF.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def matrix_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    """Inverse of :func:`rpy_to_matrix` (pitch in [-pi/2, pi/2])."""
    sp = -R[2, 0]
    pitch = math.asin(max(-1.0, min(1.0, sp)))
    if abs(sp) < 1.0 - 1e-9:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    return roll, pitch, yaw


def make_T(R: np.ndarray | None = None, t: Sequence[float] | None = None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def xyz_rpy_to_T(x: float, y: float, z: float, roll: float = 0.0, pitch: float = 0.0,
                 yaw: float = 0.0) -> np.ndarray:
    return make_T(rpy_to_matrix(roll, pitch, yaw), (x, y, z))


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion (x, y, z, w), w >= 0."""
    m = np.asarray(R, float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
                      (m[1, 0] - m[0, 1]) / s, 0.25 * s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = np.array([0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s,
                      (m[2, 1] - m[1, 2]) / s])
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = np.array([(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s,
                      (m[0, 2] - m[2, 0]) / s])
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = np.array([(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s,
                      (m[1, 0] - m[0, 1]) / s])
    q = q / np.linalg.norm(q)
    return q if q[3] >= 0 else -q


def quat_to_matrix(q: Iterable[float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array([[1 - yy - zz, xy - wz, xz + wy],
                     [xy + wz, 1 - xx - zz, yz - wx],
                     [xz - wy, yz + wx, 1 - xx - yy]])


def pose_to_T(position: Sequence[float], quat_xyzw: Sequence[float]) -> np.ndarray:
    return make_T(quat_to_matrix(quat_xyzw), position)


def T_to_pose(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return T[:3, 3].copy(), matrix_to_quat(T[:3, :3])


def wrap_pi(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    a = (a + math.pi) % (2 * math.pi) - math.pi
    return math.pi if a == -math.pi else a


def wrap_half_pi(a: float) -> float:
    """Wrap an angle to (-pi/2, pi/2]  (for 180-degree symmetric things, like a 2-finger grasp)."""
    a = (a + math.pi / 2) % math.pi - math.pi / 2
    return math.pi / 2 if a == -math.pi / 2 else a


def top_down_orientation(closing_yaw: float) -> np.ndarray:
    """Tool orientation for a vertical (top-down) grasp.

    The tool z axis (approach direction) points down (-z of the base frame) and
    the tool x axis (the gripper closing direction) lies in the horizontal plane
    at angle ``closing_yaw`` from the base x axis.  Right handed:
    ``y = z x x = (sin(yaw), -cos(yaw), 0)``.
    """
    c, s = math.cos(closing_yaw), math.sin(closing_yaw)
    x = np.array([c, s, 0.0])
    z = np.array([0.0, 0.0, -1.0])
    y = np.cross(z, x)
    return np.column_stack([x, y, z])
