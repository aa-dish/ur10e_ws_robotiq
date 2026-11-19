"""Top-down grasp planning for the Robotiq 2F-85 parallel gripper.

For one detected object it
  * proposes closing directions (across the short side first, then across the long side and a
    few rotated variants),
  * measures the object width along each (from the segmentation mask),
  * checks that the *open* fingers do not land on a neighbouring object (height-map test),
  * chooses the grasp height (near the top of the object so it never jams in the finger mechanism,
    never so low that the fingers touch the table) and the finger opening / closing angle,
  * returns a ranked list of :class:`Grasp`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from ..geometry import top_down_orientation, make_T, wrap_half_pi
from ..vision.detector import Detection, DetectionResult


@dataclass
class GripperModel:
    """Numbers measured from the gripper meshes - see config/gripper.yaml (defaults = Robotiq 2F-85)."""
    tcp_z: float = 0.15
    open_theta: float = 0.0
    max_theta: float = 0.78
    finger_reach_closed: float = 0.024
    finger_open_reach: float = 0.020
    finger_pad_width: float = 0.029
    finger_outer: float = 0.034
    inner_edge_inset: float = 0.0
    inner_rise: float = 0.026
    max_object_width: float = 0.080
    min_object_width: float = 0.012
    squeeze: float = 0.006
    open_margin: float = 0.030
    min_open_gap: float = 0.060
    rise_profile: Sequence[Sequence[float]] = field(default_factory=lambda: [
        [0.026, 0.045], [0.036, 0.025], [0.046, 0.013], [0.056, 0.007]])
    gap_lut: Sequence[Sequence[float]] = field(default_factory=lambda: [
        [0.0, 0.0850], [0.1, 0.0760], [0.2, 0.0663], [0.3, 0.0560], [0.4, 0.0453], [0.5, 0.0343],
        [0.6, 0.0230], [0.7, 0.0116], [0.7929, 0.0010]])

    @classmethod
    def from_dict(cls, d: dict) -> 'GripperModel':
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in keys})

    @property
    def open_gap(self) -> float:
        return self.gap_for_theta(self.open_theta)

    def gap_for_theta(self, theta: float) -> float:
        th = [r[0] for r in self.gap_lut]
        gp = [r[1] for r in self.gap_lut]
        return float(np.interp(theta, th, gp))

    def theta_for_gap(self, gap: float) -> float:
        th = np.array([r[0] for r in self.gap_lut])
        gp = np.array([r[1] for r in self.gap_lut])
        return float(np.interp(gap, gp[::-1], th[::-1]))

    @property
    def max_gap(self) -> float:
        return float(max(r[1] for r in self.gap_lut))

    def open_gap_for(self, width: float) -> float:
        """Finger gap to open to before the descent: object width + margin, within [min_open_gap, max_gap]."""
        return float(min(max(width + self.open_margin, self.min_open_gap), self.max_gap))

    def max_rise(self, object_width: float) -> float:
        """How far the top of an object of this width may stick up above the tcp [m]."""
        a = object_width / 2.0 + 0.008                                    # + safety margin
        hs = [r[0] for r in self.rise_profile]
        ws = [r[1] for r in self.rise_profile]
        return float(np.interp(a, ws[::-1], hs[::-1]))                    # np.interp wants ascending x

    def joint_values(self, theta: float) -> List[float]:
        """[gripper_joint]  (all other gripper joints are mimic joints)."""
        return [float(min(max(theta, 0.0), self.max_theta))]


@dataclass
class Grasp:
    center_xy: np.ndarray            # grasp centre in base_link
    tcp_z: float                     # z of the tcp (base_link) when grasping
    closing_yaw: float               # direction the fingers close along (base_link, rad)
    width: float                     # object width along the closing direction [m]
    close_theta: float               # finger angle to command when closing
    score: float                     # higher is better
    clearance: float                 # free height above the neighbours inside the finger boxes [m] (>0 is free)
    notes: str = ''
    open_theta: Optional[float] = None               # finger angle to open to before descending (None = gripper default)
    blockers: dict = field(default_factory=dict)     # {detection id: how far it sticks above the finger limit [m]}

    def pose(self, dz: float = 0.0) -> np.ndarray:
        return make_T(top_down_orientation(self.closing_yaw), [self.center_xy[0], self.center_xy[1], self.tcp_z + dz])


class GraspPlanner:
    def __init__(self, gripper: Optional[GripperModel] = None, table_margin: float = 0.006):
        self.g = gripper or GripperModel()
        self.table_margin = table_margin

    # -------------------------------------------------------------- helpers
    def _grasp_height(self, det: Detection, width: Optional[float] = None) -> float:
        """tcp height above the table: grasp near the TOP of the object so it never jams in the finger mechanism."""
        h = det.height
        w = width if width is not None else min(det.size)
        lo = self.g.finger_reach_closed + self.table_margin           # fingers must not touch the table
        hi = max(h - 0.012, lo)                                       # fingertips must still overlap the object
        return float(min(max(h - self.g.max_rise(w), lo), hi))

    @staticmethod
    def _width_along(mask_xy: np.ndarray, centre: np.ndarray, direction: np.ndarray):
        d = mask_xy - centre
        s = d @ direction
        return float(s.max() - s.min()), float((s.max() + s.min()) / 2)

    def _clearance(self, res: DetectionResult, det: Detection, centre: np.ndarray, direction: np.ndarray,
                   width: float, grasp_h: float, blockers: Optional[dict] = None,
                   open_gap: Optional[float] = None) -> float:
        """Smallest (allowed height - neighbour height) under the OPEN gripper (>0 = free).

        Footprint of the Robotiq 2F-85 (tcp frame, closing axis x, measured from the meshes, config/gripper.yaml):
          between the pads         free up to ``inner_rise`` above the tcp, above that the inner knuckles
          finger links + pads      |x| from gap/2 to gap/2 + ``finger_outer``, their lower end reaches
                                   ``finger_open_reach`` BELOW the tcp
        so anything in the finger zone that is taller than (tcp - finger_open_reach) is hit on the way down.
        """
        g = self.g
        perp = np.array([-direction[1], direction[0]])
        c = res.cell
        tip_limit = grasp_h - g.finger_open_reach - 0.004
        inner_limit = grasp_h + g.inner_rise
        og = open_gap if open_gap is not None else g.open_gap
        reach = og / 2.0 + g.finger_outer              # outer edge of the finger links
        inner_edge = og / 2.0 - g.inner_edge_inset
        half_pad = self.g.finger_pad_width / 2.0 + 0.006
        worst = min(tip_limit, inner_limit)
        hm = res.height_map
        lm = res.label_map
        ny, nx = hm.shape
        for a in np.arange(-reach, reach + 1e-9, c):
            limit = tip_limit if abs(a) >= inner_edge else inner_limit
            for b in np.arange(-half_pad, half_pad + 1e-9, c):
                p = centre + direction * a + perp * b
                col, row = res.xy_to_px(p[0], p[1])
                if not (0 <= col < nx and 0 <= row < ny):
                    worst = min(worst, -0.05)                       # finger would leave the table region
                    continue
                if lm[row, col] == det.id:
                    continue                                        # that's the object itself
                hv = hm[row, col]
                if np.isfinite(hv):
                    worst = min(worst, limit - hv)
                    if blockers is not None and limit - hv < -0.004 and lm[row, col] > 0:
                        k = int(lm[row, col])
                        blockers[k] = max(blockers.get(k, 0.0), hv - limit)
        return float(worst)

    # -------------------------------------------------------------- main
    def plan(self, det: Detection, res: DetectionResult, extra_yaw_offsets=(0.0,)) -> List[Grasp]:
        g = self.g
        ys, xs = np.nonzero(res.label_map == det.id)
        if len(xs) == 0:
            return []
        mask_xy = np.stack([res.origin[0] + (xs + 0.5) * res.cell, res.origin[1] + (ys + 0.5) * res.cell], 1)
        centre = np.array(det.center[:2], float)
        out: List[Grasp] = []
        base = det.yaw
        cand_angles = []
        for off in (math.pi / 2, 0.0, math.pi / 2 + math.radians(30), math.pi / 2 - math.radians(30),
                    math.radians(30), -math.radians(30), math.pi / 4, -math.pi / 4):
            cand_angles.append(wrap_half_pi(base + off))
        seen = []
        for ang in cand_angles:
            if any(abs(wrap_half_pi(ang - s)) < 1e-3 for s in seen):
                continue
            seen.append(ang)
            d = np.array([math.cos(ang), math.sin(ang)])
            width, shift = self._width_along(mask_xy, centre, d)
            width += res.cell
            if width > g.max_object_width or width < g.min_object_width:
                continue
            # the object top must stay below the finger hinges: the allowed rise above the tcp depends on the
            # object width across the closing direction
            grasp_h = self._grasp_height(det, width)
            tcp_z_world = res.table_z + grasp_h
            # centre the fingers on the middle of the object along the closing direction
            c2 = centre + d * shift
            # and also along the perpendicular axis (objects are not always symmetric)
            perp = np.array([-d[1], d[0]])
            _, shift_p = self._width_along(mask_xy, c2, perp)
            c2 = c2 + perp * shift_p
            blk: dict = {}
            # open wide enough to forgive a few degrees of yaw / a few mm of position error:
            og = g.open_gap_for(width)
            open_th = g.theta_for_gap(og)
            clr = self._clearance(res, det, c2, d, width, grasp_h, blk, open_gap=og)
            gap_cmd = max(width - g.squeeze, 0.005)
            theta = min(g.theta_for_gap(gap_cmd), g.max_theta)
            # score: collision-free first, then narrow objects across their short side
            score = (2.0 if clr > 0.0 else 0.0) + min(clr, 0.03) * 10.0 - width * 3.0 \
                + (0.3 if abs(wrap_half_pi(ang - base - math.pi / 2)) < 1e-3 else 0.0)
            out.append(Grasp(center_xy=c2, tcp_z=tcp_z_world, closing_yaw=ang, width=width, close_theta=theta,
                             score=score, clearance=clr, blockers=blk, open_theta=open_th,
                             notes=f'width={width * 100:.1f}cm clearance={clr * 100:.1f}cm'))
        out.sort(key=lambda gr: -gr.score)
        return out
