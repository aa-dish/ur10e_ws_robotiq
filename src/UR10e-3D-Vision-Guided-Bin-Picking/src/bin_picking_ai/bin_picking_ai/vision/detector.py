"""RGB-D object detection + 3D pose estimation for a top-down camera (numpy + OpenCV only).

Pipeline
  1. back-project depth to 3D, transform to ``base_link``
  2. rasterise a top-down *height map* (max z per 3 mm cell) + colour map
  3. estimate the table height (mode of the heights) and threshold objects above it
  4. connected components (+ watershed split of merged blobs)
  5. per blob: oriented footprint (min-area rectangle) -> centre, size, yaw; top height;
     colour; recognition against the catalogue (size + colour, see classifier.py)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..geometry import wrap_half_pi
from .classifier import Catalog


@dataclass
class VisionParams:
    roi_x: Tuple[float, float] = (-0.62, 0.62)      # metres in base_link
    roi_y: Tuple[float, float] = (-1.00, -0.25)
    cell: float = 0.003                              # height-map resolution [m]
    min_height: float = 0.018                        # objects must stick out this far above the table
    table_z: Optional[float] = None                  # None = estimate from the depth image
    min_area_m2: float = 0.0006                      # ignore blobs smaller than ~2.5x2.5 cm
    max_depth: float = 2.5
    split_merged: bool = True


@dataclass
class Detection:
    id: int
    label: Optional[str]                  # catalogue id, None if unknown
    label_cost: float
    candidates: List[Tuple[str, float]]   # best 3 (class id, cost)
    center: Tuple[float, float, float]    # x, y of footprint centre, z of the TOP surface (base_link)
    height: float                         # top surface above the table
    size: Tuple[float, float]             # footprint (length, width) in metres, length >= width
    yaw: float                            # direction of the long footprint axis in base_link [rad], (-pi/2, pi/2]
    mean_rgb: Tuple[int, int, int]
    area_m2: float
    bbox_px: Tuple[int, int, int, int] = (0, 0, 0, 0)    # in the height-map image
    color_name: str = ''
    extra: Dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = asdict(self)
        d['center'] = [round(float(v), 4) for v in self.center]
        d['size'] = [round(float(v), 4) for v in self.size]
        d['yaw'] = round(float(self.yaw), 4)
        d['height'] = round(float(self.height), 4)
        d['candidates'] = [(c, round(float(s), 3)) for c, s in self.candidates]
        return d


@dataclass
class DetectionResult:
    detections: List[Detection]
    table_z: float
    height_map: np.ndarray          # (ny, nx) float32, NaN where nothing was seen (height above table)
    color_map: np.ndarray           # (ny, nx, 3) uint8
    label_map: np.ndarray           # (ny, nx) int32, detection id or -1
    origin: Tuple[float, float]     # base_link x, y of cell (0, 0) corner
    cell: float

    def px_to_xy(self, col: float, row: float) -> Tuple[float, float]:
        return self.origin[0] + (col + 0.5) * self.cell, self.origin[1] + (row + 0.5) * self.cell

    def xy_to_px(self, x: float, y: float) -> Tuple[int, int]:
        return int((x - self.origin[0]) / self.cell), int((y - self.origin[1]) / self.cell)


def colour_name(rgb: Tuple[float, float, float]) -> str:
    """Rough human colour word for an RGB triple (used to resolve 'the red one')."""
    px = np.uint8([[[rgb[0], rgb[1], rgb[2]]]])
    h, s, v = cv2.cvtColor(px, cv2.COLOR_RGB2HSV)[0, 0].astype(int)
    if v < 55:
        return 'black'
    if s < 35:
        return 'white' if v > 175 else 'gray'
    deg = h * 2
    if deg < 15 or deg >= 345:
        return 'red'
    if deg < 40:
        return 'brown' if v < 150 else 'orange'
    if deg < 70:
        return 'yellow'
    if deg < 165:
        return 'green'
    if deg < 255:
        return 'blue'
    if deg < 290:
        return 'purple'
    return 'pink'


class Detector:
    def __init__(self, catalog: Optional[Catalog] = None, params: Optional[VisionParams] = None):
        self.catalog = catalog
        self.p = params or VisionParams()

    # ------------------------------------------------------------------ geometry
    def _cloud_base(self, depth: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray):
        h, w = depth.shape
        v, u = np.mgrid[0:h, 0:w]
        z = depth.astype(np.float32)
        ok = np.isfinite(z) & (z > 0.05) & (z < self.p.max_depth)
        x = (u[ok] - K[0, 2]) / K[0, 0] * z[ok]
        y = (v[ok] - K[1, 2]) / K[1, 1] * z[ok]
        pts = np.stack([x, y, z[ok]], 1)
        pb = pts @ T_base_cam[:3, :3].T + T_base_cam[:3, 3]
        return pb, ok

    def _rasterise(self, pb: np.ndarray, rgb: np.ndarray, ok: np.ndarray):
        p = self.p
        nx = int(round((p.roi_x[1] - p.roi_x[0]) / p.cell))
        ny = int(round((p.roi_y[1] - p.roi_y[0]) / p.cell))
        ix = np.floor((pb[:, 0] - p.roi_x[0]) / p.cell).astype(int)
        iy = np.floor((pb[:, 1] - p.roi_y[0]) / p.cell).astype(int)
        inb = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        ix, iy, z = ix[inb], iy[inb], pb[inb, 2]
        col = rgb[ok][inb]
        order = np.argsort(z)                      # ascending: the highest point is written last
        zmap = np.full((ny, nx), np.nan, np.float32)
        cmap = np.zeros((ny, nx, 3), np.uint8)
        zmap[iy[order], ix[order]] = z[order]
        cmap[iy[order], ix[order]] = col[order]
        return zmap, cmap

    @staticmethod
    def _fill_small_holes(zmap: np.ndarray, cmap: np.ndarray, iters: int = 2):
        for _ in range(iters):
            hole = np.isnan(zmap)
            if not hole.any():
                break
            big = np.where(hole, -1e9, zmap).astype(np.float32)
            mx = cv2.dilate(big, np.ones((3, 3), np.uint8))
            fill = hole & (mx > -1e8)
            zmap[fill] = mx[fill]
            for k in range(3):
                cmap[..., k][fill] = cv2.dilate(np.where(hole, 0, cmap[..., k]).astype(np.uint8),
                                                np.ones((3, 3), np.uint8))[fill]
        return zmap, cmap

    @staticmethod
    def _estimate_table_z(zmap: np.ndarray) -> float:
        z = zmap[np.isfinite(zmap)]
        if z.size < 100:
            raise RuntimeError('not enough depth data to estimate the table height')
        lo, hi = np.percentile(z, [1, 60])
        bins = np.arange(lo - 0.01, hi + 0.01, 0.002)
        hist, edges = np.histogram(z, bins=bins)
        k = int(np.argmax(hist))
        sel = z[(z >= edges[k] - 0.002) & (z <= edges[k + 1] + 0.002)]
        return float(np.median(sel))

    # ------------------------------------------------------------------ main entry
    def detect(self, rgb: np.ndarray, depth: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray) -> DetectionResult:
        p = self.p
        pb, ok = self._cloud_base(depth, K, T_base_cam)
        zmap, cmap = self._rasterise(pb, rgb, ok)
        zmap, cmap = self._fill_small_holes(zmap, cmap)
        table_z = p.table_z if p.table_z is not None else self._estimate_table_z(zmap)
        hmap = zmap - table_z                                 # height above the table
        mask = (np.nan_to_num(hmap, nan=0.0) > p.min_height).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        labels = self._components(mask, hmap)
        origin = (p.roi_x[0], p.roi_y[0])
        dets: List[Detection] = []
        rows = []
        for lab in range(1, labels.max() + 1):
            comp = labels == lab
            area_px = int(comp.sum())
            if area_px * p.cell ** 2 < p.min_area_m2:
                labels[comp] = 0
                continue
            d = self._describe(comp, hmap, cmap, table_z, origin)
            d.id = len(dets)
            labels[comp] = d.id + 1                          # keep the label image consistent with ids
            dets.append(d)
        # relabel so label_map holds detection ids (or -1)
        label_map = labels.astype(np.int32) - 1
        # recognition
        if self.catalog is not None and dets:
            for d in dets:
                hue = self._hue_hist(cmap, label_map == d.id)
                d.extra['hue_hist'] = hue
                rows.append(self.catalog.cost_matrix_row(d.size[0], d.size[1], d.height, hue, np.array(d.mean_rgb, float)))
            assigned = self.catalog.assign(rows)
            for d, r, a in zip(dets, rows, assigned):
                best = sorted(r.items(), key=lambda kv: kv[1])[:3]
                d.candidates = best
                d.label = a
                d.label_cost = float(r[a]) if a else float(best[0][1])
        for d in dets:
            d.extra.pop('hue_hist', None)
        return DetectionResult(dets, table_z, hmap.astype(np.float32), cmap, label_map, origin, p.cell)

    # ------------------------------------------------------------------ blobs
    def _components(self, mask: np.ndarray, hmap: np.ndarray) -> np.ndarray:
        n, lab = cv2.connectedComponents(mask, connectivity=8)
        lab = lab.astype(np.int32)
        if not self.p.split_merged:
            return lab
        out = np.zeros_like(lab)
        nxt = 1
        for k in range(1, n):
            comp = (lab == k).astype(np.uint8)
            area = comp.sum()
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hull_area = cv2.contourArea(cv2.convexHull(cnts[0])) if cnts else area
            solidity = area / max(hull_area, 1)
            parts = [comp.astype(bool)]
            if solidity < 0.80 and area * self.p.cell ** 2 > 0.003:
                dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
                peaks = (dist > 0.55 * dist.max()).astype(np.uint8)
                m, markers = cv2.connectedComponents(peaks)
                if m > 2:
                    markers = markers.astype(np.int32)
                    markers[(comp == 0)] = 0
                    bg = (comp == 0)
                    markers[bg] = m + 1
                    img = cv2.cvtColor((np.clip(dist / dist.max(), 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    cv2.watershed(img, markers)
                    cand = [(markers == i) & comp.astype(bool) for i in range(1, m)]
                    if all(c.sum() > 150 for c in cand):
                        parts = cand
            for c in parts:
                out[c] = nxt
                nxt += 1
        return out

    @staticmethod
    def _hue_hist(cmap: np.ndarray, region: np.ndarray, bins: int = 18) -> np.ndarray:
        px = cmap[region]
        if len(px) == 0:
            return np.full(bins, 1.0 / bins)
        hsv = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        use = (hsv[:, 1] > 40) & (hsv[:, 2] > 40)
        if use.sum() < 10:
            use = np.ones(len(hsv), bool)
        h = np.histogram(hsv[use, 0], bins=bins, range=(0, 180))[0].astype(float)
        return h / max(h.sum(), 1)

    def _describe(self, comp: np.ndarray, hmap: np.ndarray, cmap: np.ndarray, table_z: float,
                  origin: Tuple[float, float]) -> Detection:
        c = self.p.cell
        ys, xs = np.nonzero(comp)
        pts = np.stack([xs, ys], 1).astype(np.float32)
        (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
        # rectangle in pixel space (cell grid) -> metres.  Pixel axes == base x (cols) / base y (rows).
        if w < h:
            w, h = h, w
            ang += 90.0
        yaw = wrap_half_pi(math.radians(ang))                 # direction of the long side (cols=x, rows=y)
        length, width = (w + 1) * c, (h + 1) * c
        hv = hmap[comp]
        hv = hv[np.isfinite(hv)]
        top = float(np.percentile(hv, 92)) if hv.size else 0.0
        x = origin[0] + (cx + 0.5) * c
        y = origin[1] + (cy + 0.5) * c
        px = cmap[comp & (np.nan_to_num(hmap, nan=-1.0) > top - 0.03)]
        if len(px) == 0:
            px = cmap[comp]
        rgb = tuple(int(v) for v in px.mean(0))
        x0, y0, bw, bh = cv2.boundingRect(pts.astype(np.int32))
        return Detection(id=-1, label=None, label_cost=9.9, candidates=[], center=(x, y, table_z + top),
                         height=top, size=(length, width), yaw=yaw, mean_rgb=rgb,
                         area_m2=float(comp.sum() * c * c), bbox_px=(x0, y0, bw, bh),
                         color_name=colour_name(rgb))
