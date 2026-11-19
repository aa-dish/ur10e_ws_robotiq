"""Offline RGB-D 'camera' for developing / testing the vision code without Gazebo.

Samples the object meshes of the world file, places them with the world poses, and
z-buffer-splats them into a depth + colour image seen by a pinhole camera that matches
the simulated RealSense D435 (see d435_camera.urdf.xacro).  Not photo-realistic - it is
only meant to test geometry (segmentation, poses, grasp planning).
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

import numpy as np

from ..geometry import inv_T, make_T, rpy_to_matrix

ROBOT_BASE_WORLD = np.array([0.842175, -0.500138, 0.6])
TABLE_TOP_WORLD_Z = 0.727
CAM_XYZ = np.array([0.006, -0.62, 1.25])                 # base_link frame
# optical frame axes (columns) expressed in base_link: x right=+x, y down=-y, z fwd=-z
CAM_R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)


def camera_intrinsics(width=640, height=480, hfov=1.2):
    fx = (width / 2) / np.tan(hfov / 2)
    return np.array([[fx, 0, width / 2], [0, fx, height / 2], [0, 0, 1]], float)


def parse_world(world_path: str):
    s = open(world_path, encoding='utf-8').read()
    out = []
    for m in re.finditer(r"<include>\s*<uri>([^<]+)</uri>\s*<name>(\w+)</name>\s*<pose>([^<]+)</pose>", s):
        uri, name, pose = m.groups()
        out.append(dict(model=uri.rstrip('/').split('/')[-1], name=name, pose=[float(x) for x in pose.split()]))
    return out


class SyntheticScene:
    def __init__(self, gazebo_dir: str, width=640, height=480, hfov=1.2, seed=0, points_per_m2=2.5e6):
        import trimesh
        self.trimesh = trimesh
        self.gz = gazebo_dir
        self.W, self.H = width, height
        self.K = camera_intrinsics(width, height, hfov)
        self.T_base_cam = make_T(CAM_R, CAM_XYZ)
        self.rng = np.random.default_rng(seed)
        self.ppm2 = points_per_m2
        self.objects = [o for o in parse_world(os.path.join(gazebo_dir, 'worlds', 'bin_picking_world.sdf'))
                        if o['model'] not in ('Desk', 'FoodCourtTable1', 'TrashBin')]
        self._cache: Dict[str, tuple] = {}

    def _samples(self, model: str):
        if model in self._cache:
            return self._cache[model]
        tm = self.trimesh
        d = os.path.join(self.gz, 'models', model)
        sdf = open(os.path.join(d, 'model.sdf'), encoding='utf-8').read()
        uris = re.findall(r'<uri>([^<]+)</uri>', sdf)
        f = ([u for u in uris if u.endswith('.obj')] or [u.replace('.dae', '.obj') for u in uris])[0]
        import glob, shutil, tempfile
        src = os.path.join(d, f)
        tmp = tempfile.mkdtemp(prefix='synth_')          # trimesh refuses '../' texture paths -> flatten
        for fn in glob.glob(os.path.join(os.path.dirname(src), '*')):
            shutil.copy(fn, tmp)
        for tex in glob.glob(os.path.join(d, 'materials', 'textures', '*.png')):
            shutil.copy(tex, tmp)
        for mtl in glob.glob(os.path.join(tmp, '*.mtl')):
            t = open(mtl, encoding='utf-8').read()
            t = re.sub(r'(map_Kd\s+)\S*?([^/\s]+)\s*$', r'\1\2', t, flags=re.M)
            open(mtl, 'w', encoding='utf-8').write(t)
        mesh = tm.load(os.path.join(tmp, os.path.basename(src)), force='mesh', process=False)
        shutil.rmtree(tmp, ignore_errors=True)
        n = int(max(20000, min(250000, mesh.area * self.ppm2)))
        pts, fidx, col = tm.sample.sample_surface(mesh, n, sample_color=True, seed=1)
        col = np.asarray(col)[:, :3] if col is not None and len(col) else np.full((n, 3), 128, np.uint8)
        self._cache[model] = (np.asarray(pts), col.astype(np.uint8))
        return self._cache[model]

    def object_pose_base(self, o) -> np.ndarray:
        x, y, z, r, p, yw = o['pose']
        T = make_T(rpy_to_matrix(r, p, yw), np.array([x, y, z]) - ROBOT_BASE_WORLD)
        return T

    def render(self, moved: Optional[Dict[str, np.ndarray]] = None, exclude: Optional[List[str]] = None):
        """Returns rgb (H,W,3 uint8), depth (H,W float32, NaN = no return)."""
        moved = moved or {}
        exclude = set(exclude or [])
        P, C = [], []
        # table plane
        zt = TABLE_TOP_WORLD_Z - ROBOT_BASE_WORLD[2]
        gx, gy = np.meshgrid(np.arange(-0.47, 0.48, 0.002), np.arange(-0.93, -0.30, 0.002))
        tp = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, zt)], 1)
        P.append(tp)
        C.append(np.tile(np.array([225, 222, 215], np.uint8), (len(tp), 1)))
        for o in self.objects:
            if o['name'] in exclude:
                continue
            pts, col = self._samples(o['model'])
            T = moved.get(o['name'], self.object_pose_base(o))
            P.append((T[:3, :3] @ pts.T).T + T[:3, 3])
            C.append(col)
        P = np.vstack(P)
        C = np.vstack(C)
        Pc = (inv_T(self.T_base_cam)[:3, :3] @ P.T).T + inv_T(self.T_base_cam)[:3, 3]
        z = Pc[:, 2]
        ok = z > 0.2
        u = np.round(self.K[0, 0] * Pc[ok, 0] / z[ok] + self.K[0, 2]).astype(int)
        v = np.round(self.K[1, 1] * Pc[ok, 1] / z[ok] + self.K[1, 2]).astype(int)
        zz, cc = z[ok], C[ok]
        inb = (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
        u, v, zz, cc = u[inb], v[inb], zz[inb], cc[inb]
        order = np.argsort(-zz)                      # far first, near overwrites
        depth = np.full((self.H, self.W), np.nan, np.float32)
        rgb = np.zeros((self.H, self.W, 3), np.uint8)
        depth[v[order], u[order]] = zz[order]
        rgb[v[order], u[order]] = cc[order]
        # close 1-pixel holes (min-filter of the neighbours)
        import cv2
        hole = np.isnan(depth)
        if hole.any():
            big = np.where(hole, 1e9, depth).astype(np.float32)
            mn = cv2.erode(big, np.ones((3, 3), np.uint8))
            fill = hole & (mn < 1e8)
            depth[fill] = mn[fill]
            for k in range(3):
                src = np.where(hole, 0, rgb[..., k]).astype(np.uint8)
                rgb[..., k][fill] = cv2.dilate(src, np.ones((3, 3), np.uint8))[fill]
        return rgb, depth
