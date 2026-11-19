"""Object recognition from size + colour (no training needed).

Every catalogue entry knows its three box dimensions.  Seen from above an object rests on
one of its three axes, so it has three possible (footprint-long, footprint-short, height)
triples; the best of them is compared with the measurement.  Colour (hue histogram of the
visible top surface vs. the texture) adds a smaller term.  Finally a global assignment makes
sure two blobs do not get the same name unless the world really contains two of them.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ClassEntry:
    cid: str
    model: str
    entities: List[str]
    dims: np.ndarray              # model-frame bounding box (x, y, z)
    names: List[str]
    categories: List[str]
    mean_rgb: np.ndarray
    hue_hist: np.ndarray
    sat_hist: np.ndarray

    @property
    def count(self) -> int:
        return max(1, len(self.entities))

    def orientations(self) -> List[Tuple[float, float, float, int]]:
        """(footprint long, footprint short, height, index of the dim that is vertical)."""
        out = []
        for i in range(3):
            fp = sorted([self.dims[j] for j in range(3) if j != i], reverse=True)
            out.append((fp[0], fp[1], self.dims[i], i))
        return out


class Catalog:
    def __init__(self, path: str):
        raw = json.load(open(path, encoding='utf-8'))
        self.entries: Dict[str, ClassEntry] = {}
        for cid, v in raw.items():
            ap = v.get('appearance') or {}
            self.entries[cid] = ClassEntry(
                cid=cid, model=v['model'], entities=v.get('entities', []),
                dims=np.array(v['dims_xyz'], float), names=v['names'],
                categories=v.get('categories', []),
                mean_rgb=np.array(ap.get('mean_rgb', [128, 128, 128]), float),
                hue_hist=np.array(ap.get('hue_hist', [1 / 18] * 18), float),
                sat_hist=np.array(ap.get('sat_hist', [0.25] * 4), float))

    def ids(self) -> List[str]:
        return list(self.entries)

    def display_name(self, cid: str) -> str:
        return self.entries[cid].names[0]

    # ------------------------------------------------------------------ scoring
    @staticmethod
    def _size_cost(meas: Tuple[float, float, float], cand: Tuple[float, float, float]) -> float:
        # log-ratio of the three numbers; bottle tops are smaller than the full footprint, so the
        # height (best measured) is weighted more than the footprint.
        w = (0.8, 1.0, 1.4)
        return sum(wi * abs(math.log(max(m, 1e-3) / max(c, 1e-3))) for wi, m, c in zip(w, meas, cand)) / sum(w)

    @staticmethod
    def _colour_cost(hue_hist: np.ndarray, mean_rgb: np.ndarray, e: ClassEntry) -> float:
        bc = float(np.sum(np.sqrt(np.clip(hue_hist, 0, None) * e.hue_hist)))       # Bhattacharyya coefficient
        c1 = 1.0 - min(1.0, bc)
        a = mean_rgb / max(mean_rgb.sum(), 1.0)
        b = e.mean_rgb / max(e.mean_rgb.sum(), 1.0)
        c2 = float(np.abs(a - b).sum())
        return 0.6 * c1 + 0.4 * c2

    def cost_matrix_row(self, long_: float, short: float, height: float, hue_hist: np.ndarray,
                        mean_rgb: np.ndarray, w_colour: float = 0.35) -> Dict[str, float]:
        res = {}
        for cid, e in self.entries.items():
            size = min(self._size_cost((long_, short, height), (L, S, Ht)) for L, S, Ht, _ in e.orientations())
            res[cid] = size + w_colour * self._colour_cost(hue_hist, mean_rgb, e)
        return res

    def assign(self, rows: Sequence[Dict[str, float]], max_cost: float = 0.9) -> List[Optional[str]]:
        """Hungarian assignment of detections to class slots (a class with N instances has N slots)."""
        from scipy.optimize import linear_sum_assignment
        if not rows:
            return []
        slots = [cid for cid, e in self.entries.items() for _ in range(e.count)]
        n, m = len(rows), len(slots)
        C = np.full((n, m + n), max_cost)                      # n spare "unknown" columns
        for i, r in enumerate(rows):
            for j, cid in enumerate(slots):
                C[i, j] = r[cid]
        ri, ci = linear_sum_assignment(C)
        out: List[Optional[str]] = [None] * n
        for i, j in zip(ri, ci):
            if j < m and C[i, j] < max_cost:
                out[i] = slots[j]
        return out

    # ------------------------------------------------------------------ name lookup
    def match_name(self, text: str) -> List[str]:
        """Class ids whose names/aliases/categories match a free-text phrase (best first)."""
        t = ' ' + ' '.join(text.lower().replace('_', ' ').replace('-', ' ').split()) + ' '
        scored = []
        for cid, e in self.entries.items():
            best = 0.0
            for nm in e.names:
                n = ' ' + nm.lower() + ' '
                if n in t:
                    best = max(best, 1.0 + len(nm) / 40.0)
                else:
                    words = [w for w in nm.lower().split() if len(w) > 2]
                    hit = sum(1 for w in words if ' ' + w in t)
                    if words and hit:
                        best = max(best, 0.5 * hit / len(words))
            if best > 0:
                scored.append((best, cid))
        scored.sort(reverse=True)
        return [c for _, c in scored]
