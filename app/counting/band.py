"""Counting band: two lines, direction from the order they are crossed.

Config gives each line as (x1, y1, x2, y2) in frame fractions. The axis from
the outer-line midpoint to the inner-line midpoint defines OUTSIDE -> INSIDE
progress, so the band works horizontal, vertical or diagonal.
"""
from __future__ import annotations

from enum import Enum

import numpy as np

OUTSIDE, BAND, INSIDE = "OUTSIDE", "BAND", "INSIDE"


class Zone(str, Enum):
    OUTSIDE = OUTSIDE
    BAND = BAND
    INSIDE = INSIDE


def _mid(line):
    x1, y1, x2, y2 = line
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float64)


class Band:
    def __init__(self, outer_frac, inner_frac):
        self.outer_frac = list(outer_frac)
        self.inner_frac = list(inner_frac)
        self._px = None  # (Mo, axis_hat, length) cached per frame size

    def _prep(self, w: int, h: int):
        s = np.array([w, h, w, h], dtype=np.float64)
        mo = _mid(np.array(self.outer_frac) * s)
        mi = _mid(np.array(self.inner_frac) * s)
        axis = mi - mo
        length = float(np.linalg.norm(axis))
        axis_hat = axis / length if length > 1e-6 else np.array([0.0, 1.0])
        self._px = (mo, axis_hat, length)
        return self._px

    def progress(self, point, frame_shape) -> float:
        h, w = frame_shape[:2]
        mo, axis_hat, _ = self._prep(w, h)
        q = np.asarray(point, dtype=np.float64)
        return float(np.dot(q - mo, axis_hat))

    def zone(self, point, frame_shape) -> str:
        h, w = frame_shape[:2]
        mo, axis_hat, length = self._prep(w, h)
        p = float(np.dot(np.asarray(point, dtype=np.float64) - mo, axis_hat))
        if p < 0:
            return OUTSIDE
        if p > length:
            return INSIDE
        return BAND

    def line_px(self, frac, w: int, h: int):
        x1, y1, x2, y2 = frac
        return (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
