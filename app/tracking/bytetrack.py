"""Compact ByteTrack.

Two-stage association: high-confidence detections first, then a second pass
that recovers tracks using low-confidence detections (the ByteTrack idea).
Kalman motion model bridges brief gaps; `track_buffer` sets how many frames a
lost track survives before it is dropped — the knob that helps tailgating.

Adapted from the reference ByteTrack (Zhang et al., 2022), trimmed to what this
project needs and wired to app.tracking.base.Track.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .base import Track, Tracker
from .kalman_filter import KalmanFilter


def _xyxy_to_xyah(b):
    x1, y1, x2, y2 = b
    w, h = x2 - x1, y2 - y1
    return np.array([x1 + w / 2, y1 + h / 2, w / max(h, 1e-6), h], dtype=np.float32)


def _xyah_to_xyxy(m):
    cx, cy, a, h = m[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


def _ious(atlbrs, btlbrs):
    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    a = np.ascontiguousarray(atlbrs, dtype=np.float32)
    b = np.ascontiguousarray(btlbrs, dtype=np.float32)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)


def _match(cost, thresh):
    """cost = 1 - IoU. Returns matches, unmatched_a, unmatched_b."""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    matches, ua, ub = [], [], []
    row, col = linear_sum_assignment(cost)
    matched_rows = set()
    matched_cols = set()
    for r, c in zip(row, col):
        if cost[r, c] <= thresh:
            matches.append((r, c))
            matched_rows.add(r)
            matched_cols.add(c)
    ua = [r for r in range(cost.shape[0]) if r not in matched_rows]
    ub = [c for c in range(cost.shape[1]) if c not in matched_cols]
    return matches, ua, ub


class _STrack:
    _kf = KalmanFilter()

    def __init__(self, xyxy, score, cls):
        self.xyxy = np.asarray(xyxy, dtype=np.float32)
        self.score = float(score)
        self.cls = int(cls)
        self.track_id = 0
        self.mean = None
        self.cov = None
        self.state_lost = False
        self.age = 0
        self.hits = 0
        self.time_since_update = 0
        self._traj: list[tuple[float, float]] = []

    def activate(self, tid):
        self.track_id = tid
        self.mean, self.cov = self._kf.initiate(_xyxy_to_xyah(self.xyxy))
        self.state_lost = False
        self.age = self.hits = 1
        self.time_since_update = 0
        self._push()

    def predict(self):
        self.mean, self.cov = self._kf.predict(self.mean, self.cov)
        self.xyxy = _xyah_to_xyxy(self.mean)
        self.age += 1
        self.time_since_update += 1

    def update(self, det_xyxy, score):
        self.mean, self.cov = self._kf.update(self.mean, self.cov, _xyxy_to_xyah(det_xyxy))
        self.xyxy = _xyah_to_xyxy(self.mean)
        self.score = float(score)
        self.state_lost = False
        self.hits += 1
        self.time_since_update = 0
        self._push()

    def _push(self):
        x1, y1, x2, y2 = self.xyxy
        self._traj.append(((x1 + x2) / 2, y1))  # head anchor
        if len(self._traj) > 64:
            self._traj = self._traj[-64:]


class ByteTracker(Tracker):
    def __init__(self, track_buffer: int = 45, match_thresh: float = 0.8,
                 high_thresh: float = 0.6, low_thresh: float = 0.1):
        self.max_lost = int(track_buffer)
        self.match_thresh = float(match_thresh)   # IoU distance threshold (1 - IoU)
        self.high = float(high_thresh)
        self.low = float(low_thresh)
        self._tracked: list[_STrack] = []
        self._lost: list[_STrack] = []
        self._next_id = 0

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def update(self, detections, frame_shape) -> list[Track]:
        H, W = frame_shape[:2]
        boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=np.float32).reshape(-1, 4)
        scores = np.array([d.score for d in detections], dtype=np.float32).reshape(-1)
        clss = np.array([d.cls for d in detections], dtype=np.int32).reshape(-1)

        hi = scores >= self.high
        lo = (scores >= self.low) & (scores < self.high)

        pool = self._tracked + self._lost
        for t in pool:
            t.predict()

        # --- stage 1: high-confidence ---
        dets_hi = [_STrack(boxes[i], scores[i], clss[i]) for i in np.where(hi)[0]]
        cost = 1.0 - _ious([t.xyxy for t in pool], [d.xyxy for d in dets_hi])
        matches, u_track, u_det = _match(cost, self.match_thresh)

        activated: list[_STrack] = []
        for it, idet in matches:
            t = pool[it]
            t.update(dets_hi[idet].xyxy, dets_hi[idet].score)
            activated.append(t)

        # --- stage 2: low-confidence dets vs tracks still unmatched after stage 1 ---
        remain_tracks = [pool[i] for i in u_track]
        dets_lo = [_STrack(boxes[i], scores[i], clss[i]) for i in np.where(lo)[0]]
        cost2 = 1.0 - _ious([t.xyxy for t in remain_tracks], [d.xyxy for d in dets_lo])
        matches2, _u_track2, _ = _match(cost2, 0.5)
        for it, idet in matches2:
            t = remain_tracks[it]
            t.update(dets_lo[idet].xyxy, dets_lo[idet].score)
            activated.append(t)

        # --- new tracks from leftover high-conf dets ---
        new_tracks: list[_STrack] = []
        for idet in u_det:
            d = dets_hi[idet]
            d.activate(self._new_id())
            new_tracks.append(d)

        # --- age out: keep matched, demote the rest to lost until max_lost ---
        act_ids = {id(t) for t in activated}
        new_tracked, new_lost = [], []
        for t in pool:
            if id(t) in act_ids:
                new_tracked.append(t)
            else:
                t.state_lost = True
                if t.time_since_update <= self.max_lost:
                    new_lost.append(t)
        new_tracked.extend(new_tracks)
        self._tracked = new_tracked
        self._lost = new_lost

        out: list[Track] = []
        for t in self._tracked:
            x1 = float(np.clip(t.xyxy[0], 0, W - 1)); y1 = float(np.clip(t.xyxy[1], 0, H - 1))
            x2 = float(np.clip(t.xyxy[2], 0, W - 1)); y2 = float(np.clip(t.xyxy[3], 0, H - 1))
            tr = Track(track_id=t.track_id, x1=x1, y1=y1, x2=x2, y2=y2,
                       score=t.score, cls=t.cls, age=t.age, hits=t.hits,
                       time_since_update=t.time_since_update)
            tr.trajectory.extend(t._traj)
            out.append(tr)
        return out
