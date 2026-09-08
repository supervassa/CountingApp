"""Per-camera crossing counter.

Per track: a small state machine over the band zone. A confirmed OUTSIDE->INSIDE
is an IN crossing, INSIDE->OUTSIDE is an OUT crossing. Guards:
  - min_track_len : track must have existed this many frames
  - confirm_frames: the new solid zone must hold this many consecutive frames
  - cooldown_s    : per track_id, no second event within this window
  - direction filter: cam_out emits IN only, cam_in emits OUT only
A track first seen already INSIDE/OUTSIDE just seeds last_solid — no event.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .band import Band, BAND, INSIDE, OUTSIDE

log = logging.getLogger(__name__)


@dataclass
class CrossEvent:
    camera_id: str
    direction: str          # "IN" | "OUT"
    track_id: int
    ts: float
    identity: str = "UNKNOWN"
    idpersonal: str | None = None
    similarity: float | None = None


class _TrackState:
    __slots__ = ("last_solid", "cand_zone", "cand_count", "last_event_ts")

    def __init__(self):
        self.last_solid: str | None = None
        self.cand_zone: str | None = None
        self.cand_count = 0
        self.last_event_ts = float("-inf")


class CrossingCounter:
    def __init__(self, camera_id: str, role: str, band: Band, *,
                 anchor: str = "head", min_track_len: int = 5,
                 confirm_frames: int = 3, cooldown_s: float = 2.0):
        self.camera_id = camera_id
        self.allowed = "IN" if role == "in" else "OUT"
        self.band = band
        self.anchor = anchor
        self.min_track_len = int(min_track_len)
        self.confirm_frames = int(confirm_frames)
        self.cooldown_s = float(cooldown_s)
        self._st: dict[int, _TrackState] = {}

    def _point(self, track):
        return track.anchor_head if self.anchor == "head" else track.anchor_centroid

    def update(self, tracks, frame_shape, ts: float) -> list[CrossEvent]:
        events: list[CrossEvent] = []
        live = {t.track_id for t in tracks}
        for dead in [tid for tid in self._st if tid not in live]:
            del self._st[dead]

        for t in tracks:
            st = self._st.setdefault(t.track_id, _TrackState())
            z = self.band.zone(self._point(t), frame_shape)

            # debounce the incoming solid zone
            if z == BAND:
                st.cand_zone, st.cand_count = None, 0
                continue
            if z == st.cand_zone:
                st.cand_count += 1
            else:
                st.cand_zone, st.cand_count = z, 1
            if st.cand_count < self.confirm_frames:
                continue

            # z is now a confirmed solid zone
            if st.last_solid is None:
                st.last_solid = z
                continue
            if z == st.last_solid:
                continue

            direction = "IN" if (st.last_solid == OUTSIDE and z == INSIDE) else \
                        "OUT" if (st.last_solid == INSIDE and z == OUTSIDE) else None
            st.last_solid = z
            if direction is None:
                continue
            if t.age < self.min_track_len:
                log.debug("[%s] #%d %s dropped: track too young (%d)", self.camera_id, t.track_id, direction, t.age)
                continue
            if ts - st.last_event_ts < self.cooldown_s:
                log.debug("[%s] #%d %s dropped: cooldown", self.camera_id, t.track_id, direction)
                continue
            if direction != self.allowed:
                log.debug("[%s] #%d %s dropped: camera only emits %s", self.camera_id, t.track_id, direction, self.allowed)
                st.last_event_ts = ts
                continue

            st.last_event_ts = ts
            ev = CrossEvent(self.camera_id, direction, t.track_id, ts)
            events.append(ev)
            log.info("[%s] CROSS #%d -> %s", self.camera_id, t.track_id, direction)
        return events
