"""Single occupancy tally fed by both cameras.

Dedup is deliberately conservative: it only collapses a second event with the
SAME direction and the SAME known idpersonal within dedup_window_s — the case
of one identified person seen by both cameras (or a fragmented track). Two
UNKNOWN crossings are never merged: they are almost always two different people
(the confirm_frames / cooldown / min_track_len guards in CrossingCounter
already suppress jitter from a single track).

inside_set is keyed by idpersonal when known, else "<camera>:<track_id>".
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class Occupancy:
    def __init__(self, dedup_window_s: float = 2.0):
        self.dedup_window_s = float(dedup_window_s)
        self.total_in = 0
        self.total_out = 0
        self.inside: dict[str, float] = {}      # key -> entry ts
        self._recent: list[tuple[str, str, float]] = []  # (direction, idpersonal, ts)

    def key(self, ev) -> str:
        return ev.idpersonal or f"{ev.camera_id}:{ev.track_id}"

    def _dup(self, ev) -> bool:
        self._recent = [(d, i, t) for (d, i, t) in self._recent if ev.ts - t < self.dedup_window_s]
        if ev.idpersonal is not None:
            for d, i, _t in self._recent:
                if d == ev.direction and i == ev.idpersonal:
                    return True
        self._recent.append((ev.direction, ev.idpersonal, ev.ts))
        return False

    def apply(self, ev) -> bool:
        """Returns True if the event was counted, False if deduped."""
        if self._dup(ev):
            log.debug("occupancy: deduped %s %s", ev.direction, self.key(ev))
            return False
        key = self.key(ev)
        if ev.direction == "IN":
            self.total_in += 1
            self.inside[key] = ev.ts
        else:
            self.total_out += 1
            self.inside.pop(key, None)
        return True

    def load(self, keys) -> None:
        """Restore the live tally from occupancy_state on restart. `keys` is an
        iterable of (key, entered_at)."""
        for key, entered in keys:
            self.inside[key] = entered
        if self.inside:
            log.info("occupancy restored: %d already inside", len(self.inside))

    @property
    def current(self) -> int:
        return len(self.inside)

    def summary(self) -> str:
        return (f"inside={self.current}  IN={self.total_in}  OUT={self.total_out}")
