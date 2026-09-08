"""Cross-check the two cameras against the shared occupancy (PRD §17).

cam_out only emits IN, cam_in only emits OUT, so the running occupancy is
Σ IN − Σ OUT. Consistency alarms:

  negative_occupancy  : an OUT with no matching IN -> cam_out missed an entry
  occupancy_too_high  : occupancy above a plausible ceiling -> missed exits
  camera_silent       : one camera has produced no event for a long stretch
                        while the other is active -> shifted / blocked / dead

Alarms are rate-limited per kind and handed to a sink (SQLite + WARNING log).
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class CrossCheck:
    def __init__(self, sink, *, max_plausible: int = 200, silent_alarm_s: float = 900.0,
                 realarm_s: float = 300.0):
        self.sink = sink                          # callable(kind, detail)
        self.max_plausible = int(max_plausible)
        self.silent_alarm_s = float(silent_alarm_s)
        self.realarm_s = float(realarm_s)
        self.last_event_ts = {"cam_out": None, "cam_in": None}
        self._last_alarm_ts: dict[str, float] = {}

    def _raise(self, kind: str, detail: str, now: float):
        prev = self._last_alarm_ts.get(kind)
        if prev is not None and now - prev < self.realarm_s:
            return
        self._last_alarm_ts[kind] = now
        log.warning("CROSSCHECK %s: %s", kind, detail)
        try:
            self.sink(kind, detail)
        except Exception:
            log.exception("alarm sink failed")

    def on_event(self, ev, occupancy_now: int):
        self.last_event_ts[ev.camera_id] = ev.ts
        if occupancy_now < 0:
            self._raise("negative_occupancy",
                        f"occupancy={occupancy_now} after {ev.camera_id} {ev.direction} "
                        f"#{ev.track_id} — an entry was missed", ev.ts)
        elif occupancy_now > self.max_plausible:
            self._raise("occupancy_too_high",
                        f"occupancy={occupancy_now} > {self.max_plausible} — exits are being missed", ev.ts)

    def tick(self, now: float):
        """Call once per loop; catches a silent camera even with no events."""
        seen = [t for t in self.last_event_ts.values() if t is not None]
        if len(seen) < 2:
            return
        other_active = max(seen)
        if now - other_active > self.silent_alarm_s:
            return  # both quiet — nothing to compare
        for cam, ts in self.last_event_ts.items():
            if ts is not None and now - ts > self.silent_alarm_s:
                self._raise("camera_silent",
                            f"{cam} silent {now - ts:.0f}s while the other camera is active", now)
