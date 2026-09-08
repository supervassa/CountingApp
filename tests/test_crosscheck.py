"""CrossCheck alarms — Capaian 9 acceptance checks.

Run: python tests/test_crosscheck.py   (or pytest)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.counting.counter import CrossEvent          # noqa: E402
from app.counting.crosscheck import CrossCheck        # noqa: E402


class _Sink:
    def __init__(self):
        self.rows = []

    def __call__(self, kind, detail):
        self.rows.append((kind, detail))


def test_negative_occupancy_alarms():
    s = _Sink()
    cc = CrossCheck(s, realarm_s=0)
    cc.on_event(CrossEvent("cam_in", "OUT", 1, ts=1.0), occupancy_now=-1)
    assert [k for k, _ in s.rows] == ["negative_occupancy"]


def test_occupancy_too_high_alarms():
    s = _Sink()
    cc = CrossCheck(s, max_plausible=10, realarm_s=0)
    cc.on_event(CrossEvent("cam_out", "IN", 1, ts=1.0), occupancy_now=11)
    assert [k for k, _ in s.rows] == ["occupancy_too_high"]


def test_healthy_flow_no_alarm():
    s = _Sink()
    cc = CrossCheck(s)
    for i in range(20):
        cc.on_event(CrossEvent("cam_out", "IN", i, ts=float(i)), occupancy_now=1)
        cc.on_event(CrossEvent("cam_in", "OUT", i, ts=i + 0.5), occupancy_now=0)
    assert s.rows == []


def test_camera_silent_alarms():
    s = _Sink()
    cc = CrossCheck(s, silent_alarm_s=100.0, realarm_s=0)
    cc.on_event(CrossEvent("cam_out", "IN", 1, ts=0.0), occupancy_now=1)
    cc.on_event(CrossEvent("cam_in", "OUT", 1, ts=1000.0), occupancy_now=0)  # cam_in active now
    cc.tick(now=1010.0)   # cam_out last spoke at t=0, 1010s ago, cam_in active
    assert [k for k, _ in s.rows] == ["camera_silent"]
    assert "cam_out" in s.rows[0][1]


def test_rate_limited_per_kind():
    s = _Sink()
    cc = CrossCheck(s, max_plausible=10, realarm_s=300.0)
    cc.on_event(CrossEvent("cam_out", "IN", 1, ts=1.0), occupancy_now=11)
    cc.on_event(CrossEvent("cam_out", "IN", 2, ts=2.0), occupancy_now=12)   # < 300 s later
    cc.on_event(CrossEvent("cam_out", "IN", 3, ts=400.0), occupancy_now=13) # > 300 s later
    assert [k for k, _ in s.rows] == ["occupancy_too_high", "occupancy_too_high"]


def test_no_alarm_before_both_cameras_seen():
    s = _Sink()
    cc = CrossCheck(s, silent_alarm_s=1.0)
    cc.on_event(CrossEvent("cam_out", "IN", 1, ts=0.0), occupancy_now=1)
    cc.tick(now=100.0)   # only one camera has ever spoken -> cannot compare
    assert s.rows == []


if __name__ == "__main__":
    mod = sys.modules[__name__]
    for n in [x for x in dir(mod) if x.startswith("test_")]:
        getattr(mod, n)()
        print("ok", n)
