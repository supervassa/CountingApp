"""CrossingCounter + Occupancy — the Capaian 5 acceptance checks.

Synthetic tracks walked across a band; assert one crossing = one event, the
direction filter, per-track cooldown (not global), and min_track_len.

Run: python tests/test_counting.py   (or pytest)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.counting.band import Band                       # noqa: E402
from app.counting.counter import CrossingCounter         # noqa: E402
from app.counting.occupancy import Occupancy             # noqa: E402
from app.tracking.base import Track                       # noqa: E402

SHAPE = (720, 1280, 3)
OUTER = [0.0, 0.35, 1.0, 0.35]   # y ~ 252
INNER = [0.0, 0.62, 1.0, 0.62]   # y ~ 446


def _band():
    return Band(OUTER, INNER)


def _track(tid, cy_top, age):
    # head anchor = top-centre; box 120x240
    return Track(track_id=tid, x1=600, y1=cy_top, x2=720, y2=cy_top + 240, score=0.9, age=age)


def _walk(counter, tid, ys, ts0=0.0, dt=0.1):
    """Feed one track through a list of head-y positions; return all events."""
    evs = []
    for i, y in enumerate(ys):
        evs += counter.update([_track(tid, y, age=10 + i)], SHAPE, ts0 + i * dt)
    return evs


def test_one_crossing_one_in_event():
    c = CrossingCounter("cam_out", "in", _band(), confirm_frames=3, min_track_len=5)
    ys = [100] * 4 + [200, 280, 360, 440] + [520] * 4   # OUTSIDE -> band -> INSIDE
    evs = _walk(c, 1, ys)
    assert [e.direction for e in evs] == ["IN"], [e.direction for e in evs]


def test_direction_filter_drops_wrong_way():
    c = CrossingCounter("cam_out", "in", _band(), confirm_frames=3, min_track_len=5)
    ys = [520] * 4 + [440, 360, 280, 200] + [100] * 4   # INSIDE -> OUTSIDE on an IN-only camera
    assert _walk(c, 1, ys) == []


def test_out_camera_emits_out():
    c = CrossingCounter("cam_in", "out", _band(), confirm_frames=3, min_track_len=5)
    ys = [520] * 4 + [440, 360, 280, 200] + [100] * 4
    assert [e.direction for e in _walk(c, 1, ys)] == ["OUT"]


def test_loiter_no_event():
    c = CrossingCounter("cam_out", "in", _band(), confirm_frames=3, min_track_len=5)
    ys = [120, 150, 130, 160, 140, 150, 120, 160, 140] * 2   # never leaves OUTSIDE
    assert _walk(c, 1, ys) == []


def test_two_tracks_simultaneous_two_events():
    c = CrossingCounter("cam_out", "in", _band(), confirm_frames=3, min_track_len=5)
    ys = [100] * 4 + [200, 280, 360, 440] + [520] * 4
    evs = []
    for i, y in enumerate(ys):
        evs += c.update([_track(1, y, 10 + i), _track(2, y + 5, 10 + i)], SHAPE, i * 0.1)
    got = sorted((e.track_id, e.direction) for e in evs)
    assert got == [(1, "IN"), (2, "IN")], got


def test_cooldown_blocks_immediate_recross():
    c = CrossingCounter("cam_out", "in", _band(), confirm_frames=3, min_track_len=5, cooldown_s=2.0)
    # IN, then immediately walk back out and in again within 2 s -> 2nd IN suppressed
    ys = ([100] * 4 + [200, 300, 440] + [520] * 4          # IN  @ ~t=1.0
          + [440, 300, 200] + [100] * 4                     # back to OUTSIDE
          + [200, 300, 440] + [520] * 4)                    # would-be 2nd IN, still < 2 s
    evs = _walk(c, 1, ys, dt=0.1)
    assert [e.direction for e in evs] == ["IN"], [e.direction for e in evs]


def test_min_track_len_gate():
    c = CrossingCounter("cam_out", "in", _band(), confirm_frames=3, min_track_len=50)
    ys = [100] * 4 + [200, 280, 360, 440] + [520] * 4
    assert _walk(c, 1, ys) == []   # track never old enough


def test_occupancy_dedup_known_person_only():
    occ = Occupancy(dedup_window_s=2.0)
    from app.counting.counter import CrossEvent
    # same idpersonal, same direction, within window -> deduped
    a = CrossEvent("cam_out", "IN", 1, ts=10.0, idpersonal="juris")
    b = CrossEvent("cam_in", "IN", 9, ts=10.5, idpersonal="juris")
    assert occ.apply(a) is True
    assert occ.apply(b) is False
    assert occ.total_in == 1 and occ.current == 1


def test_occupancy_two_unknowns_not_merged():
    occ = Occupancy(dedup_window_s=2.0)
    from app.counting.counter import CrossEvent
    a = CrossEvent("cam_out", "IN", 1, ts=10.0)   # UNKNOWN
    b = CrossEvent("cam_out", "IN", 2, ts=10.8)   # different person, still UNKNOWN
    assert occ.apply(a) is True
    assert occ.apply(b) is True
    assert occ.total_in == 2 and occ.current == 2


if __name__ == "__main__":
    import types
    mod = sys.modules[__name__]
    for n in [x for x in dir(mod) if x.startswith("test_")]:
        getattr(mod, n)()
        print("ok", n)
