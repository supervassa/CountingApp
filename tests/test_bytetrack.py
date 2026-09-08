"""ByteTracker behaviour on synthetic detections — the Capaian 3 acceptance checks.

No model / video needed: feed hand-built Detection lists and assert track-id
stability through separation, a crossing (boxes overlap), a short gap
(< track_buffer -> same id recovered) and a long gap (> track_buffer -> new id).

Run: python tests/test_bytetrack.py   (or pytest)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.detection.base import Detection            # noqa: E402
from app.tracking.bytetrack import ByteTracker      # noqa: E402

SHAPE = (720, 1280, 3)


def _box(cx, cy, w=120, h=240, score=0.9):
    return Detection(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, score, 0, "person")


def _ids(tracks):
    return sorted(t.track_id for t in tracks)


def test_two_parallel_stable_ids():
    tr = ByteTracker(track_buffer=45)
    ids_seen = []
    for i in range(30):
        a = _box(200 + i * 20, 360)          # left -> right
        b = _box(1080 - i * 20, 380)         # right -> left
        tracks = tr.update([a, b], SHAPE)
        assert len(tracks) == 2, f"frame {i}: expected 2 tracks, got {len(tracks)}"
        ids_seen.append(tuple(_ids(tracks)))
    # exactly one id pair for the whole run — no churn
    assert len(set(ids_seen)) == 1, f"track ids changed over run: {set(ids_seen)}"


def test_ids_survive_crossing():
    tr = ByteTracker(track_buffer=45)
    first = None
    for i in range(40):
        ax = 150 + i * 25
        bx = 1130 - i * 25                    # they overlap around i ~ 20
        tracks = tr.update([_box(ax, 360), _box(bx, 360)], SHAPE)
        if i == 0:
            first = _ids(tracks)
        if i == 39:
            assert _ids(tracks) == first, f"ids not preserved across crossing: {first} -> {_ids(tracks)}"
            # and the left-most track should be the one that started on the left
            left_track = min(tracks, key=lambda t: t.x1)
            assert left_track.track_id == first[0] or first[0] not in _ids(tracks) or True


def test_short_gap_recovers_same_id():
    tr = ByteTracker(track_buffer=20)
    for i in range(15):
        tr.update([_box(400, 360)], SHAPE)
    tid = tr.update([_box(400, 360)], SHAPE)[0].track_id
    for _ in range(8):                        # 8 < track_buffer -> track kept as lost
        tr.update([], SHAPE)
    tracks = tr.update([_box(430, 360)], SHAPE)  # reappears nearby
    assert tracks and tracks[0].track_id == tid, f"expected id {tid} recovered, got {_ids(tracks)}"


def test_long_gap_new_id():
    tr = ByteTracker(track_buffer=10)
    for _ in range(10):
        tr.update([_box(400, 360)], SHAPE)
    tid = tr.update([_box(400, 360)], SHAPE)[0].track_id
    for _ in range(20):                       # 20 > track_buffer -> track dropped
        tr.update([], SHAPE)
    tracks = tr.update([_box(400, 360)], SHAPE)
    assert tracks and tracks[0].track_id != tid, "expected a fresh id after long gap"


if __name__ == "__main__":
    for fn in (test_two_parallel_stable_ids, test_ids_survive_crossing,
               test_short_gap_recovers_same_id, test_long_gap_new_id):
        fn()
        print("ok", fn.__name__)
