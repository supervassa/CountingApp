"""Build a per-camera Tracker from config."""
from __future__ import annotations

from .base import Tracker
from .bytetrack import ByteTracker


def build_tracker(node) -> Tracker:
    return ByteTracker(
        track_buffer=getattr(node, "track_buffer", 45),
        match_thresh=getattr(node, "match_thresh", 0.8),
        high_thresh=getattr(node, "high_thresh", 0.6),
        low_thresh=getattr(node, "low_thresh", 0.1),
    )
