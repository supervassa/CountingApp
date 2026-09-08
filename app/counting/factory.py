"""Build per-camera CrossingCounter + the shared Occupancy / CrossCheck from config."""
from __future__ import annotations

from .band import Band
from .counter import CrossingCounter
from .crosscheck import CrossCheck
from .occupancy import Occupancy


def build_band(cfg_counting, cam_id: str) -> Band:
    node = getattr(cfg_counting.band, cam_id)
    return Band(node.outer, node.inner)


def build_counter(cfg_counting, cam_id: str, role: str) -> CrossingCounter:
    return CrossingCounter(
        camera_id=cam_id,
        role=role,
        band=build_band(cfg_counting, cam_id),
        anchor=getattr(cfg_counting, "crossing_anchor", "head"),
        min_track_len=getattr(cfg_counting, "min_track_len", 5),
        confirm_frames=getattr(cfg_counting, "confirm_frames", 3),
        cooldown_s=getattr(cfg_counting, "cooldown_s", 2.0),
    )


def build_occupancy(cfg_counting) -> Occupancy:
    return Occupancy(dedup_window_s=getattr(cfg_counting, "dedup_window_s", 2.0))


def build_crosscheck(cfg_counting, sink) -> CrossCheck:
    node = getattr(cfg_counting, "crosscheck", None)
    return CrossCheck(
        sink,
        max_plausible=getattr(node, "max_plausible", 200) if node else 200,
        silent_alarm_s=getattr(node, "silent_alarm_s", 900.0) if node else 900.0,
        realarm_s=getattr(node, "realarm_s", 300.0) if node else 300.0,
    )
