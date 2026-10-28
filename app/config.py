"""Load and validate config/config.yaml.

Access is attribute-style: cfg.cameras.cam_out.source
Fail loud and early on a missing/wrong key — better than a KeyError deep in the loop.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Union

import yaml

_REQUIRED = (
    "app.target_fps",
    "cameras.cam_out.source",
    "cameras.cam_in.source",
    "counting.cooldown_s",
    "database.sqlite_path",
)
_VALID_SOURCES = {"webcam", "file", "csi"}


def _to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def _get(ns, dotted: str):
    cur = ns
    for part in dotted.split("."):
        if not hasattr(cur, part):
            raise KeyError(f"config missing: {dotted}")
        cur = getattr(cur, part)
    return cur


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


def _deep_merge(base: dict, over: dict) -> dict:
    """Overlay `over` onto `base`: dicts merge key by key, anything else is replaced."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Union[str, Path] = "config/config.yaml", *overrides: Union[str, Path]) -> SimpleNamespace:
    """Load the base config, then overlay each override file in order
    (e.g. config.yaml + jetson.yaml — the overlay only lists what differs)."""
    path = Path(path)
    raw = _read_yaml(path)
    for ov in overrides:
        raw = _deep_merge(raw, _read_yaml(Path(ov)))
    cfg = _to_ns(raw)

    for key in _REQUIRED:
        _get(cfg, key)  # raises if absent

    for cam_name in ("cam_out", "cam_in"):
        cam = getattr(cfg.cameras, cam_name)
        if cam.source not in _VALID_SOURCES:
            raise ValueError(f"cameras.{cam_name}.source={cam.source!r} not in {_VALID_SOURCES}")
        if cam.source == "file" and not getattr(cam, "file_path", None):
            raise ValueError(f"cameras.{cam_name}.source=file but file_path is empty")

    cfg._path = " + ".join(str(p) for p in (path,) + tuple(overrides))
    return cfg
