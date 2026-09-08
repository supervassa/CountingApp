"""Config loader sanity — runnable with `python -m pytest` or plain `python tests/test_config.py`."""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import load_config  # noqa: E402


def _write(tmp: Path, body: str) -> Path:
    p = tmp / "c.yaml"
    p.write_text(textwrap.dedent(body))
    return p


MINIMAL = """
    app: {target_fps: 8, display: false}
    logging: {level: INFO, dir: logs, rotate_mb: 20, backups: 5}
    cameras:
      cam_out: {role: in, source: webcam, webcam_index: 0, width: 1280, height: 720, fps: 30, flip_method: 0}
      cam_in:  {role: out, source: webcam, webcam_index: 1, width: 1280, height: 720, fps: 30, flip_method: 0}
    counting: {cooldown_s: 2.0}
    database: {sqlite_path: data/events.db}
"""


def test_loads_minimal(tmp_path):
    cfg = load_config(_write(tmp_path, MINIMAL))
    assert cfg.cameras.cam_out.source == "webcam"
    assert cfg.counting.cooldown_s == 2.0


def test_rejects_bad_source(tmp_path):
    bad = MINIMAL.replace("source: webcam, webcam_index: 0", "source: potato, webcam_index: 0")
    try:
        load_config(_write(tmp_path, bad))
    except ValueError:
        return
    raise AssertionError("expected ValueError for bad source")


def test_rejects_file_without_path(tmp_path):
    bad = MINIMAL.replace("cam_in:  {role: out, source: webcam, webcam_index: 1",
                          "cam_in:  {role: out, source: file, file_path: null, webcam_index: 1")
    try:
        load_config(_write(tmp_path, bad))
    except ValueError:
        return
    raise AssertionError("expected ValueError for file source without file_path")


if __name__ == "__main__":
    import tempfile

    d = Path(tempfile.mkdtemp())
    test_loads_minimal(d)
    test_rejects_bad_source(d)
    test_rejects_file_without_path(d)
    print("ok")
