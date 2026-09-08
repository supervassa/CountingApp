"""One place to configure logging: console + rotating file."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(level: str = "INFO", log_dir: str = "logs",
                  rotate_mb: int = 20, backups: int = 5) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s", "%H:%M:%S")

    con = logging.StreamHandler()
    con.setFormatter(fmt)
    root.addHandler(con)

    fh = RotatingFileHandler(Path(log_dir) / "app.log",
                             maxBytes=rotate_mb * 1024 * 1024, backupCount=backups)
    fh.setFormatter(fmt)
    root.addHandler(fh)
