#!/usr/bin/env python3
"""Entry point.

  python main.py                      # uses config/config.yaml
  python main.py -c config/dev.yaml
  python main.py --check              # load+validate config, then exit (Capaian 0)
"""
from __future__ import annotations

import argparse
import logging
import sys

from app.config import load_config
from app.logging_setup import setup_logging


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default="config/config.yaml")
    ap.add_argument("--check", action="store_true", help="validate config and exit")
    a = ap.parse_args()

    try:
        cfg = load_config(a.config)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2

    setup_logging(cfg.logging.level, cfg.logging.dir, cfg.logging.rotate_mb, cfg.logging.backups)
    log = logging.getLogger("main")
    log.info("config ok: %s", cfg._path)
    log.info("cam_out.source=%s  cam_in.source=%s",
             cfg.cameras.cam_out.source, cfg.cameras.cam_in.source)

    if a.check:
        log.info("--check passed")
        return 0

    from app.pipeline import Pipeline
    try:
        Pipeline(cfg).run()
    except RuntimeError as e:
        log.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
