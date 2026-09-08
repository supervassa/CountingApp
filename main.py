#!/usr/bin/env python3
"""Entry point.

  python main.py                      # run the pipeline (uses config/config.yaml)
  python main.py -c config/dev.yaml
  python main.py --check              # load+validate config, then exit
  python main.py --summary [DATE]     # print the daily people-flow summary (default: today)
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime

from app.config import load_config
from app.logging_setup import setup_logging


def _print_summary(cfg, date: str | None) -> int:
    from app.database.factory import build_repository

    repo = build_repository(cfg.database)
    try:
        date = date or datetime.now().strftime("%Y-%m-%d")
        rows = repo.daily_summary(date)
        occ = repo.occupancy()
        anomalies = repo.anomalies()
        alarms = repo.alarms_on(date)

        print(f"\n================ DAILY PEOPLE FLOW — {date} ================")
        if not rows:
            print("  (no events)")
        for r in rows:
            who = r["name"] or (r["idpersonal"][:8] if r["idpersonal"] else r["identity"])
            status = "-" if not r["idpersonal"] else ("INSIDE" if r["inside"] else "OUTSIDE")
            print(f"  {who:<24}  IN {r['in']:>3}   OUT {r['out']:>3}   {status}")
        print("  " + "-" * 52)
        print(f"  Current occupancy : {occ['known']} known + {occ['unknown']} unknown = {occ['total']}")

        if anomalies:
            print(f"\n  ANOMALY — still marked inside ({len(anomalies)}): "
                  "exit not seen, or stayed past close")
            for a in anomalies:
                who = a["name"] or a["idpersonal"] or a["key"]
                print(f"    {who:<28}  since {a['entered_at']}")

        if alarms:
            print(f"\n  CROSS-CHECK ALARMS ({len(alarms)}):")
            for al in alarms:
                print(f"    {al['ts']}  {al['kind']}  {al['detail']}")
        print()
        return 0
    finally:
        repo.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default="config/config.yaml")
    ap.add_argument("--check", action="store_true", help="validate config and exit")
    ap.add_argument("--summary", nargs="?", const="", metavar="DATE",
                    help="print the daily summary (YYYY-MM-DD; default today) and exit")
    a = ap.parse_args()

    try:
        cfg = load_config(a.config)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2

    setup_logging(cfg.logging.level, cfg.logging.dir, cfg.logging.rotate_mb, cfg.logging.backups)
    log = logging.getLogger("main")
    log.info("config ok: %s", cfg._path)

    if a.summary is not None:
        return _print_summary(cfg, a.summary or None)

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
