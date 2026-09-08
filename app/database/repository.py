"""SQLite access — schema init, event writes, daily rollups, persons_cache.

The capture loop only writes events + occupancy_state here (fast, local). Name
resolution is done out-of-band by EnrichmentWorker into persons_cache.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)
_SCHEMA = Path(__file__).with_name("schema.sql")


def _now_iso() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


class Repository:
    def __init__(self, sqlite_path: str):
        self.path = sqlite_path
        p = Path(sqlite_path)
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(sqlite_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA.read_text())
        self.conn.commit()
        log.info("sqlite ready: %s", sqlite_path)

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    # --- events ---
    def insert_event(self, ev, *, ts: datetime | None = None,
                     all_scores=None, image_path: str | None = None) -> int:
        when = (ts or datetime.now()).astimezone()
        row = (
            ev.idpersonal, ev.identity, ev.direction, ev.camera_id,
            None, ev.similarity,
            json.dumps(all_scores) if all_scores is not None else None,
            ev.track_id, when.replace(microsecond=0).isoformat(),
            when.strftime("%Y-%m-%d"), image_path,
        )
        cur = self.conn.execute(
            """INSERT INTO events
               (idpersonal, identity, event_type, camera_id, confidence, similarity,
                all_scores, track_id, timestamp, date, image_path)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""", row)
        self.conn.commit()
        return cur.lastrowid

    # --- occupancy_state (survives restart) ---
    def set_inside(self, key: str, idpersonal: str | None, entered_at_iso: str | None = None):
        self.conn.execute(
            """INSERT INTO occupancy_state (key, idpersonal, entered_at, last_event)
               VALUES (?,?,?, 'IN')
               ON CONFLICT(key) DO UPDATE SET idpersonal=excluded.idpersonal,
                   entered_at=excluded.entered_at, last_event='IN'""",
            (key, idpersonal, entered_at_iso or _now_iso()))
        self.conn.commit()

    def clear_inside(self, key: str):
        self.conn.execute("DELETE FROM occupancy_state WHERE key = ?", (key,))
        self.conn.commit()

    def inside_keys(self) -> list[tuple[str, str]]:
        rows = self.conn.execute(
            "SELECT key, entered_at FROM occupancy_state").fetchall()
        return [(r["key"], r["entered_at"]) for r in rows]

    def anomalies(self) -> list[dict]:
        """Everyone still marked inside — an exit that was never seen, or someone
        who stayed. Names filled from persons_cache when known."""
        rows = self.conn.execute(
            """SELECT o.key, o.idpersonal, o.entered_at, pc.name
               FROM occupancy_state o
               LEFT JOIN persons_cache pc ON pc.idpersonal = o.idpersonal
               ORDER BY o.entered_at""").fetchall()
        return [{"key": r["key"], "idpersonal": r["idpersonal"],
                 "name": r["name"], "entered_at": r["entered_at"]} for r in rows]

    def occupancy(self) -> dict:
        rows = self.conn.execute("SELECT key, idpersonal FROM occupancy_state").fetchall()
        known = sum(1 for r in rows if r["idpersonal"])
        return {"total": len(rows), "known": known, "unknown": len(rows) - known}

    # --- reporting ---
    def daily_summary(self, date: str | None = None) -> list[dict]:
        date = date or datetime.now().strftime("%Y-%m-%d")
        rows = self.conn.execute(
            """SELECT e.identity, e.idpersonal,
                      SUM(e.event_type='IN')  AS n_in,
                      SUM(e.event_type='OUT') AS n_out,
                      pc.name AS name
               FROM events e
               LEFT JOIN persons_cache pc ON pc.idpersonal = e.idpersonal
               WHERE e.date = ?
               GROUP BY e.identity, e.idpersonal
               ORDER BY e.identity""", (date,)).fetchall()
        inside = {r["key"] for r in
                  self.conn.execute("SELECT key FROM occupancy_state").fetchall()}
        out = []
        for r in rows:
            key = r["idpersonal"] or None
            out.append({
                "identity": r["identity"],
                "idpersonal": r["idpersonal"],
                "name": r["name"],
                "in": int(r["n_in"] or 0),
                "out": int(r["n_out"] or 0),
                "inside": bool(key and key in inside),
            })
        return out

    # --- persons_cache (written by EnrichmentWorker) ---
    def unresolved_idpersonals(self) -> list[str]:
        rows = self.conn.execute(
            """SELECT DISTINCT e.idpersonal FROM events e
               LEFT JOIN persons_cache pc ON pc.idpersonal = e.idpersonal
               WHERE e.idpersonal IS NOT NULL AND pc.idpersonal IS NULL""").fetchall()
        return [r["idpersonal"] for r in rows]

    def upsert_person_cache(self, idpersonal: str, name: str | None, info: dict | None):
        self.conn.execute(
            """INSERT INTO persons_cache (idpersonal, name, info_json, fetched_at)
               VALUES (?,?,?,?)
               ON CONFLICT(idpersonal) DO UPDATE SET name=excluded.name,
                   info_json=excluded.info_json, fetched_at=excluded.fetched_at""",
            (idpersonal, name, json.dumps(info) if info is not None else None, _now_iso()))
        self.conn.commit()

    def person_name(self, idpersonal: str) -> str | None:
        r = self.conn.execute(
            "SELECT name FROM persons_cache WHERE idpersonal = ?", (idpersonal,)).fetchone()
        return r["name"] if r else None
