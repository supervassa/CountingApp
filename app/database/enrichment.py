"""Async personnel-name enrichment.

Runs on a background thread. Each tick: take idpersonal values seen in events
but missing from persons_cache, resolve name/info from the external PostgreSQL
DB, write them to persons_cache. The capture loop never touches PostgreSQL.

PostgreSQL access is behind PersonnelSource:
  - NullPersonnelSource  : personnel_db.enabled = false (default) — no-op
  - PgPersonnelSource     : SELECT * FROM <info_function>($1)  (parameterised arg)

Guards: idpersonal must be a UUID; info_function must be a bare
schema.function identifier; short timeout; failures are logged and retried
next tick (simple circuit-breaker via consecutive-failure backoff).
"""
from __future__ import annotations

import logging
import re
import threading
import time
from uuid import UUID

log = logging.getLogger(__name__)

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")


def _is_uuid(s: str) -> bool:
    try:
        UUID(str(s))
        return True
    except (ValueError, TypeError, AttributeError):
        return False


class PersonnelSource:
    def get_info(self, idpersonal: str) -> dict | None:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self):
        pass


class NullPersonnelSource(PersonnelSource):
    def get_info(self, idpersonal: str) -> dict | None:
        return None


class PgPersonnelSource(PersonnelSource):
    def __init__(self, dsn: str, info_function: str, connect_timeout_s: float = 3.0):
        if not _IDENT_RE.match(info_function or ""):
            raise ValueError(f"unsafe info_function: {info_function!r}")
        try:
            import psycopg  # lazy; only needed when enabled
        except ImportError as e:  # pragma: no cover
            raise ImportError("psycopg required for personnel_db: pip install 'psycopg[binary]'") from e
        self._psycopg = psycopg
        self._dsn = dsn
        self._fn = info_function
        self._timeout = connect_timeout_s

    def get_info(self, idpersonal: str) -> dict | None:
        if not _is_uuid(idpersonal):
            log.warning("skip non-UUID idpersonal: %r", idpersonal)
            return None
        sql = f"SELECT * FROM {self._fn}(%s)"   # arg parameterised; fn validated at init
        with self._psycopg.connect(self._dsn, connect_timeout=self._timeout) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (idpersonal,))
                row = cur.fetchone()
                if row is None:
                    return None
                cols = [d.name for d in cur.description]
        info = dict(zip(cols, row))
        return info


def build_source(cfg_personnel, get_env) -> PersonnelSource:
    if not getattr(cfg_personnel, "enabled", False):
        return NullPersonnelSource()
    dsn = get_env(cfg_personnel.dsn_env)
    if not dsn:
        log.warning("personnel_db enabled but env %s is empty — running without enrichment",
                    cfg_personnel.dsn_env)
        return NullPersonnelSource()
    return PgPersonnelSource(dsn, cfg_personnel.info_function,
                             getattr(cfg_personnel, "connect_timeout_s", 3.0))


class EnrichmentWorker:
    def __init__(self, repo, source: PersonnelSource, interval_s: float = 3600.0):
        self.repo = repo
        self.source = source
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fails = 0

    def tick(self) -> int:
        """One resolution pass. Returns how many names were written."""
        wrote = 0
        for pid in self.repo.unresolved_idpersonals():
            try:
                info = self.source.get_info(pid)
            except Exception as e:
                self._fails += 1
                log.warning("enrichment: get_info(%s) failed (%d in a row): %s", pid, self._fails, e)
                return wrote
            self._fails = 0
            name = None
            if info:
                name = info.get("name") or info.get("nama") or info.get("full_name")
            self.repo.upsert_person_cache(pid, name, info)
            wrote += 1
        return wrote

    def _loop(self):
        while not self._stop.is_set():
            try:
                n = self.tick()
                if n:
                    log.info("enrichment: resolved %d name(s)", n)
            except Exception:
                log.exception("enrichment tick crashed")
            backoff = min(self.interval_s * (2 ** self._fails), 3600.0) if self._fails else self.interval_s
            self._stop.wait(backoff)

    def start(self):
        if isinstance(self.source, NullPersonnelSource):
            log.info("enrichment worker not started (personnel_db disabled)")
            return
        self._thread = threading.Thread(target=self._loop, name="enrichment", daemon=True)
        self._thread.start()
        log.info("enrichment worker started (interval %.0fs)", self.interval_s)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.source.close()
