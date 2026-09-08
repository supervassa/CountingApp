"""Repository + EnrichmentWorker — Capaian 6 acceptance checks.

Run: python tests/test_database.py   (or pytest)
"""
from __future__ import annotations

import sys
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.counting.counter import CrossEvent                       # noqa: E402
from app.database.enrichment import (EnrichmentWorker, NullPersonnelSource,  # noqa: E402
                                     PersonnelSource, PgPersonnelSource, _is_uuid)
from app.database.repository import Repository                     # noqa: E402


def _repo() -> Repository:
    d = tempfile.mkdtemp()
    return Repository(str(Path(d) / "events.db"))


def test_events_persist_and_summarise():
    r = _repo()
    pid = str(uuid.uuid4())
    r.insert_event(CrossEvent("cam_out", "IN", 1, ts=0.0, identity=pid, idpersonal=pid))
    r.set_inside(pid, pid)
    r.insert_event(CrossEvent("cam_in", "OUT", 1, ts=1.0, identity=pid, idpersonal=pid))
    r.clear_inside(pid)
    r.insert_event(CrossEvent("cam_out", "IN", 2, ts=2.0, identity=pid, idpersonal=pid))
    r.set_inside(pid, pid)
    r.insert_event(CrossEvent("cam_out", "IN", 3, ts=3.0))          # UNKNOWN
    r.set_inside("cam_out:3", None)

    rows = {row["identity"]: row for row in r.daily_summary()}
    assert rows[pid]["in"] == 2 and rows[pid]["out"] == 1 and rows[pid]["inside"] is True
    assert rows["UNKNOWN"]["in"] == 1 and rows["UNKNOWN"]["out"] == 0
    occ = r.occupancy()
    assert occ == {"total": 2, "known": 1, "unknown": 1}, occ
    r.close()


def test_anomalies_lists_still_inside():
    r = _repo()
    pid = str(uuid.uuid4())
    r.insert_event(CrossEvent("cam_out", "IN", 1, ts=0.0, identity=pid, idpersonal=pid))
    r.set_inside(pid, pid, entered_at_iso="2026-09-08T07:30:00+07:00")
    r.insert_event(CrossEvent("cam_out", "IN", 2, ts=1.0))
    r.set_inside("cam_out:2", None, entered_at_iso="2026-09-08T08:00:00+07:00")
    r.insert_event(CrossEvent("cam_out", "IN", 3, ts=2.0))
    r.set_inside("cam_out:3", None)
    r.insert_event(CrossEvent("cam_in", "OUT", 3, ts=3.0))
    r.clear_inside("cam_out:3")                                    # #3 left cleanly

    an = r.anomalies()
    assert {a["key"] for a in an} == {pid, "cam_out:2"}, an
    assert an[0]["key"] == pid                                     # ordered by entered_at
    r.close()


def test_timestamp_is_system_time_not_event_ts():
    r = _repo()
    r.insert_event(CrossEvent("cam_out", "IN", 1, ts=0.0))         # ts=epoch 0 (1970)
    ts = r.conn.execute("SELECT timestamp, date FROM events").fetchone()
    assert not ts["timestamp"].startswith("1970"), ts["timestamp"]
    r.close()


class _FakeSource(PersonnelSource):
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = 0

    def get_info(self, idpersonal):
        self.calls += 1
        return self.mapping.get(idpersonal)


def test_enrichment_resolves_names_offline_safe():
    r = _repo()
    pid = str(uuid.uuid4())
    r.insert_event(CrossEvent("cam_out", "IN", 1, ts=0.0, identity=pid, idpersonal=pid))
    assert r.unresolved_idpersonals() == [pid]

    w = EnrichmentWorker(r, _FakeSource({pid: {"name": "Juris X", "dept": "IT"}}))
    assert w.tick() == 1
    assert r.person_name(pid) == "Juris X"
    assert r.unresolved_idpersonals() == []
    assert w.tick() == 0                                           # nothing left to do
    r.close()


def test_disabled_source_is_noop():
    r = _repo()
    pid = str(uuid.uuid4())
    r.insert_event(CrossEvent("cam_out", "IN", 1, ts=0.0, identity=pid, idpersonal=pid))
    w = EnrichmentWorker(r, NullPersonnelSource())
    w.start()                                                      # must not spawn a thread
    assert w._thread is None
    assert w.tick() == 1                                           # writes a cache row with name=None
    assert r.person_name(pid) is None
    r.close()


def test_pg_source_rejects_unsafe_function_name():
    for bad in ("person.get_info_person; DROP TABLE persons", "1bad", "a b", "sel--ect"):
        try:
            PgPersonnelSource("postgresql://x", bad)
        except (ValueError, ImportError):
            continue
        raise AssertionError(f"accepted unsafe info_function: {bad!r}")


def test_is_uuid():
    assert _is_uuid(str(uuid.uuid4()))
    assert not _is_uuid("not-a-uuid")
    assert not _is_uuid(None)


if __name__ == "__main__":
    mod = sys.modules[__name__]
    for n in [x for x in dir(mod) if x.startswith("test_")]:
        getattr(mod, n)()
        print("ok", n)
