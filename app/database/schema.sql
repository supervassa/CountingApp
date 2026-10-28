-- Local SQLite store (offline-first). Personnel names live in an external
-- PostgreSQL DB and are mirrored into persons_cache by the enrichment worker.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS persons (
    idpersonal  TEXT PRIMARY KEY,          -- UUID from the personnel system
    enrolled_at TEXT NOT NULL DEFAULT (datetime('now')),
    n_samples   INTEGER NOT NULL DEFAULT 0,
    source      TEXT NOT NULL DEFAULT 'imx219'  -- imx219 | raw | mixed
);

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    idpersonal  TEXT,                              -- NULL when UNKNOWN
    identity    TEXT NOT NULL,                     -- idpersonal string or 'UNKNOWN'
    event_type  TEXT NOT NULL CHECK (event_type IN ('IN','OUT')),
    camera_id   TEXT NOT NULL CHECK (camera_id IN ('cam_out','cam_in')),
    confidence  REAL,
    similarity  REAL,                              -- top-1 cosine at decision time
    all_scores  TEXT,                              -- JSON top-k, for later threshold re-calibration
    track_id    INTEGER,
    timestamp   TEXT NOT NULL,                     -- system time, ISO-8601
    date        TEXT NOT NULL,                     -- YYYY-MM-DD
    image_path  TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_date ON events(date);
CREATE INDEX IF NOT EXISTS idx_events_idpersonal ON events(idpersonal);

CREATE TABLE IF NOT EXISTS occupancy_state (
    key        TEXT PRIMARY KEY,                   -- idpersonal, else '<camera>:<track_id>'
    idpersonal TEXT,
    entered_at TEXT NOT NULL,
    last_event TEXT NOT NULL CHECK (last_event IN ('IN','OUT'))
);

CREATE TABLE IF NOT EXISTS persons_cache (
    idpersonal TEXT PRIMARY KEY,
    name       TEXT,
    info_json  TEXT,                               -- get_info_person() row as opaque JSON
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alarms (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL,
    date      TEXT NOT NULL,
    kind      TEXT NOT NULL,                        -- negative_occupancy | occupancy_too_high | camera_silent
    detail    TEXT
);
CREATE INDEX IF NOT EXISTS idx_alarms_date ON alarms(date);
