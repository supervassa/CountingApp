"""Build the Repository + EnrichmentWorker from config."""
from __future__ import annotations

import os

from .enrichment import EnrichmentWorker, build_source
from .repository import Repository


def build_repository(cfg_database) -> Repository:
    return Repository(cfg_database.sqlite_path)


def build_enrichment(cfg_personnel, repo) -> EnrichmentWorker:
    source = build_source(cfg_personnel, os.environ.get)
    return EnrichmentWorker(repo, source,
                            interval_s=getattr(cfg_personnel, "refresh_interval_s", 3600.0))
