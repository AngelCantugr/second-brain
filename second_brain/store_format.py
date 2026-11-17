"""Keep bank-keyed data separate from indexes that use legacy paths."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from second_brain.config import RagConfig

_BANK_SQLITE_VERSION = 2
_BANK_VECTOR_MARKER = ".second-brain-bank-index-v2"


def prepare_bank_stores(config: RagConfig, *, use_in_memory_vector: bool) -> None:
    """Check every persistent path before opening or advancing any store."""
    databases = {config.fts_path, config.sync_state_path}
    new_databases = []
    for database in databases:
        if database.is_file() and database.stat().st_size:
            with closing(sqlite3.connect(
                f"{database.resolve().as_uri()}?mode=ro&immutable=1", uri=True
            )) as connection:
                version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version != _BANK_SQLITE_VERSION:
                raise _fresh_store_required(database)
        else:
            new_databases.append(database)

    vectors = config.qdrant_path
    marker = vectors / _BANK_VECTOR_MARKER
    if not use_in_memory_vector and vectors.is_dir() and not marker.is_file() and any(vectors.iterdir()):
        raise _fresh_store_required(vectors)

    for database in new_databases:
        database.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(database)) as connection:
            connection.execute(f"PRAGMA user_version = {_BANK_SQLITE_VERSION}")
    if not use_in_memory_vector and not marker.is_file():
        vectors.mkdir(parents=True, exist_ok=True)
        marker.touch(exist_ok=True)


def _fresh_store_required(path: Path) -> ValueError:
    return ValueError(f"{path} may contain a legacy index; configure fresh index storage for banks")
