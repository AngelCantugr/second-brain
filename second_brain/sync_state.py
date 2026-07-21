"""Crash-safe sync state tracking for incremental indexing."""

from __future__ import annotations

import sqlite3
from pathlib import Path


class SyncStateStore:
    """Persist per-note hashes and timestamps for reindex decisions."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        """Open sqlite connection, creating parent directory as needed."""

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path)

    def initialize(self) -> None:
        """Create note state table if it does not exist."""

        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS note_state (
                    path TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    mtime REAL NOT NULL,
                    updated_at REAL DEFAULT (strftime('%s', 'now'))
                )
                """
            )

    def should_reindex(self, path: str, content_hash: str) -> bool:
        """Return whether a note path should be re-indexed."""

        with self._connect() as conn:
            row = conn.execute(
                "SELECT content_hash FROM note_state WHERE path = ?", (path,)
            ).fetchone()
        if row is None:
            return True
        return row[0] != content_hash

    def record_note(self, path: str, content_hash: str, mtime: float) -> None:
        """Upsert latest hash/mtime for a note path."""

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO note_state(path, content_hash, mtime)
                VALUES(?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    content_hash=excluded.content_hash,
                    mtime=excluded.mtime,
                    updated_at=strftime('%s', 'now')
                """,
                (path, content_hash, mtime),
            )

    def remove_note(self, path: str) -> None:
        """Remove note path state after file deletion."""

        with self._connect() as conn:
            conn.execute("DELETE FROM note_state WHERE path = ?", (path,))

    def tracked_paths(self) -> list[str]:
        """Return all currently tracked note paths."""

        with self._connect() as conn:
            rows = conn.execute("SELECT path FROM note_state").fetchall()
        return [r[0] for r in rows]

    def last_sync_timestamp(self) -> float | None:
        """Return latest sync timestamp in epoch seconds, if any."""

        with self._connect() as conn:
            row = conn.execute("SELECT MAX(updated_at) FROM note_state").fetchone()
        if row is None or row[0] is None:
            return None
        return float(row[0])
