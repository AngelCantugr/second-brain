"""Crash-safe sync state tracking for incremental indexing."""

from __future__ import annotations

import sqlite3
from pathlib import Path


_CHUNK_IDENTITY_VERSION = 1


class SyncStateStore:
    """Persist per-note hashes and timestamps for reindex decisions."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        """Open sqlite connection, creating parent directory as needed."""

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path)

    def initialize(self) -> None:
        """Add identity bookkeeping without resetting hashes or runtime rows.

        Old rows start at version zero so unchanged notes still reindex once.
        Only recording successful indexing advances a note's version.
        """

        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS note_state (
                    path TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    mtime REAL NOT NULL,
                    updated_at REAL DEFAULT (strftime('%s', 'now')),
                    identity_version INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(note_state)")}
            if "identity_version" not in columns:
                conn.execute(
                    "ALTER TABLE note_state ADD COLUMN identity_version INTEGER NOT NULL DEFAULT 0"
                )

    def should_reindex(self, path: str, content_hash: str) -> bool:
        """Return whether a note path should be re-indexed."""

        with self._connect() as conn:
            row = conn.execute(
                "SELECT content_hash, identity_version FROM note_state WHERE path = ?", (path,)
            ).fetchone()
        if row is None:
            return True
        return row[0] != content_hash or row[1] != _CHUNK_IDENTITY_VERSION

    def record_note(self, path: str, content_hash: str, mtime: float) -> None:
        """Checkpoint hash/mtime and identity version after successful indexing.

        The indexer calls this only after both stores and note metadata succeed;
        failed migrations keep their old hash/version and remain retryable.
        """

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO note_state(path, content_hash, mtime, identity_version)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    content_hash=excluded.content_hash,
                    mtime=excluded.mtime,
                    updated_at=strftime('%s', 'now'),
                    identity_version=excluded.identity_version
                """,
                (path, content_hash, mtime, _CHUNK_IDENTITY_VERSION),
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
