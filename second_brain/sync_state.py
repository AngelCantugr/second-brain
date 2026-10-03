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
        """Create additive sync bookkeeping without resetting existing rows.

        Legacy note rows retain their saved hash and identity version. A separate
        pending table records interrupted replacements without changing those diagnostics.
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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pending_replacements (
                    path TEXT PRIMARY KEY
                )
                """
            )

    def mark_pending(self, path: str) -> None:
        """Durably mark a path before mutating either index."""

        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO pending_replacements(path) VALUES(?)", (path,)
            )

    def pending_paths(self) -> list[str]:
        """Return paths with an interrupted or unfinished replacement."""

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT path FROM pending_replacements ORDER BY path"
            ).fetchall()
        return [row[0] for row in rows]

    def should_reindex(self, path: str, content_hash: str) -> bool:
        """Return whether a note path should be re-indexed or recovered."""

        with self._connect() as conn:
            if conn.execute(
                "SELECT 1 FROM pending_replacements WHERE path = ?", (path,)
            ).fetchone():
                return True
            row = conn.execute(
                "SELECT content_hash, identity_version FROM note_state WHERE path = ?", (path,)
            ).fetchone()
        if row is None:
            return True
        return row[0] != content_hash or row[1] != _CHUNK_IDENTITY_VERSION

    def record_note(self, path: str, content_hash: str, mtime: float) -> None:
        """Checkpoint completed indexing and clear its pending marker atomically."""

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
            conn.execute("DELETE FROM pending_replacements WHERE path = ?", (path,))

    def remove_note(self, path: str) -> None:
        """Remove checkpoint and pending state after deleted-path cleanup succeeds."""

        with self._connect() as conn:
            conn.execute("DELETE FROM note_state WHERE path = ?", (path,))
            conn.execute("DELETE FROM pending_replacements WHERE path = ?", (path,))

    def tracked_paths(self) -> list[str]:
        """Return checkpointed and pending paths for indexing or deletion recovery."""

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT path FROM note_state
                UNION
                SELECT path FROM pending_replacements
                ORDER BY path
                """
            ).fetchall()
        return [row[0] for row in rows]

    def last_sync_timestamp(self) -> float | None:
        """Return latest sync timestamp in epoch seconds, if any."""

        with self._connect() as conn:
            row = conn.execute("SELECT MAX(updated_at) FROM note_state").fetchone()
        if row is None or row[0] is None:
            return None
        return float(row[0])
