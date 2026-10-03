"""Crash-safe sync state tracking for incremental indexing."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


_CHUNK_IDENTITY_VERSION = 1


@dataclass(frozen=True, slots=True)
class TrackedFileState:
    """Read-only freshness projection for one checkpointed or pending path."""

    content_hash: str | None
    mtime: float | None
    requires_reindex: bool


class SyncStateStore:
    """Persist per-note hashes and timestamps for reindex decisions."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        """Open sqlite connection, creating parent directory as needed."""

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path)

    def initialize(self) -> None:
        """Initialize identity, pending replacement, and watcher state additively.

        Legacy note rows retain their saved hash and identity version. A separate
        pending table records interrupted replacements without changing those diagnostics.
        Watcher timestamps remain in their own runtime metadata table.
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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runtime_metadata (
                    key TEXT PRIMARY KEY,
                    value REAL NOT NULL
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

    def tracked_file_state(self) -> dict[str, TrackedFileState]:
        """Return checkpoint, identity, and pending state in one read-only projection."""

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT note_state.path, note_state.content_hash, note_state.mtime,
                    (note_state.identity_version != ? OR pending_replacements.path IS NOT NULL)
                FROM note_state
                LEFT JOIN pending_replacements USING (path)
                UNION ALL
                SELECT pending_replacements.path, NULL, NULL, 1
                FROM pending_replacements
                LEFT JOIN note_state USING (path)
                WHERE note_state.path IS NULL
                ORDER BY 1
                """,
                (_CHUNK_IDENTITY_VERSION,),
            ).fetchall()
        return {
            path: TrackedFileState(
                content_hash=content_hash,
                mtime=None if mtime is None else float(mtime),
                requires_reindex=bool(requires_reindex),
            )
            for path, content_hash, mtime, requires_reindex in rows
        }

    def record_watcher_event(self, timestamp: float) -> None:
        """Persist the time a watcher batch was observed, independently of sync."""

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runtime_metadata(key, value) VALUES('watcher_last_event', ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (timestamp,),
            )

    def watcher_last_event(self) -> float | None:
        """Return the most recent observed watcher batch time, if present."""

        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM runtime_metadata WHERE key = ?",
                ("watcher_last_event",),
            ).fetchone()
        return None if row is None else float(row[0])

    def last_sync_timestamp(self) -> float | None:
        """Return latest sync timestamp in epoch seconds, if any."""

        with self._connect() as conn:
            row = conn.execute("SELECT MAX(updated_at) FROM note_state").fetchone()
        if row is None or row[0] is None:
            return None
        return float(row[0])
