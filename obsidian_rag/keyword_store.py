"""SQLite FTS-based keyword index used for lexical retrieval and filtering."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from obsidian_rag.models import ChunkRecord, RetrievalHit

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _chunk_links_target(metadata_json: str, target_title: str) -> bool:
    """SQLite UDF: does a chunk's metadata contain a wikilink to ``target_title``?"""

    try:
        metadata = json.loads(metadata_json)
    except (json.JSONDecodeError, TypeError):
        return False
    for link in metadata.get("links", []):
        link_title = link.split("#", 1)[0].strip().lower()
        if link_title == target_title:
            return True
    return False


def build_fts_match_query(query: str) -> str:
    """Turn a raw user query into a syntax-safe FTS5 MATCH expression.

    FTS5's MATCH operand has its own query grammar where characters like
    ``? " ( ) * : ^ -`` are syntax, not literal text. Extracting word tokens
    and quoting each as an FTS5 phrase neutralizes that syntax while
    preserving implicit-AND matching across terms.
    """

    tokens = _TOKEN_RE.findall(query)
    return " ".join('"' + token.replace('"', '""') + '"' for token in tokens)


class KeywordStore:
    """Manage keyword index lifecycle, writes, and reads."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        """Open sqlite connection with row-dict access enabled."""

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        """Create storage + FTS tables if missing."""

        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    note_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                USING fts5(chunk_id UNINDEXED, text)
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_note_id ON chunks(note_id)"
            )

    def upsert_chunks(self, chunks: list[ChunkRecord]) -> None:
        """Insert or update chunk rows and corresponding FTS rows."""

        with self._connect() as conn:
            for chunk in chunks:
                metadata_json = json.dumps(chunk.metadata, sort_keys=True)
                conn.execute(
                    """
                    INSERT INTO chunks(chunk_id, note_id, text, metadata_json)
                    VALUES(?, ?, ?, ?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        note_id=excluded.note_id,
                        text=excluded.text,
                        metadata_json=excluded.metadata_json
                    """,
                    (chunk.chunk_id, chunk.note_id, chunk.bm25_text, metadata_json),
                )
                conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk.chunk_id,))
                conn.execute(
                    "INSERT INTO chunks_fts(rowid, chunk_id, text) VALUES(NULL, ?, ?)",
                    (chunk.chunk_id, chunk.bm25_text),
                )

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        """Delete chunks from primary and FTS tables."""

        if not chunk_ids:
            return
        placeholders = ",".join("?" for _ in chunk_ids)
        with self._connect() as conn:
            conn.execute(f"DELETE FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids)
            conn.execute(f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})", chunk_ids)

    def delete_note_chunks(self, note_id: str) -> None:
        """Delete all chunks that belong to a given note id."""

        with self._connect() as conn:
            rows = conn.execute("SELECT chunk_id FROM chunks WHERE note_id = ?", (note_id,)).fetchall()
            self.delete_chunks([r[0] for r in rows])

    def chunk_ids_by_path(self, rel_path: str) -> list[str]:
        """Return chunk ids whose metadata path matches ``rel_path``."""

        with self._connect() as conn:
            rows = conn.execute("SELECT chunk_id, metadata_json FROM chunks").fetchall()
        result: list[str] = []
        for row in rows:
            metadata = json.loads(row["metadata_json"])
            if metadata.get("path") == rel_path:
                result.append(row["chunk_id"])
        return result

    def chunks_by_path(self, rel_path: str) -> list[RetrievalHit]:
        """Return all chunks for a note path as retrieval hits."""

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT chunk_id, text, metadata_json FROM chunks"
            ).fetchall()
        hits: list[RetrievalHit] = []
        for row in rows:
            metadata = json.loads(row["metadata_json"])
            if metadata.get("path") != rel_path:
                continue
            hits.append(
                RetrievalHit(
                    chunk_id=row["chunk_id"],
                    score=0.0,
                    source="keyword",
                    text=row["text"],
                    metadata=metadata,
                )
            )
        return hits

    def backlinks_for_title(self, note_title: str, *, exclude_path: str | None = None) -> list[str]:
        """Return paths of notes whose chunks contain a wikilink to ``note_title``."""

        target = note_title.strip().lower()
        with self._connect() as conn:
            conn.create_function("chunk_links_target", 2, _chunk_links_target)
            sql = (
                "SELECT DISTINCT json_extract(metadata_json, '$.path') FROM chunks "
                "WHERE chunk_links_target(metadata_json, ?)"
            )
            params: list[str] = [target]
            if exclude_path:
                sql += " AND json_extract(metadata_json, '$.path') != ?"
                params.append(exclude_path)
            rows = conn.execute(sql, params).fetchall()

        return sorted(row[0] for row in rows if row[0])

    def search(self, query: str, limit: int = 10, filters: dict | None = None) -> list[RetrievalHit]:
        """Search FTS index and apply structured metadata filters."""

        if not query.strip() or query.strip() == "*":
            return self._list_chunks(limit=limit, filters=filters or {})
        match_query = build_fts_match_query(query)
        if not match_query:
            return []
        sql = (
            "SELECT c.chunk_id, c.text, c.metadata_json, bm25(chunks_fts) AS rank "
            "FROM chunks_fts JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id "
            "WHERE chunks_fts MATCH ? "
            "ORDER BY rank LIMIT ?"
        )
        hits: list[RetrievalHit] = []

        with self._connect() as conn:
            rows = conn.execute(sql, (match_query, limit * 3)).fetchall()

        for row in rows:
            metadata = json.loads(row["metadata_json"])
            if not matches_filters(metadata, filters or {}):
                continue
            hits.append(
                RetrievalHit(
                    chunk_id=row["chunk_id"],
                    score=float(-row["rank"]),
                    source="keyword",
                    text=row["text"],
                    metadata=metadata,
                )
            )
            if len(hits) >= limit:
                break

        return hits

    def _list_chunks(self, limit: int, filters: dict) -> list[RetrievalHit]:
        """List chunks without FTS matching, still applying filters."""

        with self._connect() as conn:
            rows = conn.execute("SELECT chunk_id, text, metadata_json FROM chunks LIMIT ?", (limit * 4,)).fetchall()
        hits: list[RetrievalHit] = []
        for row in rows:
            metadata = json.loads(row["metadata_json"])
            if not matches_filters(metadata, filters):
                continue
            hits.append(
                RetrievalHit(
                    chunk_id=row["chunk_id"],
                    score=0.0,
                    source="keyword",
                    text=row["text"],
                    metadata=metadata,
                )
            )
            if len(hits) >= limit:
                break
        return hits

    def count_chunks(self) -> int:
        """Return number of indexed chunks."""

        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return int(row[0]) if row else 0


def matches_filters(metadata: dict, filters: dict) -> bool:
    """Evaluate supported filter predicates against chunk metadata."""

    for key, expected in filters.items():
        if key == "path_prefix":
            if not str(metadata.get("path", "")).startswith(str(expected)):
                return False
            continue

        if key == "tags":
            if expected is None:
                expected_tags = []
            elif isinstance(expected, (list, set, tuple)):
                expected_tags = expected
            else:
                expected_tags = [expected]
            # Obsidian tags are typed inconsistently across notes, so tag
            # matching is case-insensitive on both sides of the comparison.
            expected_tags = {str(t).lower() for t in expected_tags}
            tags = {str(t).lower() for t in metadata.get("tags", [])}
            if not expected_tags.issubset(tags):
                return False
            continue

        if key == "date_range":
            if not isinstance(expected, dict):
                raise ValueError(
                    f"filters['date_range'] must be an object with 'start'/'end' keys, "
                    f"got {type(expected).__name__}"
                )
            derived = metadata.get("derived_fields", {})
            candidate = (
                derived.get("due_date")
                or derived.get("deadline_date")
                or derived.get("start_date")
                or derived.get("created_date")
            )
            if not candidate:
                return False
            start = expected.get("start")
            end = expected.get("end")
            if start and str(candidate) < str(start):
                return False
            if end and str(candidate) > str(end):
                return False
            continue

        if key == "frontmatter_contains":
            if not isinstance(expected, dict):
                raise ValueError(
                    f"filters['frontmatter_contains'] must be an object of key/value pairs, "
                    f"got {type(expected).__name__}"
                )
            fm = metadata.get("raw_frontmatter", {})
            if not all(fm.get(k) == v for k, v in expected.items()):
                return False
            continue

        if metadata.get(key) != expected and metadata.get("derived_fields", {}).get(key) != expected:
            return False

    return True
