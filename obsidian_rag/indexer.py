"""Index orchestration: scan -> parse -> chunk -> embed -> upsert."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from obsidian_rag.chunker import chunk_note
from obsidian_rag.config import RagConfig
from obsidian_rag.embedder import Embedder
from obsidian_rag.graph import GraphBuilder, GraphStore
from obsidian_rag.keyword_store import KeywordStore
from obsidian_rag.models import SyncResult
from obsidian_rag.parser import parse_note
from obsidian_rag.scanner import iter_markdown_files
from obsidian_rag.sync_state import SyncStateStore


_VALID_SYNC_MODES = {"full", "incremental", "file"}


class Indexer:
    """Coordinates the full indexing lifecycle for note content."""

    def __init__(
        self,
        config: RagConfig,
        embedder: Embedder,
        vector_store,
        keyword_store: KeywordStore,
        sync_state: SyncStateStore,
        graph_store: GraphStore,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.vector_store = vector_store
        self.keyword_store = keyword_store
        self.sync_state = sync_state
        self.graph_store = graph_store
        self.graph_builder = GraphBuilder(graph_store, config)

    def initialize(self) -> None:
        """Initialize backing stores required for indexing."""

        self.keyword_store.initialize()
        self.sync_state.initialize()
        self.graph_store.initialize()

    def sync(self, mode: str = "incremental", file_path: str | None = None) -> SyncResult:
        """Run full/incremental/file-based sync and return summary stats."""

        if mode not in _VALID_SYNC_MODES:
            raise ValueError(
                f"mode must be one of {sorted(_VALID_SYNC_MODES)}, got {mode!r}"
            )

        if mode == "file":
            if not file_path:
                raise ValueError("file_path is required when mode='file'")
            return self._sync_single(Path(file_path))

        files = iter_markdown_files(self.config.vault_path, self.config.exclude_globs)
        current_paths = {str(p.relative_to(self.config.vault_path)) for p in files}

        processed = 0
        skipped = 0
        errors: list[str] = []
        changed_paths: set[str] = set()
        deleted_paths: set[str] = set()
        old_meta: dict[str, dict | None] = {}

        for path in files:
            try:
                parsed = parse_note(path, self.config.vault_path)
                if mode != "full" and not self.sync_state.should_reindex(parsed.path, parsed.content_hash):
                    skipped += 1
                    continue

                old_meta[parsed.path] = self._upsert_parsed(parsed)
                self.sync_state.record_note(parsed.path, parsed.content_hash, parsed.mtime)
                changed_paths.add(parsed.path)
                processed += 1
            except Exception as exc:
                errors.append(f"{path}: {exc}")

        deleted = 0
        for tracked in self.sync_state.tracked_paths():
            if tracked in current_paths:
                continue
            old_meta[tracked] = self._delete_missing_path(tracked)
            deleted_paths.add(tracked)
            deleted += 1

        edges_updated = 0
        if self.config.graph_enabled:
            try:
                if mode == "full":
                    result = self.graph_builder.rebuild_full()
                elif changed_paths or deleted_paths:
                    result = self.graph_builder.update_for_changes(
                        changed_paths, deleted_paths, old_meta
                    )
                else:
                    result = None
                if result is not None:
                    edges_updated = result["edges_updated"]
            except Exception as exc:
                errors.append(f"graph build: {exc}")

        return SyncResult(
            processed=processed,
            skipped=skipped,
            deleted=deleted,
            errors=errors,
            graph_edges_updated=edges_updated,
        )

    def _sync_single(self, path: Path) -> SyncResult:
        """Sync exactly one file path."""

        resolved_path = path if path.is_absolute() else self.config.vault_path / path
        resolved_path = resolved_path.resolve()
        try:
            resolved_path.relative_to(self.config.vault_path.resolve())
        except ValueError as exc:
            raise ValueError("file_path must resolve inside vault_path") from exc

        parsed = parse_note(resolved_path, self.config.vault_path)
        if not self.sync_state.should_reindex(parsed.path, parsed.content_hash):
            return SyncResult(processed=0, skipped=1, deleted=0, errors=[])

        previous = self._upsert_parsed(parsed)
        self.sync_state.record_note(parsed.path, parsed.content_hash, parsed.mtime)

        edges_updated = 0
        errors: list[str] = []
        if self.config.graph_enabled:
            try:
                result = self.graph_builder.update_for_changes(
                    {parsed.path}, set(), {parsed.path: previous}
                )
                edges_updated = result["edges_updated"]
            except Exception as exc:
                errors.append(f"graph build: {exc}")

        return SyncResult(
            processed=1,
            skipped=0,
            deleted=0,
            errors=errors,
            graph_edges_updated=edges_updated,
        )

    def _upsert_parsed(self, parsed) -> dict | None:
        """Chunk parsed note, embed chunks, and upsert into both indexes.

        Always upserts graph metadata (title/tags/links), even when the note
        produces no chunks, so a chunkless note can still participate in the
        association graph via its links and tags. Returns the note's
        previous graph metadata row (or None if it's new), for the caller to
        hand to the graph builder's incremental update.
        """

        chunks = chunk_note(
            parsed,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        if not chunks:
            return self.graph_store.upsert_note_meta(
                parsed.path, parsed.title, parsed.tags, parsed.links, None
            )

        embeddings = self.embedder.embed([c.text for c in chunks])
        self.vector_store.ensure_collection(len(embeddings[0]))
        self.vector_store.upsert_chunks(chunks, embeddings)
        self.keyword_store.upsert_chunks(chunks)

        centroid = np.mean(np.asarray(embeddings, dtype=np.float64), axis=0).tolist()
        return self.graph_store.upsert_note_meta(
            parsed.path, parsed.title, parsed.tags, parsed.links, centroid
        )

    def _delete_missing_path(self, rel_path: str) -> dict | None:
        """Delete indexed records for a note removed from disk.

        Returns the note's previous graph metadata row for the caller to
        hand to the graph builder's incremental update.
        """

        ids = self.keyword_store.chunk_ids_by_path(rel_path)
        if ids:
            self.keyword_store.delete_chunks(ids)
            self.vector_store.delete_chunks(ids)
        self.sync_state.remove_note(rel_path)
        return self.graph_store.delete_note_meta(rel_path)
