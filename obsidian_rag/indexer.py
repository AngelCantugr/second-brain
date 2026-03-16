"""Index orchestration: scan -> parse -> chunk -> embed -> upsert."""

from __future__ import annotations

from pathlib import Path

from obsidian_rag.chunker import chunk_note
from obsidian_rag.config import RagConfig
from obsidian_rag.embedder import Embedder
from obsidian_rag.keyword_store import KeywordStore
from obsidian_rag.models import SyncResult
from obsidian_rag.parser import parse_note
from obsidian_rag.scanner import iter_markdown_files
from obsidian_rag.sync_state import SyncStateStore


class Indexer:
    """Coordinates the full indexing lifecycle for note content."""

    def __init__(
        self,
        config: RagConfig,
        embedder: Embedder,
        vector_store,
        keyword_store: KeywordStore,
        sync_state: SyncStateStore,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.vector_store = vector_store
        self.keyword_store = keyword_store
        self.sync_state = sync_state

    def initialize(self) -> None:
        """Initialize backing stores required for indexing."""

        self.keyword_store.initialize()
        self.sync_state.initialize()

    def sync(self, mode: str = "incremental", file_path: str | None = None) -> SyncResult:
        """Run full/incremental/file-based sync and return summary stats."""

        if mode == "file":
            if not file_path:
                raise ValueError("file_path is required when mode='file'")
            return self._sync_single(Path(file_path))

        files = iter_markdown_files(self.config.vault_path, self.config.exclude_globs)
        current_paths = {str(p.relative_to(self.config.vault_path)) for p in files}

        processed = 0
        skipped = 0
        errors: list[str] = []

        for path in files:
            try:
                parsed = parse_note(path, self.config.vault_path)
                if mode != "full" and not self.sync_state.should_reindex(parsed.path, parsed.content_hash):
                    skipped += 1
                    continue

                self._upsert_parsed(parsed)
                self.sync_state.record_note(parsed.path, parsed.content_hash, parsed.mtime)
                processed += 1
            except Exception as exc:
                errors.append(f"{path}: {exc}")

        deleted = 0
        for tracked in self.sync_state.tracked_paths():
            if tracked in current_paths:
                continue
            self._delete_missing_path(tracked)
            deleted += 1

        return SyncResult(processed=processed, skipped=skipped, deleted=deleted, errors=errors)

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
        self._upsert_parsed(parsed)
        self.sync_state.record_note(parsed.path, parsed.content_hash, parsed.mtime)
        return SyncResult(processed=1, skipped=0, deleted=0, errors=[])

    def _upsert_parsed(self, parsed) -> None:
        """Chunk parsed note, embed chunks, and upsert into both indexes."""

        chunks = chunk_note(
            parsed,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        if not chunks:
            return
        embeddings = self.embedder.embed([c.text for c in chunks])
        self.vector_store.ensure_collection(len(embeddings[0]))
        self.vector_store.upsert_chunks(chunks, embeddings)
        self.keyword_store.upsert_chunks(chunks)

    def _delete_missing_path(self, rel_path: str) -> None:
        """Delete indexed records for a note removed from disk."""

        ids = self.keyword_store.chunk_ids_by_path(rel_path)
        if ids:
            self.keyword_store.delete_chunks(ids)
            self.vector_store.delete_chunks(ids)
        self.sync_state.remove_note(rel_path)
