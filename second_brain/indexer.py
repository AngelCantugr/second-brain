"""Index orchestration: scan -> parse -> chunk -> embed -> upsert."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from second_brain.chunker import chunk_note, embedding_text
from second_brain.config import RagConfig
from second_brain.embedder import Embedder
from second_brain.graph import GraphBuilder, GraphStore
from second_brain.keyword_store import KeywordStore
from second_brain.models import ChunkRecord, ParsedNote, SyncResult
from second_brain.parser import frontmatter_aliases, parse_note
from second_brain.redaction import Redactor
from second_brain.scanner import is_excluded_path, iter_markdown_files
from second_brain.sync_state import SyncStateStore


_VALID_SYNC_MODES = {"full", "incremental", "file"}


@dataclass(slots=True)
class _PreparedNote:
    """A redacted, chunked note ready to embed; nothing is written for it yet."""

    source: Path | str
    parsed: ParsedNote
    chunks: list[ChunkRecord]
    embed_inputs: list[str]


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
        self.redactor = Redactor(config.redact_patterns)

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

        to_index: list[_PreparedNote] = []
        for path in files:
            try:
                parsed = parse_note(path, self.config.vault_path)
                if mode != "full" and not self.sync_state.should_reindex(parsed.path, parsed.content_hash):
                    skipped += 1
                    continue
                to_index.append(self._prepare(parsed, source=path))
            except Exception as exc:
                errors.append(f"{path}: {exc}")

        for note, embeddings in self._embed_in_batches(to_index):
            parsed = note.parsed
            try:
                if isinstance(embeddings, Exception):
                    raise embeddings
                old_meta[parsed.path] = self._write_note(note, embeddings)
                self.sync_state.record_note(parsed.path, parsed.content_hash, parsed.mtime)
                changed_paths.add(parsed.path)
                processed += 1
            except Exception as exc:
                errors.append(f"{note.source}: {exc}")

        deleted = 0
        for tracked in self.sync_state.tracked_paths():
            if tracked in current_paths:
                continue
            try:
                old_meta[tracked] = self._delete_missing_path(tracked)
                deleted_paths.add(tracked)
                deleted += 1
            except Exception as exc:
                errors.append(f"{tracked}: {exc}")

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
        """Sync one file while applying the same inclusion rules as vault scans."""

        resolved_path = path if path.is_absolute() else self.config.vault_path / path
        resolved_path = resolved_path.resolve()
        vault_root = self.config.vault_path.resolve()
        try:
            relative_path = resolved_path.relative_to(vault_root)
        except ValueError as exc:
            raise ValueError("file_path must resolve inside vault_path") from exc

        rel_path = relative_path.as_posix()
        if is_excluded_path(resolved_path, vault_root, self.config.exclude_globs):
            errors: list[str] = []
            deleted = 0
            graph_edges_updated = 0
            previous_meta: dict[str, dict | None] = {}
            if rel_path in self.sync_state.tracked_paths():
                try:
                    previous_meta[rel_path] = self._delete_missing_path(rel_path)
                    deleted = 1
                except Exception as exc:
                    errors.append(f"{rel_path}: {exc}")

                if deleted and self.config.graph_enabled:
                    try:
                        result = self.graph_builder.update_for_changes(
                            set(), {rel_path}, previous_meta
                        )
                        graph_edges_updated = result["edges_updated"]
                    except Exception as exc:
                        errors.append(f"graph build: {exc}")
                        try:
                            self.sync_state.mark_pending(rel_path)
                            old_meta = previous_meta[rel_path]
                            if old_meta is not None:
                                self.graph_store.upsert_note_meta(
                                    rel_path,
                                    old_meta["title"],
                                    old_meta["tags"],
                                    old_meta["links"],
                                    old_meta["centroid"],
                                    old_meta["aliases"],
                                )
                        except Exception as recovery_exc:
                            errors.append(
                                f"graph cleanup recovery for {rel_path}: {recovery_exc}"
                            )

            return SyncResult(
                processed=0,
                skipped=0 if deleted or errors else 1,
                deleted=deleted,
                errors=errors,
                graph_edges_updated=graph_edges_updated,
            )

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

    def _upsert_parsed(self, parsed: ParsedNote) -> dict | None:
        """Embed one note, mark it pending, then replace both indexes.

        Embedding happens before the durable pending marker and cleanup, so an
        embedding failure leaves current data untouched. The caller records the
        successful checkpoint only after both stores and note metadata succeed.
        """

        note = self._prepare(parsed, source=parsed.path)
        embeddings = self.embedder.embed(note.embed_inputs) if note.embed_inputs else []
        return self._write_note(note, embeddings)

    def _prepare(self, parsed: ParsedNote, source: Path | str) -> _PreparedNote:
        """Redact and chunk a note, and build its embedder inputs.

        The note is redacted before anything else reads it, so chunk text, the
        embedder input, FTS rows, Qdrant payloads and graph metadata never see
        a matched string. ``skip_headings`` therefore matches redacted headings.
        """

        parsed = self.redactor.note(parsed)
        chunks = chunk_note(
            parsed,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            skip_headings=self.config.skip_headings,
        )
        embed_inputs = (
            [embedding_text(chunk) for chunk in chunks]
            if self.config.embed_context_header
            else [chunk.text for chunk in chunks]
        )
        return _PreparedNote(source, parsed, chunks, embed_inputs)

    def _embed_in_batches(
        self, notes: Iterable[_PreparedNote]
    ) -> Iterator[tuple[_PreparedNote, list[list[float]] | Exception]]:
        """Yield each note, in order, with its vectors or the error that stopped it.

        Whole notes are packed into one ``embed()`` call up to
        ``embed_batch_size`` texts, so a full sync sends few, full ``/api/embed``
        requests instead of one part-empty request per note. A call never holds
        more than one batch unless a single note is larger than a batch, so one
        failed request touches only the notes packed into it.
        """

        batch_size = self.config.embed_batch_size
        group: list[_PreparedNote] = []
        group_texts = 0
        for note in notes:
            if group and group_texts + len(note.embed_inputs) > batch_size:
                yield from self._embed_group(group)
                group, group_texts = [], 0
            group.append(note)
            group_texts += len(note.embed_inputs)
        if group:
            yield from self._embed_group(group)

    def _embed_group(
        self, group: list[_PreparedNote]
    ) -> Iterator[tuple[_PreparedNote, list[list[float]] | Exception]]:
        texts = [text for note in group for text in note.embed_inputs]
        if not texts:
            for note in group:
                yield note, []
            return
        try:
            vectors = self.embedder.embed(texts)
            if len(vectors) != len(texts):
                raise ValueError(f"embedder returned {len(vectors)} vectors for {len(texts)} texts")
        except Exception as exc:
            if len(group) == 1:
                yield group[0], exc
                return
            # Re-embed each note alone. Packing is deterministic, so without
            # this one note Ollama rejects would fail the notes packed beside
            # it on every sync, not just its own.
            for note in group:
                yield from self._embed_group([note])
            return

        offset = 0
        for note in group:
            count = len(note.embed_inputs)
            yield note, vectors[offset : offset + count]
            offset += count

    def _write_note(self, note: _PreparedNote, embeddings: list[list[float]]) -> dict | None:
        """Mark an embedded note pending, then replace its chunks in both indexes.

        Called only once the note's vectors exist, so a failed embedding never
        reaches the pending marker or the deletes below.
        """

        parsed, chunks = note.parsed, note.chunks
        old_chunk_ids = self._chunk_ids_by_path(parsed.path)
        new_chunk_ids = {chunk.chunk_id for chunk in chunks}
        self.sync_state.mark_pending(parsed.path)

        obsolete_ids = old_chunk_ids - new_chunk_ids
        if obsolete_ids:
            obsolete_list = sorted(obsolete_ids)
            self.vector_store.delete_chunks(obsolete_list)
            self.keyword_store.delete_chunks(obsolete_list)

        aliases = frontmatter_aliases(parsed.frontmatter)
        if not chunks:
            return self.graph_store.upsert_note_meta(
                parsed.path, parsed.title, parsed.tags, parsed.links, None, aliases
            )

        self.vector_store.ensure_collection(len(embeddings[0]))
        self.vector_store.upsert_chunks(chunks, embeddings)
        self.keyword_store.upsert_chunks(chunks)

        centroid = np.mean(np.asarray(embeddings, dtype=np.float64), axis=0).tolist()
        return self.graph_store.upsert_note_meta(
            parsed.path, parsed.title, parsed.tags, parsed.links, centroid, aliases
        )

    def _chunk_ids_by_path(self, rel_path: str) -> set[str]:
        """Find ownership independently in each store for partial-write recovery."""

        return set(self.keyword_store.chunk_ids_by_path(rel_path)) | {
            hit.chunk_id for hit in self.vector_store.get_by_path(rel_path)
        }

    def _delete_missing_path(self, rel_path: str) -> dict | None:
        """Remove a deleted note while retaining the checkpoint on failure.

        Discover IDs in both stores so retry also removes vector-only leftovers.
        Tracking is removed only after both deletions and metadata succeed.
        """

        ids = sorted(self._chunk_ids_by_path(rel_path))
        if ids:
            self.vector_store.delete_chunks(ids)
            self.keyword_store.delete_chunks(ids)
        previous = self.graph_store.delete_note_meta(rel_path)
        self.sync_state.remove_note(rel_path)
        return previous
