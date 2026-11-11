"""Index orchestration: scan -> parse -> chunk -> embed -> upsert."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from second_brain.chunker import chunk_note, embedding_text
from second_brain.code import body_fingerprint
from second_brain.config import RagConfig
from second_brain.embedder import Embedder
from second_brain.graph import GraphBuilder, GraphStore
from second_brain.keyword_store import KeywordStore
from second_brain.models import ChunkRecord, ParsedNote, SyncResult
from second_brain.loaders import load_source, loader_for
from second_brain.parser import frontmatter_aliases
from second_brain.redaction import Redactor
from second_brain.scanner import is_excluded_path
from second_brain.sources import iter_all_source_files, load_source_file, split_code_source_path
from second_brain.sync_state import SyncStateStore


_VALID_SYNC_MODES = {"full", "incremental", "file"}


def _code_meta(parsed: ParsedNote) -> dict:
    """What the graph needs to draw typed code edges, with empty keys left out.

    A note records the code it may mention; a source file records the symbols
    it defines and what it imports. Both sides of a possible mention also
    record their content ``hash`` and ``mtime``, and a source file a
    fingerprint per symbol, so drift detection can tell which side changed.
    """

    is_code_source = split_code_source_path(parsed.path) is not None
    if parsed.symbols is None:
        code: dict = {"mentions": parsed.code_mentions} if parsed.code_mentions else {}
    else:
        code = {}
        symbols = sorted({s.qualname for s in parsed.symbols if s.symbol_kind != "module"})
        if symbols:
            code["symbols"] = symbols
            code["symbol_hashes"] = _symbol_hashes(parsed.symbols)
        if parsed.imports:
            code["imports"] = parsed.imports
    if code or is_code_source:
        code["hash"] = parsed.content_hash
        code["mtime"] = parsed.mtime
    return code


def _symbol_hashes(symbols) -> dict[str, str]:
    """One fingerprint per qualname; a name defined twice (overloads, a
    conditional ``def``) gets one hash over every definition, in file order."""

    by_name: dict[str, list[str]] = {}
    for symbol in symbols:
        if symbol.symbol_kind != "module" and symbol.body_hash:
            by_name.setdefault(symbol.qualname, []).append(symbol.body_hash)
    return {
        name: hashes[0] if len(hashes) == 1 else body_fingerprint("\n".join(hashes))
        for name, hashes in sorted(by_name.items())
    }


def _in_missing_source(path: str, missing_sources: dict[str, str]) -> bool:
    repo = split_code_source_path(path)
    return repo is not None and repo[0] in missing_sources


@dataclass(slots=True)
class _PreparedNote:
    """A redacted, chunked note ready to embed; nothing is written for it yet."""

    source: Path | str
    parsed: ParsedNote
    chunks: list[ChunkRecord]
    embed_inputs: list[str]


class Indexer:
    """Coordinates the full indexing lifecycle for vault files of every included kind."""

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
        self._snapshot_vector_ownership = False
        self._vector_ids_by_path: dict[str, set[str]] | None = None

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

        files, missing_sources = iter_all_source_files(self.config)
        current_paths = {file.rel_path for file in files}

        processed = 0
        skipped = 0
        errors: list[str] = [
            f"code source {name!r}: {reason}; its index entries are kept"
            for name, reason in missing_sources.items()
        ]
        changed_paths: set[str] = set()
        deleted_paths: set[str] = set()
        old_meta: dict[str, dict | None] = {}

        to_index: list[_PreparedNote] = []
        for file in files:
            try:
                parsed = load_source_file(file, self.config)
                if mode != "full" and not self.sync_state.should_reindex(parsed.path, parsed.content_hash):
                    skipped += 1
                    continue
                to_index.append(self._prepare(parsed, source=file.abs_path))
            except Exception as exc:
                errors.append(f"{file.abs_path}: {exc}")

        deleted = 0
        self._snapshot_vector_ownership = True
        try:
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

            for tracked in self.sync_state.tracked_paths():
                if tracked in current_paths or _in_missing_source(tracked, missing_sources):
                    continue
                try:
                    old_meta[tracked] = self._delete_missing_path(tracked)
                    deleted_paths.add(tracked)
                    deleted += 1
                except Exception as exc:
                    errors.append(f"{tracked}: {exc}")
        finally:
            self._snapshot_vector_ownership = False
            self._vector_ids_by_path = None

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
        # An extension that is not included is handled like an excluded path,
        # so a file indexed before its extension was dropped gets removed.
        ineligible = is_excluded_path(
            resolved_path, vault_root, self.config.exclude_globs
        ) or loader_for(resolved_path, self.config.include_extensions) is None
        # A note deleted from disk is cleaned up like an excluded one. A path
        # that was never indexed is a mistyped argument, so it fails loudly
        # instead of being counted as skipped.
        missing = not resolved_path.is_file()
        if missing and not ineligible and rel_path not in self.sync_state.tracked_paths():
            raise FileNotFoundError(f"{rel_path} does not exist in the vault")
        if ineligible or missing:
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
                                    old_meta["link_pairs"],
                                    old_meta["code"],
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

        # The resolved root, not config.vault_path: the loader takes the note's
        # path relative to it, and resolved_path has symlinks and ``..`` removed.
        parsed = load_source(resolved_path, vault_root, self.config.include_extensions)
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
        code = _code_meta(parsed)
        if not chunks:
            return self.graph_store.upsert_note_meta(
                parsed.path, parsed.title, parsed.tags, parsed.links, None, aliases,
                parsed.link_pairs, code,
            )

        self.vector_store.ensure_collection(len(embeddings[0]))
        self.vector_store.upsert_chunks(chunks, embeddings)
        self.keyword_store.upsert_chunks(chunks)

        centroid = np.mean(np.asarray(embeddings, dtype=np.float64), axis=0).tolist()
        return self.graph_store.upsert_note_meta(
            parsed.path, parsed.title, parsed.tags, parsed.links, centroid, aliases,
            parsed.link_pairs, code,
        )

    def _chunk_ids_by_path(self, rel_path: str) -> set[str]:
        """Find ownership independently in each store for partial-write recovery."""

        keyword_ids = set(self.keyword_store.chunk_ids_by_path(rel_path))
        return keyword_ids | self._vector_chunk_ids_by_path(rel_path)

    def _vector_chunk_ids_by_path(self, rel_path: str) -> set[str]:
        """Return a path's vector ids, from one snapshot per vault scan.

        Local-mode Qdrant has no payload index, so each per-path lookup reads the
        whole collection, which made a vault scan quadratic. The snapshot is taken
        at the scan's first lookup, before any write, and stays exact for every
        later lookup: a scan looks each path up once, before writing it, and chunk
        ids hash the note path, so other notes' writes never add ids under this
        path. The only stale entries are ids another note's cleanup already
        deleted, and deleting them again is a no-op.
        """

        if not self._snapshot_vector_ownership:
            return set(self.vector_store.chunk_ids_by_path(rel_path))
        if self._vector_ids_by_path is None:
            self._vector_ids_by_path = self.vector_store.chunk_ids_grouped_by_path()
        return self._vector_ids_by_path.get(rel_path, set())

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
