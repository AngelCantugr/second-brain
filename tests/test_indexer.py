from pathlib import Path

import pytest

from second_brain.config import DEFAULT_EXCLUDE_GLOBS, RagConfig
from second_brain.graph import GraphStore
from second_brain.indexer import Indexer
from second_brain.keyword_store import KeywordStore
from second_brain.parser import parse_note
from second_brain.sync_state import SyncStateStore
from second_brain.vector_store import InMemoryVectorStore


class _StubEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]


def _build_indexer(tmp_path: Path, vault_path: Path) -> Indexer:
    config = RagConfig(
        vault_path=vault_path,
        qdrant_path=tmp_path / "qdrant",
        fts_path=tmp_path / "fts.sqlite",
        sync_state_path=tmp_path / "sync_state.sqlite",
        chunk_size=20,
        chunk_overlap=0,
    )
    indexer = Indexer(
        config=config,
        embedder=_StubEmbedder(),
        vector_store=InMemoryVectorStore(),
        keyword_store=KeywordStore(config.fts_path),
        sync_state=SyncStateStore(config.sync_state_path),
        graph_store=GraphStore(config.fts_path),
    )
    indexer.initialize()
    return indexer


def test_sync_rejects_unknown_mode(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    indexer = _build_indexer(tmp_path, vault)

    with pytest.raises(ValueError, match="mode"):
        indexer.sync(mode="bogus_mode")


def test_file_mode_requires_file_path(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("hello", encoding="utf-8")
    indexer = _build_indexer(tmp_path, vault)

    with pytest.raises(ValueError, match="file_path"):
        indexer.sync(mode="file")


def test_file_mode_accepts_path_relative_to_vault(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# A\nhello world", encoding="utf-8")
    (vault / "b.md").write_text("# B\nother note", encoding="utf-8")
    indexer = _build_indexer(tmp_path, vault)

    result = indexer.sync(mode="file", file_path="a.md")

    assert result.processed == 1
    assert result.skipped == 0
    assert result.deleted == 0
    assert result.errors == []
    assert indexer.sync_state.tracked_paths() == ["a.md"]


def test_file_mode_skips_default_exclusions_and_keeps_explicit_empty_override(
    tmp_path: Path,
) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    instruction = vault / "CLAUDE.md"
    instruction.write_text("# Private instructions\nneedle", encoding="utf-8")
    hidden = vault / ".agent" / "config.md"
    hidden.parent.mkdir()
    hidden.write_text("# Hidden config\nneedle", encoding="utf-8")
    indexer = _build_indexer(tmp_path, vault)

    excluded = indexer.sync(mode="file", file_path=str(instruction))
    hidden_result = indexer.sync(mode="file", file_path=str(hidden))

    assert excluded.processed == 0
    assert excluded.skipped == 1
    assert hidden_result.processed == 0
    assert hidden_result.skipped == 1
    assert indexer.sync_state.tracked_paths() == []
    assert indexer.vector_store.get_by_path("CLAUDE.md") == []

    indexer.config.exclude_globs = []
    included = indexer.sync(mode="file", file_path=str(instruction))

    assert included.processed == 1
    assert indexer.sync_state.tracked_paths() == ["CLAUDE.md"]
    assert len(indexer.vector_store.get_by_path("CLAUDE.md")) == 1


def test_file_sync_removes_newly_excluded_path_from_all_stores_and_retries_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "active.md"
    note.write_text("# Active\n[[Other]]\nsearchable needle", encoding="utf-8")
    (vault / "other.md").write_text("# Other\nlinked target", encoding="utf-8")
    indexer = _build_indexer(tmp_path, vault)
    assert indexer.sync(mode="full").errors == []
    assert indexer.graph_store.counts()["edges"] == 1

    # Model an interrupted replacement too: pending-only paths participate in
    # the same cleanup retry contract as successfully checkpointed notes.
    indexer.sync_state.mark_pending("active.md")
    indexer.config.exclude_globs = ["active.md"]
    original_delete = indexer.keyword_store.delete_chunks
    attempts = 0

    def fail_once(chunk_ids: list[str]) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary keyword deletion failure")
        original_delete(chunk_ids)

    monkeypatch.setattr(indexer.keyword_store, "delete_chunks", fail_once)

    failed = indexer.sync(mode="file", file_path=str(note))

    assert failed.deleted == 0
    assert failed.errors
    assert indexer.sync_state.tracked_paths() == ["active.md", "other.md"]
    assert indexer.graph_store.note_meta_for("active.md") is not None

    removed = indexer.sync(mode="file", file_path=str(note))

    assert removed.deleted == 1
    assert removed.errors == []
    assert indexer.sync_state.tracked_paths() == ["other.md"]
    assert indexer.vector_store.get_by_path("active.md") == []
    assert indexer.keyword_store.chunk_ids_by_path("active.md") == []
    assert indexer.graph_store.note_meta_for("active.md") is None
    assert indexer.graph_store.counts()["edges"] == 0


def test_file_sync_cleans_pending_only_excluded_path(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "_templates-old" / "pending.md"
    note.parent.mkdir()
    note.write_text("# Template\nsearchable needle", encoding="utf-8")
    indexer = _build_indexer(tmp_path, vault)
    indexer._upsert_parsed(parse_note(note, vault))
    indexer.config.exclude_globs = list(DEFAULT_EXCLUDE_GLOBS)

    result = indexer.sync(mode="file", file_path=str(note))

    assert result.deleted == 1
    assert indexer.sync_state.tracked_paths() == []
    assert indexer.vector_store.get_by_path("_templates-old/pending.md") == []
    assert indexer.keyword_store.chunk_ids_by_path("_templates-old/pending.md") == []
    assert indexer.graph_store.note_meta_for("_templates-old/pending.md") is None


def test_file_sync_retries_graph_cleanup_with_metadata_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "active.md"
    note.write_text("# Active\n[[Other]]\nsearchable needle", encoding="utf-8")
    (vault / "other.md").write_text("# Other\nlinked target", encoding="utf-8")
    indexer = _build_indexer(tmp_path, vault)
    assert indexer.sync(mode="full").errors == []
    indexer.config.exclude_globs = ["active.md"]
    original_update = indexer.graph_builder.update_for_changes
    attempts = 0

    def fail_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary graph update failure")
        return original_update(*args, **kwargs)

    monkeypatch.setattr(indexer.graph_builder, "update_for_changes", fail_once)

    failed = indexer.sync(mode="file", file_path=str(note))

    assert failed.errors
    assert indexer.sync_state.tracked_paths() == ["active.md", "other.md"]
    assert indexer.graph_store.note_meta_for("active.md") is not None
    assert indexer.graph_store.counts()["edges"] == 1

    retried = indexer.sync(mode="file", file_path=str(note))

    assert retried.deleted == 1
    assert retried.errors == []
    assert indexer.sync_state.tracked_paths() == ["other.md"]
    assert indexer.graph_store.note_meta_for("active.md") is None
    assert indexer.graph_store.counts()["edges"] == 0


def test_file_sync_skips_excluded_content_in_qdrant_and_sqlite_stores(
    tmp_path: Path,
) -> None:
    from second_brain.vector_store import QdrantVectorStore

    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "CLAUDE.md"
    note.write_text("# Instructions\nsearchable needle", encoding="utf-8")
    config = RagConfig(
        vault_path=vault,
        qdrant_path=tmp_path / "qdrant",
        fts_path=tmp_path / "fts.sqlite",
        sync_state_path=tmp_path / "sync_state.sqlite",
    )
    vector_store = QdrantVectorStore(config.qdrant_path, "chunks")
    indexer = Indexer(
        config=config,
        embedder=_StubEmbedder(),
        vector_store=vector_store,
        keyword_store=KeywordStore(config.fts_path),
        sync_state=SyncStateStore(config.sync_state_path),
        graph_store=GraphStore(config.fts_path),
    )
    indexer.initialize()

    try:
        result = indexer.sync(mode="file", file_path=str(note))

        assert result.processed == 0
        assert result.skipped == 1
        assert vector_store.get_by_path("CLAUDE.md") == []
        assert indexer.keyword_store.chunk_ids_by_path("CLAUDE.md") == []
        assert indexer.graph_store.note_meta_for("CLAUDE.md") is None
        assert indexer.sync_state.tracked_paths() == []
    finally:
        vector_store.client.close()


def test_full_sync_captures_note_centroid_and_meta(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# A\nhello world [[B]]\n#project", encoding="utf-8")
    (vault / "b.md").write_text("# B\nother note content here", encoding="utf-8")
    indexer = _build_indexer(tmp_path, vault)

    indexer.sync(mode="full")

    meta = indexer.graph_store.note_meta_for("a.md")
    assert meta is not None
    assert meta["centroid"] is not None
    assert meta["title"] == "a"
    assert meta["links"] == ["B"]
    assert meta["tags"] == ["project"]


def test_chunkless_note_still_gets_meta_with_null_centroid(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "empty.md").write_text("---\ntags: [x]\n---\n", encoding="utf-8")
    indexer = _build_indexer(tmp_path, vault)

    indexer.sync(mode="full")

    meta = indexer.graph_store.note_meta_for("empty.md")
    assert meta is not None
    assert meta["centroid"] is None
    assert meta["dim"] is None


def test_graph_disabled_skips_edge_build(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# A\nhello world [[B]]", encoding="utf-8")
    (vault / "b.md").write_text("# B\nother note content", encoding="utf-8")
    indexer = _build_indexer(tmp_path, vault)
    indexer.config.graph_enabled = False

    result = indexer.sync(mode="full")

    assert result.graph_edges_updated == 0
    assert indexer.graph_store.counts()["edges"] == 0
    # note_meta capture is independent of graph_enabled -- it just feeds
    # nothing into edge computation when disabled.
    assert indexer.graph_store.note_meta_for("a.md") is not None


def test_graph_build_failure_is_recorded_without_failing_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# A\nhello world", encoding="utf-8")
    indexer = _build_indexer(tmp_path, vault)

    def _boom(*args, **kwargs):
        raise RuntimeError("graph explosion")

    monkeypatch.setattr(indexer.graph_builder, "rebuild_full", _boom)

    result = indexer.sync(mode="full")

    assert result.processed == 1
    assert any("graph build" in err for err in result.errors)


def test_incremental_sync_replaces_stale_chunks_in_qdrant_and_fts(
    tmp_path: Path,
) -> None:
    from second_brain.vector_store import QdrantVectorStore

    vault = tmp_path / "vault"
    vault.mkdir()
    note_path = vault / "edited.md"
    note_path.write_text(
        "# Edited\n" + " ".join(f"oldword{i}" for i in range(25)),
        encoding="utf-8",
    )
    config = RagConfig(
        vault_path=vault,
        qdrant_path=tmp_path / "qdrant",
        fts_path=tmp_path / "fts.sqlite",
        sync_state_path=tmp_path / "sync_state.sqlite",
        chunk_size=20,
        chunk_overlap=0,
    )
    vector_store = QdrantVectorStore(config.qdrant_path, "chunks")
    indexer = Indexer(
        config=config,
        embedder=_StubEmbedder(),
        vector_store=vector_store,
        keyword_store=KeywordStore(config.fts_path),
        sync_state=SyncStateStore(config.sync_state_path),
        graph_store=GraphStore(config.fts_path),
    )
    indexer.initialize()

    try:
        initial = indexer.sync(mode="full")
        assert initial.errors == []
        old_ids = {
            hit.chunk_id for hit in vector_store.get_by_path("edited.md")
        }
        assert len(old_ids) == 2
        assert {
            hit.chunk_id
            for hit in indexer.keyword_store.chunks_by_path("edited.md")
        } == old_ids

        note_path.write_text(
            "# Edited\nfreshneedle one two three",
            encoding="utf-8",
        )
        edited = indexer.sync(mode="incremental")
        assert edited.errors == []
        new_hits = vector_store.get_by_path("edited.md")
        assert len(new_hits) == 1
        new_id = new_hits[0].chunk_id
        assert new_id not in old_ids
        assert indexer.keyword_store.chunk_ids_by_path("edited.md") == [new_id]
        assert len(indexer.keyword_store.search("freshneedle", limit=10)) == 1

        note_path.write_text("---\ntags: [x]\n---\n", encoding="utf-8")
        emptied = indexer.sync(mode="incremental")
        assert emptied.errors == []
        assert vector_store.get_by_path("edited.md") == []
        assert indexer.keyword_store.chunk_ids_by_path("edited.md") == []
    finally:
        vector_store.client.close()
