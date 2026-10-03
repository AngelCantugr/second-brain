from pathlib import Path

import pytest

from second_brain.config import RagConfig
from second_brain.graph import GraphStore
from second_brain.indexer import Indexer
from second_brain.keyword_store import KeywordStore
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
