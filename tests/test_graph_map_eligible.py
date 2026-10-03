"""Stored-note regressions for transient, exclusion-aware map evidence."""

from pathlib import Path
import sqlite3

import pytest

from second_brain.config import RagConfig
from second_brain.service import RagService


@pytest.fixture
def service(tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    config = RagConfig(
        vault_path=vault, qdrant_path=tmp_path / "qdrant",
        fts_path=tmp_path / "fts.sqlite", sync_state_path=tmp_path / "sync.sqlite",
        exclude_globs=[], graph_weight_semantic=0, graph_weight_link=0,
        graph_weight_tag=0, graph_weight_comention=1, graph_comention_cap=1,
    )
    return RagService(config, use_in_memory_vector=True)


def _store_note(service: RagService, path: str, links=None, centroid=None) -> None:
    service.graph_store.upsert_note_meta(
        path, Path(path).stem, [], links or [], centroid,
    )


def _stored_snapshot(service: RagService) -> tuple:
    """Compare stored rows, including timestamps, rather than only edge counts."""
    with sqlite3.connect(f"file:{service.config.fts_path}?mode=ro", uri=True) as conn:
        rows = tuple(
            tuple(conn.execute(sql).fetchall())
            for sql in [
                "SELECT * FROM note_meta ORDER BY path",
                "SELECT * FROM edges ORDER BY src, dst",
                "SELECT * FROM chunks ORDER BY chunk_id",
                "SELECT * FROM chunks_fts ORDER BY rowid",
            ]
        )
    with sqlite3.connect(f"file:{service.config.sync_state_path}?mode=ro", uri=True) as conn:
        state = tuple(conn.execute("SELECT * FROM note_state ORDER BY path").fetchall())
    return rows, state, service.graph_store.counts()


def test_map_excluded_only_comention_source_cannot_cluster_survivors(service: RagService) -> None:
    _store_note(service, "a.md", centroid=[1.0, 0.0])
    _store_note(service, "b.md", centroid=[0.0, 1.0])
    _store_note(service, "CLAUDE.md", links=["a", "b"])
    service.indexer.graph_builder.rebuild_full()
    stored = service.graph_store.edge_between("a.md", "b.md")
    assert stored.comention == 1.0
    assert stored.semantic == stored.link == stored.tag == 0
    service.config.exclude_globs = ["CLAUDE.md"]
    before = _stored_snapshot(service)

    result = service.graph_map()

    assert result == {
        "note_count": 2, "edge_count": 0, "clusters": [],
        "orphans": ["a.md", "b.md"], "bridges": [],
    }
    assert _stored_snapshot(service) == before


def test_map_mixed_comention_sources_use_only_eligible_count_and_cap(service: RagService) -> None:
    service.config.graph_comention_cap = 3
    for path in ["a.md", "b.md"]:
        _store_note(service, path)
    for path in ["source.md", "CLAUDE.md", "README.md"]:
        _store_note(service, path, links=["a", "b"])
    service.indexer.graph_builder.rebuild_full()
    stored = service.graph_store.edge_between("a.md", "b.md")
    assert stored.comention_count == 3
    assert stored.comention == 1.0
    service.config.exclude_globs = ["CLAUDE.md", "README.md"]
    before = _stored_snapshot(service)

    result = service.graph_map(min_score=0.5)
    assert result["edge_count"] == 0
    assert result["clusters"] == []
    assert result["orphans"] == ["a.md", "b.md", "source.md"]
    assert result["bridges"] == []
    lower_cutoff = service.graph_map(min_score=0.3)
    assert lower_cutoff["edge_count"] == 1
    assert lower_cutoff["clusters"][0]["notes"] == ["a.md", "b.md"]
    assert lower_cutoff["orphans"] == ["source.md"]
    assert _stored_snapshot(service) == before


def test_map_excluded_semantic_neighbors_cannot_consume_knn_slots(service: RagService) -> None:
    service.config.graph_weight_semantic = 1.0
    service.config.graph_weight_comention = 0.0
    service.config.graph_knn_k = 1
    for path, centroid in [
        ("a.md", [1.0, 0.0]), ("b.md", [0.8, 0.6]),
        ("CLAUDE.md", [1.0, 0.0]), ("README.md", [0.8, 0.6]),
    ]:
        _store_note(service, path, centroid=centroid)
    service.indexer.graph_builder.rebuild_full()
    assert service.graph_store.edge_between("a.md", "b.md") is None
    service.config.exclude_globs = ["CLAUDE.md", "README.md"]
    before = _stored_snapshot(service)

    result = service.graph_map()

    assert result["note_count"] == 2
    assert result["edge_count"] == 1
    assert result["clusters"][0]["notes"] == ["a.md", "b.md"]
    assert result["orphans"] == result["bridges"] == []
    assert service.graph_map(min_score=0.9)["edge_count"] == 0
    assert _stored_snapshot(service) == before


@pytest.mark.parametrize("max_fanout, edges", [(1, 0), (2, 1)])
def test_map_comention_fanout_counts_only_eligible_targets(
    service: RagService, max_fanout: int, edges: int,
) -> None:
    service.config.graph_comention_max_fanout = max_fanout
    for path in ["a.md", "b.md", "CLAUDE.md"]:
        _store_note(service, path)
    _store_note(service, "source.md", links=["a", "b", "CLAUDE"])
    service.indexer.graph_builder.rebuild_full()
    service.config.exclude_globs = ["CLAUDE.md"]
    before = _stored_snapshot(service)
    assert service.graph_map()["edge_count"] == edges
    assert _stored_snapshot(service) == before


@pytest.mark.parametrize("stored_paths", [[], ["CLAUDE.md"]], ids=["empty", "all-excluded"])
def test_map_empty_eligible_view_is_read_only(service: RagService, stored_paths: list[str]) -> None:
    for path in stored_paths:
        _store_note(service, path, centroid=[1.0, 0.0])
    service.config.exclude_globs = ["CLAUDE.md"]
    before = _stored_snapshot(service)
    assert service.graph_map() == {
        "note_count": 0, "edge_count": 0, "clusters": [], "orphans": [], "bridges": [],
    }
    assert _stored_snapshot(service) == before


def test_map_disabled_graph_returns_eligible_orphans_without_computation(service: RagService) -> None:
    _store_note(service, "a.md", links=["b"])
    _store_note(service, "b.md", links=["a"])
    service.config.graph_weight_link = 1.0
    service.config.graph_weight_comention = 0.0
    service.indexer.graph_builder.rebuild_full()
    assert service.graph_store.edge_between("a.md", "b.md").link == 1.0
    service.config.graph_enabled = False
    before = _stored_snapshot(service)
    assert service.graph_map() == {
        "note_count": 2, "edge_count": 0, "clusters": [],
        "orphans": ["a.md", "b.md"], "bridges": [],
    }
    assert _stored_snapshot(service) == before


def test_map_needs_no_indexing_persisted_edges_or_extra_centroid_population(
    service: RagService, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _store_note(service, "a.md", centroid=[1.0, 0.0])
    _store_note(service, "b.md", centroid=[1.0, 0.0])
    service.config.graph_weight_semantic = 1.0
    service.config.graph_weight_comention = 0.0
    before = _stored_snapshot(service)

    def unexpected_work(*args, **kwargs):
        raise AssertionError("map attempted indexing, persistence, or unfiltered population")

    monkeypatch.setattr(service.indexer, "sync", unexpected_work)
    monkeypatch.setattr(service.embedder, "embed", unexpected_work)
    for method in ["all_edges", "all_centroids", "replace_all_edges", "replace_edges_for_paths"]:
        monkeypatch.setattr(service.graph_store, method, unexpected_work)
    result = service.graph_map()
    assert result["edge_count"] == 1
    assert result["clusters"][0]["notes"] == ["a.md", "b.md"]
    assert _stored_snapshot(service) == before
