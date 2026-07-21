"""End-to-end tests for GraphBuilder via RagService.sync (full and incremental)."""

import sqlite3
from pathlib import Path

import pytest

from second_brain.config import RagConfig
from second_brain.service import RagService


class _KeywordEmbedder:
    """Deterministic, content-dependent embedder for meaningful cosine tests.

    Unlike a length-based stub (whose vectors are always colinear, forcing
    cosine similarity to 1.0 for any two non-empty texts), this maps a small
    fixed vocabulary to orthogonal-ish dimensions so unrelated notes get a
    genuinely low semantic score and related notes get a genuinely high one.
    """

    _VOCAB = ["alpha", "beta", "gamma", "delta", "apple", "banana"]

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append([float(lower.count(word)) for word in self._VOCAB])
        return vectors

    def health(self) -> bool:
        return True


def _build_service(tmp_path: Path) -> RagService:
    vault = tmp_path / "vault"
    vault.mkdir()
    config = RagConfig(
        vault_path=vault,
        qdrant_path=tmp_path / "qdrant",
        fts_path=tmp_path / "fts.sqlite",
        sync_state_path=tmp_path / "sync_state.sqlite",
        chunk_size=20,
        chunk_overlap=0,
    )
    service = RagService(config, use_in_memory_vector=True)
    service.embedder = _KeywordEmbedder()
    service.indexer.embedder = _KeywordEmbedder()
    return service


def _write(vault: Path, name: str, content: str) -> None:
    (vault / name).write_text(content, encoding="utf-8")


def _edge_updated_at(fts_path: Path, a: str, b: str) -> float | None:
    src, dst = sorted([a, b])
    conn = sqlite3.connect(fts_path)
    try:
        row = conn.execute(
            "SELECT updated_at FROM edges WHERE src = ? AND dst = ?", (src, dst)
        ).fetchone()
    finally:
        conn.close()
    return row[0] if row else None


def test_full_sync_builds_mutual_link_edge(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    vault = service.config.vault_path
    _write(vault, "a.md", "# A\nsome unique text about apples [[B]]")
    _write(vault, "b.md", "# B\nsome unique text about bananas [[A]]")

    service.sync(mode="full")

    edge = service.graph_store.edge_between("a.md", "b.md")
    assert edge is not None
    assert edge.link == 1.0
    assert edge.link_src_to_dst is True
    assert edge.link_dst_to_src is True


def test_full_sync_one_way_link_scores_lower(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    vault = service.config.vault_path
    _write(vault, "a.md", "# A\nsome text about apples [[B]]")
    _write(vault, "b.md", "# B\nsome text about bananas")

    service.sync(mode="full")

    edge = service.graph_store.edge_between("a.md", "b.md")
    assert edge is not None
    assert edge.link == 0.7


def test_full_sync_semantic_signal_reflects_content_similarity(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    vault = service.config.vault_path
    _write(vault, "a.md", "# A\napple apple apple")
    _write(vault, "b.md", "# B\napple apple apple too")
    _write(vault, "c.md", "# C\nbeta beta beta only")

    service.sync(mode="full")

    ab = service.graph_store.edge_between("a.md", "b.md")
    ac = service.graph_store.edge_between("a.md", "c.md")
    assert ab is not None
    assert ab.semantic == 1.0
    assert ac is None or ac.semantic < ab.semantic


def test_full_sync_shared_tags_contribute_to_composite(tmp_path: Path) -> None:
    # Tags only ever *score* an existing candidate pair, they never generate
    # one on their own (that would create tag-cliques on popular tags) -- so
    # this pair also needs a link to become a candidate in the first place.
    service = _build_service(tmp_path)
    vault = service.config.vault_path
    _write(vault, "a.md", "# A\nsome content here [[B]]\n#project/x")
    _write(vault, "b.md", "# B\nother content here\n#project/x")

    service.sync(mode="full")

    edge = service.graph_store.edge_between("a.md", "b.md")
    assert edge is not None
    assert edge.tag == 1.0
    assert "project/x" in edge.shared_tags


def test_tags_alone_do_not_create_an_edge(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    vault = service.config.vault_path
    _write(vault, "a.md", "# A\nsome content here\n#project/x")
    _write(vault, "b.md", "# B\nother content here\n#project/x")

    service.sync(mode="full")

    assert service.graph_store.edge_between("a.md", "b.md") is None


def test_comention_creates_edge_between_unlinked_notes(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    vault = service.config.vault_path
    _write(vault, "a.md", "# A\ncontent about topic one")
    _write(vault, "b.md", "# B\ncontent about topic two")
    _write(vault, "hub.md", "# Hub\nsee [[A]] and [[B]]")

    service.sync(mode="full")

    edge = service.graph_store.edge_between("a.md", "b.md")
    assert edge is not None
    assert edge.comention_count == 1


def test_empty_and_single_note_vault_do_not_crash(tmp_path: Path) -> None:
    service = _build_service(tmp_path)

    result = service.sync(mode="full")
    assert result["errors"] == []
    assert service.graph_store.counts()["edges"] == 0

    _write(service.config.vault_path, "only.md", "# Only\nsolo note")
    result = service.sync(mode="full")
    assert result["errors"] == []
    assert service.graph_store.counts()["edges"] == 0
    assert service.graph_store.note_meta_for("only.md") is not None


def test_incremental_sync_leaves_unaffected_edges_untouched(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    vault = service.config.vault_path
    _write(vault, "a.md", "# A\nalpha alpha alpha [[B]]")
    _write(vault, "b.md", "# B\nbeta beta beta [[A]]")
    _write(vault, "c.md", "# C\ngamma gamma gamma [[D]]")
    _write(vault, "d.md", "# D\ndelta delta delta [[C]]")
    service.sync(mode="full")

    before = _edge_updated_at(service.config.fts_path, "c.md", "d.md")
    assert before is not None

    _write(vault, "a.md", "# A\nalpha alpha alpha updated [[B]]")
    service.sync(mode="incremental")

    after = _edge_updated_at(service.config.fts_path, "c.md", "d.md")
    assert after == before


def test_incremental_delete_removes_edges_and_meta(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    vault = service.config.vault_path
    _write(vault, "a.md", "# A\nalpha content [[B]]")
    _write(vault, "b.md", "# B\nbeta content [[A]]")
    service.sync(mode="full")
    assert service.graph_store.edge_between("a.md", "b.md") is not None

    (vault / "a.md").unlink()
    result = service.sync(mode="incremental")

    assert result["deleted"] == 1
    assert service.graph_store.note_meta_for("a.md") is None
    assert service.graph_store.edge_between("a.md", "b.md") is None


def test_incremental_delete_reduces_comention_count(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    vault = service.config.vault_path
    _write(vault, "a.md", "# A\ncontent one")
    _write(vault, "b.md", "# B\ncontent two")
    _write(vault, "hub.md", "# Hub\nsee [[A]] and [[B]]")
    _write(vault, "hub2.md", "# Hub2\nalso mentions [[A]] and [[B]]")
    service.sync(mode="full")
    edge_before = service.graph_store.edge_between("a.md", "b.md")
    assert edge_before is not None
    assert edge_before.comention_count == 2

    (vault / "hub.md").unlink()
    service.sync(mode="incremental")

    edge_after = service.graph_store.edge_between("a.md", "b.md")
    assert edge_after is not None
    assert edge_after.comention_count == 1


def test_graph_enabled_false_via_service_builds_no_edges(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    service.config.graph_enabled = False
    vault = service.config.vault_path
    _write(vault, "a.md", "# A\nalpha content [[B]]")
    _write(vault, "b.md", "# B\nbeta content [[A]]")

    service.sync(mode="full")

    assert service.graph_store.counts()["edges"] == 0


# -- related / connections / graph_map -------------------------------------


def test_related_unknown_note_reports_not_found(tmp_path: Path) -> None:
    service = _build_service(tmp_path)

    result = service.related("nope.md")

    assert result == {"note_path": "nope.md", "found": False, "neighbors": []}


def test_related_rejects_blank_path_and_bad_top_k(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    with pytest.raises(ValueError, match="note_path"):
        service.related("  ")
    with pytest.raises(ValueError, match="top_k"):
        service.related("a.md", top_k=0)
    with pytest.raises(ValueError, match="top_k"):
        service.related("a.md", top_k=999)


def test_related_returns_neighbors_with_signal_breakdown(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    vault = service.config.vault_path
    _write(vault, "a.md", "# A\nalpha alpha alpha [[B]]")
    _write(vault, "b.md", "# B\nbeta beta beta [[A]]")
    service.sync(mode="full")

    result = service.related("a.md")

    assert result["found"] is True
    assert len(result["neighbors"]) == 1
    neighbor = result["neighbors"][0]
    assert neighbor["path"] == "b.md"
    assert neighbor["signals"]["link"] == 1.0
    assert neighbor["evidence"]["links_to"] is True
    assert neighbor["evidence"]["linked_from"] is True


def test_connections_rejects_blank_or_identical_paths(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    with pytest.raises(ValueError, match="empty"):
        service.connections("a.md", "  ")
    with pytest.raises(ValueError, match="different"):
        service.connections("a.md", "a.md")


def test_connections_direct_edge(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    vault = service.config.vault_path
    _write(vault, "a.md", "# A\nalpha alpha alpha [[B]]")
    _write(vault, "b.md", "# B\nbeta beta beta [[A]]")
    service.sync(mode="full")

    result = service.connections("a.md", "b.md")

    assert result["connected"] is True
    assert result["direct_edge"] is not None
    assert result["closeness"] == result["direct_edge"]["composite"]
    assert result["path"] == ["a.md", "b.md"]


def test_connections_two_hop_path_when_no_direct_edge(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    vault = service.config.vault_path
    # A and C both link to Mid, but Mid links to neither -- so Mid never
    # co-mentions {A, C} together and A/C get no direct edge, only a
    # two-hop chain through their shared (one-way-linked) neighbor.
    _write(vault, "a.md", "# A\nalpha alpha alpha [[Mid]]")
    _write(vault, "mid.md", "# Mid\nsome bridging note with no outgoing links")
    _write(vault, "c.md", "# C\ngamma gamma gamma [[Mid]]")
    service.sync(mode="full")

    assert service.graph_store.edge_between("a.md", "c.md") is None

    result = service.connections("a.md", "c.md")

    assert result["connected"] is True
    assert result["direct_edge"] is None
    assert result["path"] == ["a.md", "mid.md", "c.md"]
    assert len(result["path_edges"]) == 2


def test_connections_disconnected_pair(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    vault = service.config.vault_path
    _write(vault, "a.md", "# A\nalpha alpha alpha")
    _write(vault, "b.md", "# B\nbeta beta beta")
    service.sync(mode="full")

    result = service.connections("a.md", "b.md")

    assert result["connected"] is False
    assert result["closeness"] == 0.0
    assert result["path"] is None


def test_connections_unknown_note_reports_not_found(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    vault = service.config.vault_path
    _write(vault, "a.md", "# A\nalpha alpha alpha")
    service.sync(mode="full")

    result = service.connections("a.md", "ghost.md")

    assert result["found_a"] is True
    assert result["found_b"] is False
    assert result["connected"] is False


def test_graph_map_rejects_out_of_range_min_score(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    with pytest.raises(ValueError, match="min_score"):
        service.graph_map(min_score=1.5)
    with pytest.raises(ValueError, match="min_score"):
        service.graph_map(min_score=-0.1)


def test_graph_map_clusters_orphans_and_bridge(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    vault = service.config.vault_path
    # Clique one: a <-> b (mutually linked).
    _write(vault, "a.md", "# A\nalpha alpha alpha [[B]]\n#groupone")
    _write(vault, "b.md", "# B\nalpha alpha too [[A]]\n#groupone")
    # Clique two: d <-> e (mutually linked).
    _write(vault, "d.md", "# D\ndelta delta delta [[E]]\n#grouptwo")
    _write(vault, "e.md", "# E\ndelta delta too [[D]]\n#grouptwo")
    # Bridge note connecting both cliques.
    _write(vault, "bridge.md", "# Bridge\nsee [[A]] and [[D]]")
    _write(vault, "bridge2.md", "# Bridge2\nalso see [[A]] and [[D]]")
    # Fully isolated note.
    _write(vault, "lonely.md", "# Lonely\nzzz nothing shared zzz")
    service.sync(mode="full")

    result = service.graph_map()

    assert result["note_count"] == 7
    assert "lonely.md" in result["orphans"]
    cluster_sizes = sorted(c["size"] for c in result["clusters"])
    assert len(result["clusters"]) >= 1
    assert sum(cluster_sizes) >= 4


def test_incremental_sync_adds_edge_to_note_with_spare_kNN_capacity(
    tmp_path: Path,
) -> None:
    """Regression test for a reverse-kNN threshold bug: a candidate note
    with fewer than graph_knn_k semantic edges should only need to clear
    graph_semantic_min to gain a new neighbor, not beat its current
    strongest edge.

    y.md starts with exactly one (very strong) semantic edge to z.md, far
    under its knn_k=8 capacity. x.md is then added with weaker-but-above-
    threshold similarity to y.md, while ranking 8 *other* notes (f1..f8)
    higher than y.md from its own perspective -- so x.md's own top-k
    candidate generation alone would never propose the x-y pair; only a
    correct reverse-kNN check surfaces it.
    """
    service = _build_service(tmp_path)
    vault = service.config.vault_path

    _write(vault, "y.md", "# Y\n" + "alpha " * 10)
    _write(vault, "z.md", "# Z\n" + "alpha " * 10)
    service.sync(mode="full")
    assert len(service.graph_store.edges_for("y.md")) == 1

    for i in range(1, 9):
        _write(vault, f"f{i}.md", f"# F{i}\n" + "beta " * 10)
    service.sync(mode="incremental")

    # x.md: 2 "alpha" (cosine to y/z ~0.37, above the 0.35 default
    # semantic_min) and 5 "beta" (cosine to each filler ~0.93 -- ranks all
    # 8 fillers above y.md from x.md's own perspective).
    _write(vault, "x.md", "# X\nalpha alpha beta beta beta beta beta")
    service.sync(mode="incremental")

    edge = service.graph_store.edge_between("x.md", "y.md")
    assert edge is not None
    assert edge.semantic == pytest.approx(0.3714, abs=1e-3)
