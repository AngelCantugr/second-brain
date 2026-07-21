import sqlite3
from pathlib import Path

import numpy as np
import pytest

from second_brain.graph import (
    GraphStore,
    canonical_pair,
    comention_component,
    comention_pairs,
    composite_score,
    cosine,
    cosine_matrix,
    link_component,
    normalize_link_title,
    tag_jaccard,
)
from second_brain.models import Edge


def test_normalize_link_title_strips_fragment_case_and_whitespace() -> None:
    assert normalize_link_title("Project X#Overview") == "project x"
    assert normalize_link_title("  Note B  ") == "note b"
    assert normalize_link_title("note") == "note"


def test_cosine_identical_and_orthogonal_vectors() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_zero_vector_is_zero_not_nan() -> None:
    assert cosine([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_cosine_matrix_matches_pairwise_cosine() -> None:
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    matrix = cosine_matrix(vectors)
    assert matrix[0, 1] == pytest.approx(cosine([1.0, 0.0], [0.0, 1.0]))
    assert matrix[0, 2] == pytest.approx(cosine([1.0, 0.0], [1.0, 1.0]))
    assert matrix[0, 0] == pytest.approx(1.0)


def test_cosine_matrix_handles_zero_row() -> None:
    vectors = np.array([[0.0, 0.0], [1.0, 0.0]])
    matrix = cosine_matrix(vectors)
    assert matrix[0, 1] == 0.0
    assert matrix[1, 0] == 0.0


def test_tag_jaccard_overlap_disjoint_and_empty() -> None:
    assert tag_jaccard(["a", "b"], ["a", "b"]) == pytest.approx(1.0)
    assert tag_jaccard(["a"], ["b"]) == 0.0
    assert tag_jaccard([], []) == 0.0
    assert tag_jaccard(["A", "b"], ["a", "B"]) == pytest.approx(1.0)


def test_tag_jaccard_partial_overlap() -> None:
    assert tag_jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)


def test_link_component_mutual_one_way_and_none() -> None:
    assert link_component(True, True) == 1.0
    assert link_component(True, False) == 0.7
    assert link_component(False, True) == 0.7
    assert link_component(False, False) == 0.0


def test_comention_component_saturates_at_cap() -> None:
    assert comention_component(0, cap=3) == 0.0
    assert comention_component(1, cap=3) == pytest.approx(1 / 3)
    assert comention_component(3, cap=3) == pytest.approx(1.0)
    assert comention_component(10, cap=3) == 1.0


def test_comention_component_zero_cap_is_zero() -> None:
    assert comention_component(5, cap=0) == 0.0


def test_composite_score_weighted_sum() -> None:
    score = composite_score(
        semantic=0.8, link=1.0, tag=0.5, comention=0.0,
        weights=(0.5, 0.25, 0.15, 0.10),
    )
    assert score == pytest.approx(0.5 * 0.8 + 0.25 * 1.0 + 0.15 * 0.5)


def test_composite_score_zero_weights_zero_score() -> None:
    assert composite_score(1.0, 1.0, 1.0, 1.0, weights=(0, 0, 0, 0)) == 0.0


def test_comention_pairs_canonical_ordering_and_dedup() -> None:
    pairs = comention_pairs(["b.md", "a.md", "c.md"], max_fanout=20)
    assert pairs == {("a.md", "b.md"), ("a.md", "c.md"), ("b.md", "c.md")}


def test_comention_pairs_single_link_produces_no_pairs() -> None:
    assert comention_pairs(["a.md"], max_fanout=20) == set()


def test_comention_pairs_fanout_cap_excludes_hub_notes() -> None:
    links = [f"note{i}.md" for i in range(25)]
    assert comention_pairs(links, max_fanout=20) == set()


def test_comention_pairs_deduplicates_repeated_links() -> None:
    pairs = comention_pairs(["a.md", "a.md", "b.md"], max_fanout=20)
    assert pairs == {("a.md", "b.md")}


def test_canonical_pair_orders_lexicographically() -> None:
    assert canonical_pair("b.md", "a.md") == ("a.md", "b.md")
    assert canonical_pair("a.md", "b.md") == ("a.md", "b.md")


def _make_edge(src: str, dst: str, composite: float = 0.5, **overrides) -> Edge:
    src, dst = canonical_pair(src, dst)
    defaults = dict(
        src=src,
        dst=dst,
        semantic=0.4,
        link=0.0,
        tag=0.0,
        comention=0.0,
        comention_count=0,
        link_src_to_dst=False,
        link_dst_to_src=False,
        shared_tags=[],
        composite=composite,
    )
    defaults.update(overrides)
    return Edge(**defaults)


def test_graph_store_initialize_is_idempotent(tmp_path: Path) -> None:
    store = GraphStore(tmp_path / "fts.sqlite")
    store.initialize()
    store.initialize()  # should not raise
    assert store.counts() == {"nodes": 0, "edges": 0, "last_built": None}


def test_note_meta_round_trip_with_centroid(tmp_path: Path) -> None:
    store = GraphStore(tmp_path / "fts.sqlite")
    store.initialize()

    previous = store.upsert_note_meta(
        "a.md", "A", ["tag1"], ["b"], [1.0, 2.0, 3.0]
    )
    assert previous is None

    meta = store.note_meta_for("a.md")
    assert meta["title"] == "A"
    assert meta["title_key"] == "a"
    assert meta["tags"] == ["tag1"]
    assert meta["links"] == ["b"]
    assert meta["centroid"] == pytest.approx([1.0, 2.0, 3.0])
    assert meta["dim"] == 3


def test_note_meta_round_trip_without_centroid(tmp_path: Path) -> None:
    store = GraphStore(tmp_path / "fts.sqlite")
    store.initialize()

    store.upsert_note_meta("empty.md", "Empty", [], [], None)
    meta = store.note_meta_for("empty.md")
    assert meta["centroid"] is None
    assert meta["dim"] is None


def test_upsert_note_meta_returns_previous_row(tmp_path: Path) -> None:
    store = GraphStore(tmp_path / "fts.sqlite")
    store.initialize()

    store.upsert_note_meta("a.md", "A", ["x"], ["old-link"], [1.0])
    previous = store.upsert_note_meta("a.md", "A", ["y"], ["new-link"], [2.0])

    assert previous["links"] == ["old-link"]
    assert previous["tags"] == ["x"]
    current = store.note_meta_for("a.md")
    assert current["links"] == ["new-link"]


def test_delete_note_meta_returns_removed_row_and_clears(tmp_path: Path) -> None:
    store = GraphStore(tmp_path / "fts.sqlite")
    store.initialize()
    store.upsert_note_meta("a.md", "A", [], ["b"], [1.0])

    removed = store.delete_note_meta("a.md")

    assert removed["links"] == ["b"]
    assert store.note_meta_for("a.md") is None


def test_paths_by_title_key_supports_collisions(tmp_path: Path) -> None:
    store = GraphStore(tmp_path / "fts.sqlite")
    store.initialize()
    store.upsert_note_meta("folder1/dup.md", "Dup", [], [], None)
    store.upsert_note_meta("folder2/dup.md", "Dup", [], [], None)

    paths = store.paths_by_title_key("dup")

    assert sorted(paths) == ["folder1/dup.md", "folder2/dup.md"]


def test_all_centroids_returns_only_embedded_notes(tmp_path: Path) -> None:
    store = GraphStore(tmp_path / "fts.sqlite")
    store.initialize()
    store.upsert_note_meta("a.md", "A", [], [], [1.0, 0.0])
    store.upsert_note_meta("b.md", "B", [], [], [0.0, 1.0])
    store.upsert_note_meta("empty.md", "Empty", [], [], None)

    paths, matrix = store.all_centroids()

    assert sorted(paths) == ["a.md", "b.md"]
    assert matrix.shape == (2, 2)


def test_replace_edges_for_clears_both_directions(tmp_path: Path) -> None:
    store = GraphStore(tmp_path / "fts.sqlite")
    store.initialize()
    store.replace_all_edges(
        [_make_edge("a.md", "b.md"), _make_edge("b.md", "c.md")]
    )

    # b.md appears as dst in one edge and src in the other; replacing edges
    # "for" b.md must clear both regardless of stored orientation.
    store.replace_edges_for("b.md", [_make_edge("b.md", "d.md", composite=0.9)])

    remaining = store.all_edges()
    pairs = {(e.src, e.dst) for e in remaining}
    assert pairs == {("b.md", "d.md")}


def test_edge_between_is_order_independent(tmp_path: Path) -> None:
    store = GraphStore(tmp_path / "fts.sqlite")
    store.initialize()
    store.replace_all_edges([_make_edge("a.md", "b.md", composite=0.7)])

    assert store.edge_between("a.md", "b.md").composite == pytest.approx(0.7)
    assert store.edge_between("b.md", "a.md").composite == pytest.approx(0.7)
    assert store.edge_between("a.md", "z.md") is None


def test_edges_for_finds_edges_regardless_of_stored_side(tmp_path: Path) -> None:
    store = GraphStore(tmp_path / "fts.sqlite")
    store.initialize()
    store.replace_all_edges(
        [_make_edge("a.md", "b.md"), _make_edge("b.md", "c.md")]
    )

    edges = store.edges_for("b.md")

    assert {(e.src, e.dst) for e in edges} == {("a.md", "b.md"), ("b.md", "c.md")}


def test_all_edges_filters_by_min_composite(tmp_path: Path) -> None:
    store = GraphStore(tmp_path / "fts.sqlite")
    store.initialize()
    store.replace_all_edges(
        [
            _make_edge("a.md", "b.md", composite=0.1),
            _make_edge("b.md", "c.md", composite=0.8),
        ]
    )

    assert len(store.all_edges(min_composite=0.0)) == 2
    strong = store.all_edges(min_composite=0.5)
    assert len(strong) == 1
    assert strong[0].composite == pytest.approx(0.8)


def test_semantic_edge_stats_ignores_zero_semantic_and_missing(tmp_path: Path) -> None:
    store = GraphStore(tmp_path / "fts.sqlite")
    store.initialize()
    assert store.semantic_edge_stats() == {}

    store.replace_all_edges(
        [
            _make_edge("a.md", "b.md", composite=0.5, semantic=0.6),
            _make_edge("a.md", "c.md", composite=0.2, semantic=0.0, link=0.7),
        ]
    )
    # a.md's only positive-semantic edge is the 0.6 one to b.md; its edge to
    # c.md has semantic=0 (link-only) and must not count or pull the min down.
    assert store.semantic_edge_stats()["a.md"] == (1, pytest.approx(0.6))
    assert "c.md" not in store.semantic_edge_stats()


def test_semantic_edge_stats_batches_count_and_min_across_all_notes(
    tmp_path: Path,
) -> None:
    store = GraphStore(tmp_path / "fts.sqlite")
    store.initialize()
    assert store.semantic_edge_stats() == {}

    store.replace_all_edges(
        [
            _make_edge("a.md", "b.md", composite=0.5, semantic=0.6),
            _make_edge("a.md", "c.md", composite=0.4, semantic=0.3),
            _make_edge("b.md", "c.md", composite=0.2, semantic=0.0, link=0.7),
        ]
    )

    stats = store.semantic_edge_stats()

    # a.md has semantic edges of 0.6 (to b) and 0.3 (to c) -> count 2, min 0.3.
    assert stats["a.md"] == (2, pytest.approx(0.3))
    # b.md has only the 0.6 semantic edge (its edge to c has semantic=0).
    assert stats["b.md"] == (1, pytest.approx(0.6))
    # c.md's only positive-semantic edge is the 0.3 one to a.md.
    assert stats["c.md"] == (1, pytest.approx(0.3))
    assert "d.md" not in stats


def test_replace_edges_for_paths_deletes_and_inserts_atomically(
    tmp_path: Path,
) -> None:
    store = GraphStore(tmp_path / "fts.sqlite")
    store.initialize()
    store.replace_all_edges(
        [_make_edge("a.md", "b.md"), _make_edge("b.md", "c.md")]
    )

    # Replacing edges "for" {b.md} must clear both its stored orientations
    # (as src in one row, dst in the other) and insert the new set, in one
    # call -- the atomic sibling of replace_edges_for's single-path version.
    store.replace_edges_for_paths(
        {"b.md"}, [_make_edge("b.md", "d.md", composite=0.9)]
    )

    remaining = store.all_edges()
    pairs = {(e.src, e.dst) for e in remaining}
    assert pairs == {("b.md", "d.md")}


def test_replace_edges_for_paths_no_op_on_empty_input(tmp_path: Path) -> None:
    store = GraphStore(tmp_path / "fts.sqlite")
    store.initialize()
    store.replace_all_edges([_make_edge("a.md", "b.md")])

    store.replace_edges_for_paths(set(), [])

    assert len(store.all_edges()) == 1


def test_replace_edges_for_paths_rolls_back_delete_on_insert_failure(
    tmp_path: Path,
) -> None:
    """A failure partway through the insert half must not leave the delete
    committed -- otherwise a note's edges are wiped with nothing to replace
    them (the exact bug this atomic method replaces two separate
    connections to fix)."""

    store = GraphStore(tmp_path / "fts.sqlite")
    store.initialize()
    store.replace_all_edges([_make_edge("a.md", "b.md")])

    valid_edge = _make_edge("a.md", "c.md")
    # NOT NULL violation on the second edge -- forces sqlite3 to raise
    # partway through the same connection/transaction as the prior DELETE.
    invalid_edge = _make_edge("a.md", "d.md")
    invalid_edge.src = None  # type: ignore[assignment]

    with pytest.raises(sqlite3.IntegrityError):
        store.replace_edges_for_paths({"a.md"}, [valid_edge, invalid_edge])

    # The pre-existing a-b edge must survive: both the delete and the
    # partial insert rolled back together.
    remaining = store.all_edges()
    assert {(e.src, e.dst) for e in remaining} == {("a.md", "b.md")}


def test_note_meta_for_many_batches_lookup(tmp_path: Path) -> None:
    store = GraphStore(tmp_path / "fts.sqlite")
    store.initialize()
    store.upsert_note_meta("a.md", "A", ["x"], [], [1.0])
    store.upsert_note_meta("b.md", "B", ["y"], [], [2.0])
    store.upsert_note_meta("c.md", "C", ["z"], [], [3.0])

    result = store.note_meta_for_many(["a.md", "c.md", "missing.md"])

    assert set(result) == {"a.md", "c.md"}
    assert result["a.md"]["tags"] == ["x"]
    assert result["c.md"]["title"] == "C"


def test_note_meta_for_many_empty_input_returns_empty_dict(tmp_path: Path) -> None:
    store = GraphStore(tmp_path / "fts.sqlite")
    store.initialize()

    assert store.note_meta_for_many([]) == {}


def test_counts_reflects_nodes_edges_and_last_built(tmp_path: Path) -> None:
    store = GraphStore(tmp_path / "fts.sqlite")
    store.initialize()
    store.upsert_note_meta("a.md", "A", [], [], [1.0])
    store.upsert_note_meta("b.md", "B", [], [], [1.0])
    store.replace_all_edges([_make_edge("a.md", "b.md")])

    counts = store.counts()

    assert counts["nodes"] == 2
    assert counts["edges"] == 1
    assert counts["last_built"] is not None
