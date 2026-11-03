import sqlite3
from pathlib import Path

import numpy as np
import pytest

from second_brain.graph import (
    GraphStore,
    LinkResolver,
    canonical_pair,
    comention_component,
    comention_pairs,
    composite_score,
    cosine,
    cosine_matrix,
    ghost_targets,
    link_component,
    normalize_link_target,
    tag_jaccard,
)
from second_brain.models import Edge


def test_normalize_link_target_strips_fragment_case_and_whitespace() -> None:
    assert normalize_link_target("Project X#Overview") == "project x"
    assert normalize_link_target("  Note B  ") == "note b"
    assert normalize_link_target("note") == "note"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Note#^block-id", "note"),
        ("Note^block-id", "note"),
        ("Folder/Sub/Note.md#Heading", "folder/sub/note"),
        ("./Folder/Note", "folder/note"),
        ("/Folder/Note", "folder/note"),
        ("Folder\\Note", "folder/note"),
        ("#Heading in this note", ""),
        ("^block", ""),
    ],
)
def test_normalize_link_target_handles_obsidian_link_forms(raw: str, expected: str) -> None:
    assert normalize_link_target(raw) == expected


def _meta(path: str, aliases: list[str] | None = None, links: list[str] | None = None) -> dict:
    stem = path.rsplit("/", 1)[-1].removesuffix(".md")
    return {
        "path": path,
        "title": stem,
        "title_key": stem.lower(),
        "tags": [],
        "links": links or [],
        "aliases": aliases or [],
        "centroid": None,
        "dim": None,
    }


def test_resolver_matches_path_qualified_links_by_path_suffix() -> None:
    resolver = LinkResolver([_meta("Projects/Alpha/Plan.md"), _meta("Areas/Plan.md")])

    assert resolver.resolve("Alpha/Plan", "x.md") == "Projects/Alpha/Plan.md"
    assert resolver.resolve("projects/alpha/plan.md", "x.md") == "Projects/Alpha/Plan.md"
    assert resolver.resolve("Areas/Plan#Goals", "x.md") == "Areas/Plan.md"
    # A partial folder name is not a path segment match.
    assert resolver.resolve("pha/Plan", "x.md") is None
    assert resolver.resolve("Missing/Plan", "x.md") is None


def test_resolver_joins_relative_links_to_the_linking_notes_folder() -> None:
    resolver = LinkResolver([_meta("Area/Topic/Note.md"), _meta("Other/Topic/Note.md")])

    assert resolver.resolve("../Topic/Note", "Area/Daily/Today.md") == "Area/Topic/Note.md"
    assert resolver.resolve("../../Other/Topic/Note#x", "Area/Daily/Today.md") == "Other/Topic/Note.md"
    # Climbing above the vault root never matches.
    assert resolver.resolve("../../Topic/Note", "Area/Today.md") is None
    assert resolver.unresolved(["../../Topic/Note"], "Area/Today.md") == {"../../topic/note"}


def test_resolver_requires_an_exact_path_for_relative_links() -> None:
    resolver = LinkResolver([_meta("Other/Area/Topic/Note.md"), _meta("Root.md")])

    # Area/Topic/Note does not exist; a suffix match must not pick Other/...
    assert resolver.resolve("../Topic/Note", "Area/Daily/Today.md") is None
    assert resolver.unresolved(["../Topic/Note"], "Area/Daily/Today.md") == {"area/topic/note"}
    assert resolver.resolve("../Root", "Area/Today.md") == "Root.md"


def test_resolver_treats_a_leading_slash_as_an_exact_vault_path() -> None:
    resolver = LinkResolver([_meta("Alpha.md"), _meta("Projects/Alpha.md"), _meta("x/Beta.md")])

    # Without the slash, proximity picks the note in the linker's folder.
    assert resolver.resolve("Alpha", "Projects/Board.canvas") == "Projects/Alpha.md"
    assert resolver.resolve("/Alpha.md", "Projects/Board.canvas") == "Alpha.md"
    assert resolver.resolve("/Projects/Alpha", "Other.md") == "Projects/Alpha.md"
    # No suffix match: /Beta.md is not x/Beta.md.
    assert resolver.resolve("/Beta.md", "Other.md") is None


def test_resolver_picks_one_note_for_duplicate_names_like_obsidian() -> None:
    resolver = LinkResolver(
        [_meta("deep/nested/Dup.md"), _meta("b/Dup.md"), _meta("a/Dup.md"), _meta("src/Dup2.md")]
    )

    # Same folder as the linking note wins.
    assert resolver.resolve("Dup", "deep/nested/Linker.md") == "deep/nested/Dup.md"
    # Otherwise the shallowest path, then alphabetical order.
    assert resolver.resolve("Dup", "src/Linker.md") == "a/Dup.md"


def test_resolver_falls_back_to_aliases_only_when_no_file_name_matches() -> None:
    resolver = LinkResolver(
        [
            _meta("Kubernetes.md", aliases=["k8s", "Orchestration"]),
            _meta("Orchestration.md"),
        ]
    )

    assert resolver.resolve("K8s", "x.md") == "Kubernetes.md"
    assert resolver.resolve("k8s#Pods", "x.md") == "Kubernetes.md"
    assert resolver.resolve("Orchestration", "x.md") == "Orchestration.md"


def test_resolver_ignores_self_links_and_section_links() -> None:
    resolver = LinkResolver([_meta("A.md"), _meta("other/A.md")])

    assert resolver.resolve("A", "A.md") is None
    assert resolver.resolve("#Heading", "A.md") is None
    assert resolver.resolve_all(["A", "other/A", "#x"], "A.md") == ["other/A.md"]


def test_resolver_unresolved_skips_attachments_self_and_section_links() -> None:
    resolver = LinkResolver([_meta("A.md", aliases=["Alias A"])])

    missing = resolver.unresolved(
        ["Ghost Note", "ghost note#intro", "Folder/Missing", "A", "alias a", "#top", "diagram.png", "Spec.pdf"],
        "A.md",
    )

    assert missing == {"ghost note", "folder/missing"}


def test_ghost_targets_rank_by_distinct_inbound_notes() -> None:
    metas = [
        _meta("a.md", links=["Idea", "Idea#again", "Rare"]),
        _meta("b.md", links=["idea"]),
        _meta("c.md", links=["Idea", "a"]),
    ]

    assert ghost_targets(metas) == [
        {"target": "idea", "inbound": 3, "linked_from": ["a.md", "b.md", "c.md"]},
        {"target": "rare", "inbound": 1, "linked_from": ["a.md"]},
    ]
    assert [g["target"] for g in ghost_targets(metas, limit=1)] == ["idea"]


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
    assert meta["aliases"] == []


def test_note_meta_round_trips_aliases(tmp_path: Path) -> None:
    store = GraphStore(tmp_path / "fts.sqlite")
    store.initialize()

    store.upsert_note_meta("a.md", "A", [], [], None, ["Zed", "Alpha"])

    assert store.note_meta_for("a.md")["aliases"] == ["Alpha", "Zed"]


def test_initialize_adds_aliases_column_to_an_existing_database(tmp_path: Path) -> None:
    db = tmp_path / "fts.sqlite"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE note_meta (path TEXT PRIMARY KEY, title TEXT NOT NULL, "
        "title_key TEXT NOT NULL, tags_json TEXT NOT NULL, links_json TEXT NOT NULL, "
        "centroid BLOB, dim INTEGER, updated_at REAL)"
    )
    conn.execute(
        "INSERT INTO note_meta(path, title, title_key, tags_json, links_json) "
        "VALUES('old.md', 'Old', 'old', '[]', '[]')"
    )
    conn.commit()
    conn.close()

    store = GraphStore(db)
    store.initialize()
    store.initialize()

    assert store.note_meta_for("old.md")["aliases"] == []
    store.upsert_note_meta("new.md", "New", [], [], None, ["N"])
    assert store.note_meta_for("new.md")["aliases"] == ["N"]


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
