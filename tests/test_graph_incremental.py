"""Incremental graph updates must leave the edges a full rebuild would produce."""

import random
from pathlib import Path

import pytest

from second_brain.config import RagConfig
from second_brain.graph import GraphBuilder, GraphStore, vault_metas


def _graph(tmp_path: Path, **overrides) -> tuple[GraphStore, GraphBuilder]:
    config = RagConfig(
        vault_path=tmp_path,
        qdrant_path=tmp_path / "q",
        fts_path=tmp_path / "f.sqlite",
        sync_state_path=tmp_path / "s.sqlite",
        **overrides,
    )
    store = GraphStore(config.fts_path)
    store.initialize()
    return store, GraphBuilder(store, config)


def _pairs(store: GraphStore) -> list[tuple[str, str]]:
    return sorted((edge.src, edge.dst) for edge in store.all_edges())


def test_editing_a_note_keeps_edges_between_its_neighbour_and_other_notes(
    tmp_path: Path,
) -> None:
    """W-Z exists because Z is W's nearest note, not the reverse.

    Editing X makes Z affected (Z is X's neighbour). Rebuilding Z's edges
    from Z's own candidates alone would drop W-Z, since W is not among them.
    """

    store, builder = _graph(tmp_path, graph_knn_k=1)
    store.upsert_note_meta("X.md", "X", [], [], [1.0, 0.0])
    store.upsert_note_meta("Z.md", "Z", [], [], [0.9, 0.44])
    store.upsert_note_meta("W.md", "W", [], [], [0.5, 0.87])
    builder.rebuild_full()
    assert _pairs(store) == [("W.md", "Z.md"), ("X.md", "Z.md")]

    old = store.upsert_note_meta("X.md", "X", ["newtag"], [], [1.0, 0.0])
    builder.update_for_changes({"X.md"}, set(), {"X.md": old})

    assert _pairs(store) == [("W.md", "Z.md"), ("X.md", "Z.md")]


def test_deleting_a_note_lets_its_neighbour_take_the_next_nearest_note(
    tmp_path: Path,
) -> None:
    """N's only neighbour was X; with X gone, M is N's nearest note."""

    store, builder = _graph(tmp_path, graph_knn_k=1)
    for path, centroid in (
        ("X.md", [1.0, 0.0]),
        ("N.md", [0.8, 0.6]),
        ("M.md", [0.0, 1.0]),
        ("P.md", [-0.3, 0.95]),
    ):
        store.upsert_note_meta(path, path[0], [], [], centroid)
    builder.rebuild_full()
    assert _pairs(store) == [("M.md", "P.md"), ("N.md", "X.md")]

    old = store.delete_note_meta("X.md")
    builder.update_for_changes(set(), {"X.md"}, {"X.md": old})

    assert _pairs(store) == [("M.md", "N.md"), ("M.md", "P.md")]


def test_moving_a_note_away_frees_its_old_neighbours_slot(tmp_path: Path) -> None:
    """Same as a delete, but X stays in the vault with a centroid far from N."""

    store, builder = _graph(tmp_path, graph_knn_k=1)
    for path, centroid in (
        ("X.md", [1.0, 0.0]),
        ("N.md", [0.8, 0.6]),
        ("M.md", [0.0, 1.0]),
        ("P.md", [-0.3, 0.95]),
    ):
        store.upsert_note_meta(path, path[0], [], [], centroid)
    builder.rebuild_full()

    old = store.upsert_note_meta("X.md", "X", [], [], [0.0, -1.0])
    builder.update_for_changes({"X.md"}, set(), {"X.md": old})

    assert _pairs(store) == [("M.md", "N.md"), ("M.md", "P.md")]


def test_a_freed_slot_is_refilled_when_the_old_pair_never_had_a_stored_edge(
    tmp_path: Path,
) -> None:
    """N listed X, but N-X scored below ``graph_min_edge_score`` and was not stored.

    With a low semantic weight a nearest-note pair can be filtered out, so
    stored edges cannot say who was listing X. When X moves away, N takes M,
    and their shared tag lifts N-M over the threshold.
    """

    import math

    def unit(degrees: float) -> list[float]:
        return [math.cos(math.radians(degrees)), math.sin(math.radians(degrees))]

    store, builder = _graph(
        tmp_path, graph_knn_k=1, graph_weight_semantic=0.1, graph_weight_tag=0.15
    )
    store.upsert_note_meta("N.md", "N", ["shared"], [], unit(0))
    store.upsert_note_meta("X.md", "X", [], [], unit(10))
    store.upsert_note_meta("M.md", "M", ["shared"], [], unit(40))
    store.upsert_note_meta("P.md", "P", [], [], unit(50))
    builder.rebuild_full()
    assert _pairs(store) == []

    old = store.upsert_note_meta("X.md", "X", [], [], unit(-60))
    builder.update_for_changes({"X.md"}, set(), {"X.md": old})

    assert _pairs(store) == [("M.md", "N.md")]
    assert sorted(_reference(store, builder)) == [("M.md", "N.md")]


def _reference(store: GraphStore, builder: GraphBuilder) -> dict[tuple[str, str], float]:
    """Every edge scored from scratch, in the frame incremental updates use."""

    metas = sorted(vault_metas(store.all_note_meta()), key=lambda meta: meta["path"])
    ctx = builder._build_context(metas, store.vault_mean())
    return {(edge.src, edge.dst): edge.composite for edge in builder._score_all(ctx)}


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("center", [False, True])
def test_incremental_updates_match_a_rescore_of_the_whole_vault(
    tmp_path: Path, seed: int, center: bool
) -> None:
    rng = random.Random(seed)
    # Odd seeds use a low semantic weight, so a nearest-note pair can score
    # below graph_min_edge_score and leave no stored edge behind.
    weights = {"graph_weight_semantic": 0.1, "graph_weight_tag": 0.4} if seed % 2 else {}
    store, builder = _graph(tmp_path, graph_knn_k=3, graph_semantic_center=center, **weights)
    tags = ["a", "b", "c", "d"]
    names = [f"n{i:02d}" for i in range(18)]
    alive: set[str] = set()

    def write(name: str) -> dict | None:
        links = rng.sample(names, rng.randint(0, 3))
        return store.upsert_note_meta(
            f"{name}.md",
            name,
            rng.sample(tags, rng.randint(0, 2)),
            links,
            [rng.gauss(0.0, 1.0) for _ in range(4)],
        )

    for name in names[:14]:
        write(name)
        alive.add(name)
    builder.rebuild_full()

    for step in range(12):
        changed: set[str] = set()
        deleted: set[str] = set()
        old_meta: dict[str, dict | None] = {}
        for _ in range(rng.randint(1, 2)):
            name = rng.choice(names)
            path = f"{name}.md"
            if path in old_meta:
                continue
            if name in alive and rng.random() < 0.3:
                old_meta[path] = store.delete_note_meta(path)
                deleted.add(path)
                alive.discard(name)
            else:
                old_meta[path] = write(name)
                changed.add(path)
                alive.add(name)

        builder.update_for_changes(changed, deleted, old_meta)

        stored = {(edge.src, edge.dst): edge.composite for edge in store.all_edges()}
        expected = _reference(store, builder)
        assert sorted(stored) == sorted(expected), f"seed {seed} step {step}"
        assert stored == pytest.approx(expected), f"seed {seed} step {step}"
