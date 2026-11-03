import math

import pytest

from second_brain.models import RetrievalHit
from second_brain.retrieval import (
    apply_graph_boost,
    graph_boost_seeds,
    reciprocal_rank_fusion,
    validate_graph_boost,
)


def test_rrf_prioritizes_items_present_in_both_rankings() -> None:
    semantic = [
        RetrievalHit(chunk_id="c1", score=0.9, source="semantic", text=""),
        RetrievalHit(chunk_id="c2", score=0.8, source="semantic", text=""),
    ]
    keyword = [
        RetrievalHit(chunk_id="c2", score=9.0, source="keyword", text=""),
        RetrievalHit(chunk_id="c3", score=8.0, source="keyword", text=""),
    ]

    merged = reciprocal_rank_fusion(semantic, keyword, k=60)

    assert merged[0].chunk_id == "c2"
    assert {hit.chunk_id for hit in merged[:3]} == {"c1", "c2", "c3"}


def test_recency_boost_is_bounded_and_has_a_thirty_day_half_life(monkeypatch) -> None:
    import second_brain.retrieval as retrieval

    now = 1_800_000_000.0
    monkeypatch.setattr(retrieval.time, "time", lambda: now)
    future = RetrievalHit("future", 1.0, "semantic", "", {"mtime": now + 86400})
    recent = RetrievalHit("recent", 0.9, "semantic", "", {"mtime": now})
    month_old = RetrievalHit("month-old", 0.8, "semantic", "", {"mtime": now - 30 * 86400})

    merged = reciprocal_rank_fusion([future, recent, month_old], [], recency_boost=1.0)
    scores = {hit.chunk_id: hit.score for hit in merged}

    assert scores["future"] == 1 / 61 + 1 / 61
    assert scores["recent"] == 1 / 62 + 1 / 61
    assert scores["month-old"] == 1 / 63 + 1 / 122
    assert max(scores.values()) <= 2 / 61


def test_invalid_or_missing_mtime_metadata_adds_no_recency_contribution(monkeypatch) -> None:
    import math
    import second_brain.retrieval as retrieval

    now = 1_800_000_000.0
    monkeypatch.setattr(retrieval.time, "time", lambda: now)
    metadata_values = [
        {},
        {"mtime": None},
        {"mtime": True},
        {"mtime": math.nan},
        {"mtime": math.inf},
        {"mtime": "invalid"},
        {"mtime": 10**400},
    ]
    hits = [
        RetrievalHit(f"invalid-{index}", 1.0, "semantic", "", metadata)
        for index, metadata in enumerate(metadata_values)
    ]

    merged = reciprocal_rank_fusion(hits, [], recency_boost=1.0)

    assert {hit.chunk_id: hit.score for hit in merged} == {
        f"invalid-{index}": 1 / (61 + index)
        for index in range(len(metadata_values))
    }


@pytest.mark.parametrize("value", [math.nan, math.inf, -0.1, 1.1, True, "0.5", 10**400])
def test_recency_boost_validator_has_shared_range_and_type_contract(value) -> None:
    from second_brain.retrieval import validate_recency_boost

    with pytest.raises(ValueError, match="recency_boost"):
        validate_recency_boost(value)


def _hit(chunk_id: str, path: str, score: float) -> RetrievalHit:
    return RetrievalHit(chunk_id=chunk_id, score=score, source="hybrid", text="", metadata={"path": path})


def test_graph_boost_seeds_are_distinct_note_paths_in_rank_order() -> None:
    hits = [_hit("1", "a.md", 3), _hit("2", "a.md", 2), RetrievalHit("3", 1, "hybrid", ""), _hit("4", "b.md", 1)]

    assert graph_boost_seeds(hits, 5) == ["a.md", "b.md"]
    assert graph_boost_seeds(hits, 1) == ["a.md"]


def test_graph_boost_adds_at_most_one_rank_one_term_scaled_by_composite() -> None:
    top = _hit("top", "a.md", 1 / 61)
    other = _hit("other", "x.md", 1 / 62)
    neighbor = _hit("neighbor", "b.md", 1 / 63)
    weak = _hit("weak", "c.md", 1 / 64)

    boosted = apply_graph_boost(
        [top, other, neighbor, weak], {"b.md": 0.5, "c.md": 0.01}, graph_boost=1.0
    )

    assert [hit.chunk_id for hit in boosted] == ["neighbor", "top", "other", "weak"]
    assert boosted[0].score == pytest.approx(1 / 63 + 0.5 / 61)
    assert boosted[3].score == pytest.approx(1 / 64 + 0.01 / 61)


def test_full_graph_boost_cannot_pass_a_hit_both_legs_ranked_first() -> None:
    both_legs = _hit("both", "a.md", 2 / 61)
    neighbor = _hit("neighbor", "b.md", 1 / 61)

    boosted = apply_graph_boost([both_legs, neighbor], {"b.md": 1.0}, graph_boost=1.0)

    assert [hit.chunk_id for hit in boosted] == ["both", "neighbor"]


def test_zero_graph_boost_or_no_neighbors_keeps_fused_order() -> None:
    hits = [_hit("1", "a.md", 0.02), _hit("2", "b.md", 0.01)]

    assert apply_graph_boost(hits, {"b.md": 1.0}, graph_boost=0.0) == hits
    assert apply_graph_boost(hits, {}, graph_boost=1.0) == hits


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.01, 1.01, True, "1", None, 10**400])
def test_graph_boost_validator_rejects_out_of_range_values(value) -> None:
    with pytest.raises(ValueError, match="graph_boost"):
        validate_graph_boost(value)
