from second_brain.models import RetrievalHit
from second_brain.retrieval import reciprocal_rank_fusion


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
