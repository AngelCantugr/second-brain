from obsidian_rag.models import RetrievalHit
from obsidian_rag.retrieval import reciprocal_rank_fusion


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
