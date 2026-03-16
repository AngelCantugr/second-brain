"""Query normalization and hybrid rank fusion algorithms."""

from __future__ import annotations

from collections import defaultdict

from obsidian_rag.models import RetrievalHit


def normalize_query(query: str) -> str:
    """Normalize raw user query for consistent retrieval."""

    return " ".join(query.strip().lower().split())


def reciprocal_rank_fusion(
    semantic_hits: list[RetrievalHit], keyword_hits: list[RetrievalHit], k: int = 60
) -> list[RetrievalHit]:
    """Combine semantic + keyword rankings using reciprocal rank fusion."""

    score_map: dict[str, float] = defaultdict(float)
    exemplar: dict[str, RetrievalHit] = {}

    for idx, hit in enumerate(semantic_hits, start=1):
        score_map[hit.chunk_id] += 1.0 / (k + idx)
        exemplar.setdefault(hit.chunk_id, hit)

    for idx, hit in enumerate(keyword_hits, start=1):
        score_map[hit.chunk_id] += 1.0 / (k + idx)
        exemplar.setdefault(hit.chunk_id, hit)

    merged = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
    output: list[RetrievalHit] = []
    for chunk_id, score in merged:
        base = exemplar[chunk_id]
        output.append(
            RetrievalHit(
                chunk_id=chunk_id,
                score=score,
                source="hybrid",
                text=base.text,
                metadata=base.metadata,
            )
        )

    return output
