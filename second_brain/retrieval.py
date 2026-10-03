"""Query normalization and hybrid rank fusion algorithms."""

from __future__ import annotations

from collections import defaultdict
import math
import time

from second_brain.models import RetrievalHit


def normalize_query(query: str) -> str:
    """Normalize raw user query for consistent retrieval."""

    return " ".join(query.strip().lower().split())


def validate_recency_boost(value: object) -> float:
    """Validate and normalize the optional recency contribution in [0, 1]."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("recency_boost must be a finite number in [0.0, 1.0]")
    try:
        valid = math.isfinite(value) and 0.0 <= value <= 1.0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("recency_boost must be a finite number in [0.0, 1.0]")
    return float(value)


def reciprocal_rank_fusion(
    semantic_hits: list[RetrievalHit],
    keyword_hits: list[RetrievalHit],
    k: int = 60,
    recency_boost: float = 0.0,
) -> list[RetrievalHit]:
    """Fuse rankings and optionally add a bounded, exponentially decaying age signal.

    A full boost contributes at most one rank-1 RRF term (``1 / (k + 1)``),
    with a 30-day half-life. It can reorder close candidates without replacing
    semantic and keyword relevance signals. The default leaves RRF intact.
    """

    recency_boost = validate_recency_boost(recency_boost)

    score_map: dict[str, float] = defaultdict(float)
    exemplar: dict[str, RetrievalHit] = {}
    semantic_scores: dict[str, float] = {}
    keyword_scores: dict[str, float] = {}

    for idx, hit in enumerate(semantic_hits, start=1):
        score_map[hit.chunk_id] += 1.0 / (k + idx)
        exemplar.setdefault(hit.chunk_id, hit)
        semantic_scores[hit.chunk_id] = (
            hit.semantic_score if hit.semantic_score is not None else hit.score
        )

    for idx, hit in enumerate(keyword_hits, start=1):
        score_map[hit.chunk_id] += 1.0 / (k + idx)
        exemplar.setdefault(hit.chunk_id, hit)
        keyword_scores[hit.chunk_id] = hit.score

    if recency_boost:
        now = time.time()
        half_life_seconds = 30 * 24 * 60 * 60
        max_contribution = recency_boost / (k + 1)
        for chunk_id, hit in exemplar.items():
            mtime = hit.metadata.get("mtime")
            if isinstance(mtime, bool) or not isinstance(mtime, (int, float)):
                continue
            try:
                if not math.isfinite(mtime):
                    continue
            except OverflowError:
                continue
            age_seconds = max(0.0, now - mtime)
            score_map[chunk_id] += max_contribution * math.exp(
                -math.log(2.0) * age_seconds / half_life_seconds
            )

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
                semantic_score=semantic_scores.get(chunk_id),
                keyword_score=keyword_scores.get(chunk_id),
            )
        )

    return output
