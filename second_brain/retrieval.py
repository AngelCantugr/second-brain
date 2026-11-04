"""Query normalization and hybrid rank fusion algorithms."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import math
import time

from second_brain.models import RetrievalHit


def normalize_query(query: str) -> str:
    """Normalize raw user query for consistent retrieval."""

    return " ".join(query.strip().lower().split())


def _validate_unit_boost(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number in [0.0, 1.0]")
    try:
        valid = math.isfinite(value) and 0.0 <= value <= 1.0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a finite number in [0.0, 1.0]")
    return float(value)


def validate_recency_boost(value: object) -> float:
    """Validate and normalize the optional recency contribution in [0, 1]."""

    return _validate_unit_boost(value, "recency_boost")


def validate_graph_boost(value: object) -> float:
    """Validate and normalize the optional graph-neighbor contribution in [0, 1]."""

    return _validate_unit_boost(value, "graph_boost")


def validate_code_source_weight(value: object) -> float:
    """Validate the code-source score multiplier, a finite number in (0, 1]."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("code_source_weight must be a finite number in (0.0, 1.0]")
    try:
        valid = math.isfinite(value) and 0.0 < value <= 1.0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("code_source_weight must be a finite number in (0.0, 1.0]")
    return float(value)


def apply_code_source_weight(
    hits: list[RetrievalHit], weight: float
) -> list[RetrievalHit]:
    """Scale the fused scores of code-source hits by ``weight`` and re-sort.

    Code sources can hold as many chunks as the vault, so for a question about
    the vault they crowd notes out of the top ranks with chunks that merely
    share its words. Scaling lets a vault note pass a repo chunk it nearly
    ties, while a repo chunk that clearly wins both retrieval legs still ranks
    first.
    """

    if weight == 1.0:
        return hits
    weighted = [
        replace(hit, score=hit.score * weight) if hit.metadata.get("repo") else hit
        for hit in hits
    ]
    # sorted() is stable, so equal scores keep their fused order.
    return sorted(weighted, key=lambda hit: hit.score, reverse=True)


def graph_boost_seeds(hits: list[RetrievalHit], count: int) -> list[str]:
    """Return the note paths of the first ``count`` distinct notes in ``hits``."""

    seeds: list[str] = []
    for hit in hits:
        path = hit.metadata.get("path")
        if path and path not in seeds:
            seeds.append(path)
            if len(seeds) == count:
                break
    return seeds


def apply_graph_boost(
    hits: list[RetrievalHit],
    neighbor_strength: dict[str, float],
    k: int = 60,
    graph_boost: float = 0.0,
) -> list[RetrievalHit]:
    """Rerank fused hits, lifting chunks from graph neighbors of the top notes.

    ``neighbor_strength`` maps a note path to its strongest edge composite
    (0 to 1) with any seed note (one of the top fused notes). Like ``recency_boost``,
    a full boost adds at most one rank-1 RRF term (``1 / (k + 1)``), scaled
    by that composite, so related notes can pass close competitors, while a
    chunk found by only one retrieval leg cannot pass one both legs ranked
    first. Only retrieved candidates are reordered; no new notes are added.
    """

    graph_boost = validate_graph_boost(graph_boost)
    if not graph_boost or not neighbor_strength:
        return hits

    max_contribution = graph_boost / (k + 1)
    boosted = [
        replace(
            hit,
            score=hit.score
            + max_contribution * neighbor_strength.get(hit.metadata.get("path"), 0.0),
        )
        for hit in hits
    ]
    # sorted() is stable, so unboosted ties keep their fused order.
    return sorted(boosted, key=lambda hit: hit.score, reverse=True)


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
