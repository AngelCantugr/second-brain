"""Golden-query retrieval evaluation.

Scores retrieval at the note level (recall@k and MRR) so that changes to parsing
or ranking can be measured against a fixed set of queries. Real golden sets name
personal notes, so they live beside the user's vault config, not in this repo.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from second_brain.service import MAX_TOP_K

_CASE_KEYS = frozenset({"query", "expected"})
_TOP_PATHS_IN_REPORT = 5


@dataclass(slots=True)
class GoldenCase:
    query: str
    expected: list[str]


@dataclass(slots=True)
class CaseResult:
    query: str
    expected: list[str]
    ranked_paths: list[str]
    first_relevant_rank: int | None
    recall: dict[int, float]
    # Expected paths that are not indexed notes: a typo or an excluded file,
    # which would otherwise look like a retrieval miss.
    unknown_expected: list[str]


@dataclass(slots=True)
class EvalReport:
    cases: list[CaseResult]
    recall: dict[int, float]
    mrr: float


def load_golden(path: str | Path) -> list[GoldenCase]:
    """Parse and validate a golden TOML file.

    Validation is strict on purpose: a typo'd key such as `expect` would otherwise
    silently produce a case with no expectations and skew every metric.
    """

    with Path(path).open("rb") as handle:
        document = tomllib.load(handle)

    raw_cases = document.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("golden file must define a non-empty 'cases' array")

    return [_parse_case(index, raw) for index, raw in enumerate(raw_cases)]


def _parse_case(index: int, raw: object) -> GoldenCase:
    label = f"cases[{index}]"
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a table with 'query' and 'expected'")

    unknown = set(raw) - _CASE_KEYS
    if unknown:
        raise ValueError(f"{label} has unknown keys: {sorted(unknown)}")

    query = raw.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"{label}.query must be a non-empty string")

    expected = raw.get("expected")
    if (
        not isinstance(expected, list)
        or not expected
        or not all(isinstance(path, str) for path in expected)
    ):
        raise ValueError(f"{label}.expected must be a non-empty list of strings")

    return GoldenCase(query=query, expected=list(expected))


def rank_note_paths(hits: list[dict], depth: int) -> list[str]:
    """Collapse chunk hits into distinct note paths, best-first.

    Many chunks belong to one note, so rank is decided by each note's best chunk.
    Verbose hits carry the path under metadata; compact hits carry it at top level.
    """

    ranked: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        path = _note_path(hit)
        if path in seen:
            continue
        seen.add(path)
        ranked.append(path)
        if len(ranked) == depth:
            break
    return ranked


def _note_path(hit: dict) -> str:
    metadata = hit.get("metadata")
    if isinstance(metadata, dict) and "path" in metadata:
        return metadata["path"]
    return hit["path"]


def evaluate(
    service,
    cases: list[GoldenCase],
    ks: tuple[int, ...] = (5, 10),
    depth: int = 10,
) -> EvalReport:
    """Run each golden query and compute note-level recall@k and MRR.

    The search asks for MAX_TOP_K chunks because duplicate chunks from one note
    consume result slots; a small top_k would under-count distinct notes.

    Each case also lists expected paths that are not indexed notes, so a
    mistyped or excluded path is flagged instead of silently scoring as a miss.
    """

    if not cases:
        raise ValueError("cases must not be empty")
    if not ks:
        raise ValueError("ks must not be empty")
    if depth < max(ks):
        raise ValueError(f"depth ({depth}) must be >= max(ks) ({max(ks)})")

    case_results = [_evaluate_case(service, case, ks, depth) for case in cases]
    mean_recall = {k: _mean(result.recall[k] for result in case_results) for k in ks}
    mrr = _mean(
        0.0 if result.first_relevant_rank is None else 1.0 / result.first_relevant_rank
        for result in case_results
    )
    return EvalReport(cases=case_results, recall=mean_recall, mrr=mrr)


def _evaluate_case(service, case: GoldenCase, ks: tuple[int, ...], depth: int) -> CaseResult:
    response = service.search(query=case.query, top_k=MAX_TOP_K, verbose=False)
    ranked_paths = rank_note_paths(response["hits"], depth)
    expected = set(case.expected)

    first_relevant_rank = next(
        (rank for rank, path in enumerate(ranked_paths, start=1) if path in expected),
        None,
    )
    recall = {
        k: len(expected.intersection(ranked_paths[:k])) / len(expected) for k in ks
    }
    return CaseResult(
        query=case.query,
        expected=list(case.expected),
        ranked_paths=ranked_paths,
        first_relevant_rank=first_relevant_rank,
        recall=recall,
        unknown_expected=[
            path for path in case.expected if service.graph_store.note_meta_for(path) is None
        ],
    )


def report_to_dict(report: EvalReport) -> dict:
    """Render a report as JSON-safe data with floats rounded for stable diffs."""

    summary = {"cases": len(report.cases)}
    summary.update({f"recall@{k}": round(value, 4) for k, value in report.recall.items()})
    summary["mrr"] = round(report.mrr, 4)
    summary["unknown_expected"] = sum(len(result.unknown_expected) for result in report.cases)

    cases = []
    for result in report.cases:
        case_dict = {
            "query": result.query,
            "expected": result.expected,
            "first_relevant_rank": result.first_relevant_rank,
        }
        case_dict.update({f"recall@{k}": round(value, 4) for k, value in result.recall.items()})
        case_dict["unknown_expected"] = result.unknown_expected
        case_dict["top_paths"] = result.ranked_paths[:_TOP_PATHS_IN_REPORT]
        cases.append(case_dict)

    return {"summary": summary, "cases": cases}


def _mean(values) -> float:
    collected = list(values)
    return sum(collected) / len(collected)
