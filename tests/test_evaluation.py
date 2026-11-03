from pathlib import Path

import pytest

from second_brain.evaluation import (
    CaseResult,
    EvalReport,
    GoldenCase,
    evaluate,
    load_golden,
    rank_note_paths,
    report_to_dict,
)
from second_brain.service import MAX_TOP_K


class _StubGraphStore:
    """Answers note_meta_for like GraphStore; `indexed=None` means every path exists."""

    def __init__(self, indexed: set[str] | None) -> None:
        self._indexed = indexed

    def note_meta_for(self, path: str) -> dict | None:
        if self._indexed is None or path in self._indexed:
            return {"path": path}
        return None


class _StubService:
    """Returns canned compact hits per query and records the search arguments."""

    def __init__(
        self,
        hits_by_query: dict[str, list[dict]],
        indexed: set[str] | None = None,
    ) -> None:
        self._hits_by_query = hits_by_query
        self.graph_store = _StubGraphStore(indexed)
        self.calls: list[dict] = []

    def search(
        self, query: str, top_k: int, verbose: bool, graph_boost: float = 0.0, **_: object
    ) -> dict:
        self.calls.append(
            {"query": query, "top_k": top_k, "verbose": verbose, "graph_boost": graph_boost}
        )
        return {"query": query, "hits": self._hits_by_query[query]}


def _chunk(path: str) -> dict:
    return {"path": path, "score": 0.5}


def _write_golden(tmp_path: Path, body: str) -> Path:
    golden = tmp_path / "golden.toml"
    golden.write_text(body, encoding="utf-8")
    return golden


def test_load_golden_round_trips_valid_file(tmp_path: Path) -> None:
    golden = _write_golden(
        tmp_path,
        """
[[cases]]
query = "how do I handle retries with exponential backoff"
expected = ["Engineering/Resilience.md"]

[[cases]]
query = "kafka consumer groups"
expected = ["Learning/Kafka.md", "Learning/Streaming.md"]
""",
    )

    cases = load_golden(golden)

    assert cases == [
        GoldenCase(
            query="how do I handle retries with exponential backoff",
            expected=["Engineering/Resilience.md"],
        ),
        GoldenCase(
            query="kafka consumer groups",
            expected=["Learning/Kafka.md", "Learning/Streaming.md"],
        ),
    ]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        pytest.param('title = "no cases"\n', "cases", id="missing-cases"),
        pytest.param("cases = []\n", "cases", id="empty-cases"),
        pytest.param('cases = "not-an-array"\n', "cases", id="cases-not-array"),
        pytest.param(
            '[[cases]]\nexpected = ["a.md"]\n', "query", id="missing-query"
        ),
        pytest.param(
            '[[cases]]\nquery = "   "\nexpected = ["a.md"]\n',
            "query",
            id="blank-query",
        ),
        pytest.param(
            '[[cases]]\nquery = 5\nexpected = ["a.md"]\n', "query", id="non-string-query"
        ),
        pytest.param(
            '[[cases]]\nquery = "q"\n', "expected", id="missing-expected"
        ),
        pytest.param(
            '[[cases]]\nquery = "q"\nexpected = "a.md"\n',
            "expected",
            id="expected-not-list",
        ),
        pytest.param(
            '[[cases]]\nquery = "q"\nexpected = []\n', "expected", id="empty-expected"
        ),
        pytest.param(
            '[[cases]]\nquery = "q"\nexpected = [1]\n',
            "expected",
            id="expected-non-string",
        ),
        pytest.param(
            '[[cases]]\nquery = "q"\nexpected = ["a.md"]\nweight = 2\n',
            "unknown",
            id="unknown-case-key",
        ),
    ],
)
def test_load_golden_rejects_invalid_shapes(tmp_path: Path, body: str, message: str) -> None:
    golden = _write_golden(tmp_path, body)

    with pytest.raises(ValueError, match=message):
        load_golden(golden)


def test_rank_note_paths_reads_verbose_and_compact_hits_in_first_occurrence_order() -> None:
    hits = [
        {"metadata": {"path": "A.md"}, "text": "a1"},
        {"path": "B.md", "chunk_id": "b1"},
        {"metadata": {"path": "A.md"}, "text": "a2"},
        {"path": "C.md", "chunk_id": "c1"},
        {"path": "B.md", "chunk_id": "b2"},
    ]

    assert rank_note_paths(hits, depth=10) == ["A.md", "B.md", "C.md"]


def test_rank_note_paths_stops_at_depth_distinct_notes() -> None:
    hits = [{"path": "A.md"}, {"path": "A.md"}, {"path": "B.md"}, {"path": "C.md"}]

    assert rank_note_paths(hits, depth=2) == ["A.md", "B.md"]


def test_evaluate_scores_first_hit_missing_and_rank_three_cases() -> None:
    service = _StubService(
        {
            "first": [_chunk("Expected.md"), _chunk("Other.md")],
            "third": [
                _chunk("Dup.md"),
                _chunk("Dup.md"),
                _chunk("Other.md"),
                _chunk("Other.md"),
                _chunk("Target.md"),
            ],
            "absent": [_chunk("X.md"), _chunk("Y.md")],
        }
    )
    cases = [
        GoldenCase(query="first", expected=["Expected.md"]),
        GoldenCase(query="third", expected=["Target.md"]),
        GoldenCase(query="absent", expected=["Missing.md"]),
    ]

    report = evaluate(service, cases)

    assert [case.first_relevant_rank for case in report.cases] == [1, 3, None]
    assert report.recall[5] == pytest.approx(2 / 3)
    assert report.recall[10] == pytest.approx(2 / 3)
    assert report.mrr == pytest.approx((1 + 1 / 3 + 0) / 3)


def test_evaluate_counts_partial_recall_by_k() -> None:
    service = _StubService(
        {
            "q": [
                _chunk("A.md"),
                _chunk("C.md"),
                _chunk("D.md"),
                _chunk("E.md"),
                _chunk("F.md"),
                _chunk("G.md"),
                _chunk("B.md"),
            ]
        }
    )
    report = evaluate(service, [GoldenCase(query="q", expected=["A.md", "B.md"])])

    assert report.cases[0].recall == {5: 0.5, 10: 1.0}
    assert report.recall == {5: 0.5, 10: 1.0}
    assert report.mrr == pytest.approx(1.0)


def test_evaluate_searches_with_max_top_k_and_compact_hits() -> None:
    service = _StubService({"q": [_chunk("A.md")]})

    evaluate(service, [GoldenCase(query="q", expected=["A.md"])])

    assert service.calls == [
        {"query": "q", "top_k": MAX_TOP_K, "verbose": False, "graph_boost": 0.0}
    ]


def test_evaluate_passes_graph_boost_to_search() -> None:
    service = _StubService({"q": [_chunk("A.md")]})

    evaluate(service, [GoldenCase(query="q", expected=["A.md"])], graph_boost=0.5)

    assert service.calls[0]["graph_boost"] == 0.5


def test_evaluate_rejects_depth_below_largest_k() -> None:
    service = _StubService({"q": [_chunk("A.md")]})

    with pytest.raises(ValueError, match="depth"):
        evaluate(service, [GoldenCase(query="q", expected=["A.md"])], depth=9)


def test_report_to_dict_has_summary_and_rounded_case_fields() -> None:
    case = CaseResult(
        query="q",
        expected=["A.md"],
        ranked_paths=["A.md", "B.md", "C.md", "D.md", "E.md", "F.md"],
        first_relevant_rank=1,
        recall={5: 1.0, 10: 1.0},
        unknown_expected=[],
    )
    report = EvalReport(cases=[case], recall={5: 1.0, 10: 1.0}, mrr=1 / 3)

    result = report_to_dict(report)

    assert result == {
        "summary": {
            "cases": 1,
            "recall@5": 1.0,
            "recall@10": 1.0,
            "mrr": 0.3333,
            "unknown_expected": 0,
        },
        "cases": [
            {
                "query": "q",
                "expected": ["A.md"],
                "first_relevant_rank": 1,
                "recall@5": 1.0,
                "recall@10": 1.0,
                "unknown_expected": [],
                "top_paths": ["A.md", "B.md", "C.md", "D.md", "E.md"],
            }
        ],
    }


def test_evaluate_reports_expected_paths_that_are_not_indexed() -> None:
    service = _StubService(
        {"q": [_chunk("Real.md")]},
        indexed={"Real.md"},
    )
    case = GoldenCase(query="q", expected=["Real.md", "Typo/Nope.md"])

    report = evaluate(service, [case])

    assert report.cases[0].unknown_expected == ["Typo/Nope.md"]
    # Metric formulas are unchanged: the unknown path still counts as unmet.
    assert report.cases[0].recall == {5: 0.5, 10: 0.5}


def test_report_to_dict_counts_unknown_expected_in_summary_and_cases() -> None:
    service = _StubService(
        {"a": [_chunk("A.md")], "b": [_chunk("B.md")]},
        indexed={"A.md", "B.md"},
    )
    cases = [
        GoldenCase(query="a", expected=["A.md", "Gone.md", "AlsoGone.md"]),
        GoldenCase(query="b", expected=["B.md"]),
    ]

    result = report_to_dict(evaluate(service, cases))

    assert result["summary"]["unknown_expected"] == 2
    assert result["cases"][0]["unknown_expected"] == ["Gone.md", "AlsoGone.md"]
    assert result["cases"][1]["unknown_expected"] == []
