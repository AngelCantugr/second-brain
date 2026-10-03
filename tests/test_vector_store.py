from types import SimpleNamespace

import pytest

from second_brain.keyword_store import matches_filters
from second_brain.models import ChunkRecord
from second_brain.vector_store import InMemoryVectorStore, QdrantVectorStore


class _FakeQdrantClient:
    def __init__(self, points: list[SimpleNamespace]) -> None:
        self._points = points
        self.calls: list[dict] = []
        self.count_calls: list[dict] = []

    def count(self, **kwargs):
        self.count_calls.append(kwargs)
        return SimpleNamespace(count=len(self._points))

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        offset = kwargs.get("offset", 0)
        return SimpleNamespace(points=self._points[offset : offset + kwargs["limit"]])


def _build_store(fake_client: _FakeQdrantClient) -> QdrantVectorStore:
    store = QdrantVectorStore.__new__(QdrantVectorStore)
    store.collection_name = "chunks"
    store.client = fake_client
    return store


def test_qdrant_search_uses_query_points_and_maps_hits() -> None:
    points = [
        SimpleNamespace(
            id="chunk-1",
            score=0.88,
            payload={"text": "first", "metadata": {"path": "a.md"}},
        ),
        SimpleNamespace(
            id=42,
            score=0.31,
            payload={"text": "second", "metadata": {"path": "b.md"}},
        ),
    ]
    client = _FakeQdrantClient(points=points)
    store = _build_store(client)

    hits = store.search(query_vector=[0.1, 0.2], limit=5)

    assert client.count_calls == []
    assert len(client.calls) == 1
    assert client.calls[0] == {
        "collection_name": "chunks",
        "query": [0.1, 0.2],
        "limit": 5,
        "with_payload": True,
    }

    assert [h.chunk_id for h in hits] == ["chunk-1", "42"]
    assert [h.source for h in hits] == ["semantic", "semantic"]
    assert [h.text for h in hits] == ["first", "second"]
    assert [h.metadata for h in hits] == [{"path": "a.md"}, {"path": "b.md"}]


def test_qdrant_search_handles_missing_payload_fields() -> None:
    points = [
        SimpleNamespace(id="chunk-1", score=0.5, payload=None),
        SimpleNamespace(id="chunk-2", score=0.2, payload={"metadata": {"tag": "x"}}),
    ]
    client = _FakeQdrantClient(points=points)
    store = _build_store(client)

    hits = store.search(query_vector=[1.0], limit=2)

    assert hits[0].text == ""
    assert hits[0].metadata == {}
    assert hits[1].text == ""
    assert hits[1].metadata == {"tag": "x"}


def test_qdrant_search_ranks_once_before_filling_filtered_limit() -> None:
    points = [
        SimpleNamespace(
            id=f"near-{index}", score=1.0 - index / 100, payload={"text": "near", "metadata": {"keep": False}}
        )
        for index in range(40)
    ]
    points.append(
        SimpleNamespace(id="filtered-target", score=0.1, payload={"text": "target", "metadata": {"keep": True}})
    )
    client = _FakeQdrantClient(points=points)
    store = _build_store(client)

    hits = store.search([1.0], limit=1, metadata_filter=lambda metadata: metadata.get("keep", False))

    assert [hit.chunk_id for hit in hits] == ["filtered-target"]
    assert client.count_calls == [{"collection_name": "chunks", "exact": True}]
    assert len(client.calls) == 1
    assert client.calls[0]["limit"] == len(points)
    assert "offset" not in client.calls[0]


@pytest.mark.parametrize("filter_kind", ["empty", "selective"])
def test_qdrant_large_filtered_search_ranks_collection_once(filter_kind: str) -> None:
    points = [
        SimpleNamespace(
            id=f"chunk-{index}",
            score=1.0 - index / 5000,
            payload={"text": "fixture", "metadata": {"index": index}},
        )
        for index in range(5000)
    ]
    client = _FakeQdrantClient(points)
    store = _build_store(client)
    metadata_filter = (
        (lambda metadata: False)
        if filter_kind == "empty"
        else (lambda metadata: metadata["index"] >= 4995)
    )

    hits = store.search([1.0], limit=5, metadata_filter=metadata_filter)

    assert client.count_calls == [{"collection_name": "chunks", "exact": True}]
    assert len(client.calls) == 1
    assert client.calls[0]["limit"] == 5000
    assert "offset" not in client.calls[0]
    if filter_kind == "empty":
        assert hits == []
    else:
        assert [hit.chunk_id for hit in hits] == [f"chunk-{index}" for index in range(4995, 5000)]


def test_qdrant_filtered_search_skips_ranking_for_empty_collection() -> None:
    client = _FakeQdrantClient([])
    store = _build_store(client)

    hits = store.search([1.0], limit=5, metadata_filter=lambda metadata: True)

    assert hits == []
    assert client.count_calls == [{"collection_name": "chunks", "exact": True}]
    assert client.calls == []


def test_qdrant_local_filtered_search_handles_empty_collection(tmp_path) -> None:
    store = QdrantVectorStore(tmp_path / "qdrant", "chunks")
    store.ensure_collection(2)

    try:
        assert store.search([1.0, 0.0], metadata_filter=lambda metadata: True) == []
    finally:
        store.client.close()


def test_qdrant_local_search_matches_filters_after_old_candidate_window(tmp_path) -> None:
    store = QdrantVectorStore(tmp_path / "qdrant", "chunks")
    store.ensure_collection(2)
    distractors = [
        ChunkRecord(
            chunk_id=f"00000000-0000-0000-0000-{index + 1:012d}",
            note_id=f"near-note-{index}",
            text="nearby note",
            metadata={"tags": ["other"]},
            bm25_text="nearby note",
        )
        for index in range(40)
    ]
    target = ChunkRecord(
        chunk_id="00000000-0000-0000-0000-000000000041",
        note_id="target-note",
        text="distant note",
        metadata={"tags": ["wanted"]},
        bm25_text="distant note",
    )
    store.upsert_chunks([*distractors, target], [[1.0, 0.0] for _ in distractors] + [[-1.0, 0.0]])

    try:
        hits = store.search(
            [1.0, 0.0],
            limit=1,
            metadata_filter=lambda metadata: metadata.get("tags") == ["wanted"],
        )
        assert [hit.chunk_id for hit in hits] == ["00000000-0000-0000-0000-000000000041"]
    finally:
        store.client.close()


@pytest.mark.parametrize(
    ("filters", "matching_metadata"),
    [
        pytest.param(
            {"tags": "GYM"},
            {"tags": ["gym", "strength"]},
            id="case-insensitive-tag-string",
        ),
        pytest.param(
            {"tags": ["gym", "strength"]},
            {"tags": ["GYM", "Strength"]},
            id="case-insensitive-tag-list-subset",
        ),
        pytest.param(
            {"date_range": {"start": "2026-10-01", "end": "2026-10-03"}},
            [
                {
                    "derived_fields": {
                        "due_date": "2026-10-01",
                        "deadline_date": "2020-01-01",
                    },
                    "raw_frontmatter": {"date": "2020-01-01"},
                },
                {"derived_fields": {"due_date": "2026-10-03"}},
            ],
            id="date-range-inclusive-boundary-and-field-priority",
        ),
        pytest.param(
            {"path_prefix": "Projects/Issue31/"},
            {"path": "Projects/Issue31/filtered.md"},
            id="path-prefix",
        ),
        pytest.param(
            {"frontmatter_contains": {"status": "done", "owner": "Angel"}},
            {"raw_frontmatter": {"status": "done", "owner": "Angel"}},
            id="arbitrary-frontmatter",
        ),
        pytest.param(
            {"release_track": "blue"},
            {"derived_fields": {"release_track": "blue"}},
            id="derived-field-equality",
        ),
    ],
)
def test_qdrant_local_filter_results_match_reference_predicate(
    tmp_path, filters: dict, matching_metadata: dict | list[dict]
) -> None:
    """Keep Qdrant's full-ranking filter results identical to the Python reference matcher."""

    store = QdrantVectorStore(tmp_path / "qdrant", "chunks")
    store.ensure_collection(2)
    metadata_variants = (
        matching_metadata if isinstance(matching_metadata, list) else [matching_metadata, matching_metadata]
    )
    distractors = [
        ChunkRecord(
            chunk_id=f"00000000-0000-0000-0000-{index + 1:012d}",
            note_id=f"near-note-{index}",
            text="unfiltered high-ranking note",
            metadata={
                "path": f"Elsewhere/{index}.md",
                "tags": ["other"],
                "derived_fields": {"due_date": "2030-01-01"},
                "raw_frontmatter": {"status": "open"},
            },
            bm25_text="unfiltered high-ranking note",
        )
        for index in range(40)
    ]
    if "date_range" in filters:
        # The higher-priority due date is outside the range, despite the
        # lower-priority deadline date falling on an inclusive boundary.
        distractors[0].metadata["derived_fields"] = {
            "due_date": "2026-09-30",
            "deadline_date": "2026-10-01",
        }
    matching = [
        ChunkRecord(
            chunk_id=f"00000000-0000-0000-0000-{index:012d}",
            note_id=f"matching-note-{index}",
            text="matching low-ranking note",
            metadata={**metadata, "path": metadata.get("path", f"Elsewhere/match-{index}.md")},
            bm25_text="matching low-ranking note",
        )
        for index, metadata in zip((41, 42), metadata_variants, strict=True)
    ]
    store.upsert_chunks(
        [*distractors, *matching],
        [[1.0, 0.0] for _ in distractors] + [[-1.0, 0.0], [-1.0, 0.0]],
    )

    try:
        ranked_hits = store.search([1.0, 0.0], limit=100)
        expected_ids = [
            hit.chunk_id for hit in ranked_hits if matches_filters(hit.metadata, filters)
        ][:2]
        actual_hits = store.search(
            [1.0, 0.0],
            limit=2,
            metadata_filter=lambda metadata: matches_filters(metadata, filters),
        )

        assert [hit.chunk_id for hit in actual_hits] == expected_ids
        assert len(actual_hits) == 2
    finally:
        store.client.close()


def test_qdrant_local_filtered_search_exhausts_pages_for_partial_and_empty_results(tmp_path) -> None:
    store = QdrantVectorStore(tmp_path / "qdrant", "chunks")
    store.ensure_collection(2)
    chunks = [
        ChunkRecord(
            chunk_id=f"00000000-0000-0000-0000-{index + 1:012d}",
            note_id=f"note-{index}",
            text="stored note",
            metadata={"tags": ["wanted"] if index == 40 else ["other"]},
            bm25_text="stored note",
        )
        for index in range(41)
    ]
    store.upsert_chunks(chunks, [[1.0, 0.0] for _ in range(40)] + [[-1.0, 0.0]])

    try:
        partial_hits = store.search(
            [1.0, 0.0],
            limit=3,
            metadata_filter=lambda metadata: matches_filters(metadata, {"tags": ["wanted"]}),
        )
        empty_hits = store.search(
            [1.0, 0.0],
            limit=3,
            metadata_filter=lambda metadata: matches_filters(metadata, {"tags": ["missing"]}),
        )

        assert [hit.chunk_id for hit in partial_hits] == ["00000000-0000-0000-0000-000000000041"]
        assert empty_hits == []
    finally:
        store.client.close()


def test_in_memory_get_by_path_returns_only_matching_chunks() -> None:
    store = InMemoryVectorStore()
    store.upsert_chunks(
        [
            ChunkRecord(
                chunk_id="a1",
                note_id="note-a",
                text="note A",
                metadata={"path": "A.md"},
                bm25_text="note A",
            ),
            ChunkRecord(
                chunk_id="b1",
                note_id="note-b",
                text="note B",
                metadata={"path": "B.md"},
                bm25_text="note B",
            ),
        ],
        [[0.1, 0.2], [0.3, 0.4]],
    )

    hits = store.get_by_path("A.md")

    assert [h.chunk_id for h in hits] == ["a1"]
    assert hits[0].source == "semantic"
    assert hits[0].text == "note A"


def test_in_memory_get_by_path_returns_empty_for_unknown_path() -> None:
    store = InMemoryVectorStore()

    assert store.get_by_path("missing.md") == []


def test_in_memory_search_preserves_dot_score_and_exposes_cosine_similarity() -> None:
    store = InMemoryVectorStore()
    chunk = ChunkRecord(
        chunk_id="cosine",
        note_id="note-cosine",
        text="cosine candidate",
        metadata={},
        bm25_text="cosine candidate",
    )
    store.upsert_chunks([chunk], [[3.0, 4.0]])

    hits = store.search([1.0, 0.0])

    assert hits[0].score == pytest.approx(3.0)
    assert hits[0].semantic_score == pytest.approx(0.6)


def test_qdrant_unbounded_search_uses_one_complete_ranked_query() -> None:
    points = [
        SimpleNamespace(
            id=f"candidate-{index}",
            score=0.99 - index / 100,
            payload={"text": "candidate", "metadata": {}},
        )
        for index in range(40)
    ]
    points.append(
        SimpleNamespace(
            id="qualifying-after-first-page",
            score=0.88,
            payload={"text": "qualifying candidate", "metadata": {}},
        )
    )
    client = _FakeQdrantClient(points)
    store = _build_store(client)

    hits = store.search([1.0], limit=None)

    assert len(hits) == 41
    assert hits[-1].chunk_id == "qualifying-after-first-page"
    assert client.count_calls == [{"collection_name": "chunks", "exact": True}]
    assert len(client.calls) == 1
    assert client.calls[0]["limit"] == len(points)
    assert "offset" not in client.calls[0]


def test_qdrant_large_unbounded_search_ranks_all_candidates_once() -> None:
    points = [
        SimpleNamespace(
            id=f"candidate-{index}",
            score=1.0 - index / 5000,
            payload={"text": "candidate", "metadata": {}},
        )
        for index in range(5000)
    ]
    client = _FakeQdrantClient(points)
    store = _build_store(client)

    hits = store.search([1.0], limit=None)

    assert len(hits) == 5000
    assert client.count_calls == [{"collection_name": "chunks", "exact": True}]
    assert len(client.calls) == 1
    assert client.calls[0]["limit"] == 5000
    assert "offset" not in client.calls[0]


def test_qdrant_unbounded_search_skips_query_for_empty_collection() -> None:
    client = _FakeQdrantClient([])
    store = _build_store(client)

    assert store.search([1.0], limit=None) == []
    assert client.count_calls == [{"collection_name": "chunks", "exact": True}]
    assert client.calls == []
