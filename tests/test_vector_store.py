from types import SimpleNamespace

from second_brain.models import ChunkRecord
from second_brain.vector_store import InMemoryVectorStore, QdrantVectorStore


class _FakeQdrantClient:
    def __init__(self, points: list[SimpleNamespace]) -> None:
        self._points = points
        self.calls: list[dict] = []

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


def test_qdrant_search_pages_until_filtered_limit_is_filled() -> None:
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
    assert [call["offset"] for call in client.calls] == [0, 32]


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
