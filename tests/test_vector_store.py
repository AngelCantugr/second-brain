from types import SimpleNamespace

from obsidian_rag.vector_store import QdrantVectorStore


class _FakeQdrantClient:
    def __init__(self, points: list[SimpleNamespace]) -> None:
        self._points = points
        self.calls: list[dict] = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(points=self._points)


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
