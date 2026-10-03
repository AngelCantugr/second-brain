import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from second_brain.config import RagConfig
from second_brain.models import ChunkRecord
from second_brain.models import RetrievalHit
from second_brain.service import MAX_TOP_K, RagService
from second_brain.vector_store import QdrantVectorStore


class _StubEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 0.0, 0.0] for text in texts]


def _build_service(tmp_path: Path) -> RagService:
    vault = tmp_path / "vault"
    vault.mkdir()
    config = RagConfig(
        vault_path=vault,
        qdrant_path=tmp_path / "qdrant",
        fts_path=tmp_path / "fts.sqlite",
        sync_state_path=tmp_path / "sync_state.sqlite",
    )
    return RagService(config, use_in_memory_vector=True)


@pytest.mark.parametrize("query", ["", "   "])
def test_search_rejects_empty_or_whitespace_query(tmp_path: Path, query: str) -> None:
    service = _build_service(tmp_path)

    with pytest.raises(ValueError, match="query"):
        service.search(query=query)


@pytest.mark.parametrize("query", ["", "   "])
def test_query_rejects_empty_or_whitespace_query(tmp_path: Path, query: str) -> None:
    service = _build_service(tmp_path)

    with pytest.raises(ValueError, match="query"):
        service.query(query=query)


@pytest.mark.parametrize("top_k", [0, -1, MAX_TOP_K + 1, 5000])
def test_search_rejects_top_k_out_of_bounds(tmp_path: Path, top_k: int) -> None:
    service = _build_service(tmp_path)

    with pytest.raises(ValueError, match="top_k"):
        service.search(query="hello", top_k=top_k)


def test_search_accepts_top_k_at_max_boundary(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    service.embedder = _StubEmbedder()

    result = service.search(query="hello", top_k=MAX_TOP_K)

    assert result["hits"] == []


def test_search_exposes_underlying_scores_and_filters_before_top_k(tmp_path: Path, monkeypatch) -> None:
    service = _build_service(tmp_path)
    service.embedder = _StubEmbedder()
    semantic = [
        RetrievalHit("strong", 0.91, "semantic", "strong text", {}),
        RetrievalHit("weak", 0.31, "semantic", "weak text", {}),
    ]
    keyword = [
        RetrievalHit("strong", 0.6, "keyword", "strong text", {}),
        RetrievalHit("weak", 0.8, "keyword", "weak text", {}),
        RetrievalHit("keyword-only", 0.7, "keyword", "keyword text", {}),
    ]
    vector_options = {}
    keyword_options = {}

    def semantic_search(*args, **kwargs):
        vector_options.update(kwargs)
        return semantic

    def keyword_search(*args, **kwargs):
        keyword_options.update(kwargs)
        return keyword

    monkeypatch.setattr(service.vector_store, "search", semantic_search)
    monkeypatch.setattr(service.keyword_store, "search", keyword_search)

    result = service.search("hello", top_k=3, min_score=0.5)

    assert [hit["chunk_id"] for hit in result["hits"]] == ["strong"]
    assert result["hits"][0]["semantic_score"] == 0.91
    assert result["hits"][0]["keyword_score"] == 0.6
    assert result["hits"][0]["score"] == pytest.approx(2 / 61)
    assert vector_options["limit"] is None
    assert keyword_options["limit"] is None


def test_search_compact_mode_reduces_repeated_frontmatter_and_preserves_score_signals(
    tmp_path: Path, monkeypatch
) -> None:
    service = _build_service(tmp_path)
    service.embedder = _StubEmbedder()
    frontmatter = {
        "raw_frontmatter": "---\n" + "project: research\n" * 80 + "---",
        "tags": [f"tag-{index}" for index in range(30)],
        "links": [f"Notes/Related-{index}.md" for index in range(30)],
        "tasks": [{"text": "review source material", "status": "open"}] * 10,
        "derived_fields": {f"field-{index}": "derived value" * 5 for index in range(10)},
        "path": "Notes/Research.md",
        "note_title": "Research",
        "heading_path": "Evidence / Findings",
    }
    hits = [
        RetrievalHit(f"chunk-{index}", 0.5, "semantic", "repeated chunk text", frontmatter, 0.9, 0.4)
        for index in range(5)
    ]
    keyword_hits = [
        RetrievalHit(f"chunk-{index}", 0.4, "keyword", "repeated chunk text", frontmatter)
        for index in range(5)
    ]
    monkeypatch.setattr(service.vector_store, "search", lambda *args, **kwargs: hits)
    monkeypatch.setattr(service.keyword_store, "search", lambda *args, **kwargs: keyword_hits)

    verbose = service.search("research", top_k=5)
    compact = service.search("research", top_k=5, verbose=False)

    assert verbose["hits"][0]["metadata"] == frontmatter
    assert "source" in verbose["hits"][0]
    assert len(json.dumps(compact)) <= len(json.dumps(verbose)) * 0.4
    assert compact["hits"][0] == {
        "chunk_id": "chunk-0",
        "score": pytest.approx(2 / 61),
        "semantic_score": 0.9,
        "keyword_score": 0.4,
        "text": "repeated chunk text",
        "path": "Notes/Research.md",
        "note_title": "Research",
        "heading_path": "Evidence / Findings",
    }


def test_query_compact_mode_keeps_citations_and_extractive_answer(tmp_path: Path, monkeypatch) -> None:
    service = _build_service(tmp_path)
    service.embedder = _StubEmbedder()
    hit = RetrievalHit(
        "chunk-1",
        0.5,
        "semantic",
        "The relevant source text.",
        {"path": "Notes/Source.md", "note_title": "Source", "heading_path": "Evidence"},
        0.9,
        None,
    )
    monkeypatch.setattr(service.vector_store, "search", lambda *args, **kwargs: [hit])
    monkeypatch.setattr(service.keyword_store, "search", lambda *args, **kwargs: [])

    verbose = service.query("source")
    compact = service.query("source", verbose=False)

    assert verbose["chunks"][0]["metadata"]["path"] == "Notes/Source.md"
    assert compact["citations"] == verbose["citations"] == [
        {"chunk_id": "chunk-1", "path": "Notes/Source.md", "heading_path": "Evidence"}
    ]
    assert compact["answer_draft"] == verbose["answer_draft"]
    assert compact["chunks"][0]["path"] == "Notes/Source.md"


def test_search_threshold_can_return_zero_hits(tmp_path: Path, monkeypatch) -> None:
    service = _build_service(tmp_path)
    service.embedder = _StubEmbedder()
    monkeypatch.setattr(
        service.vector_store,
        "search",
        lambda *args, **kwargs: [RetrievalHit("weak", 0.2, "semantic", "weak", {})],
    )
    monkeypatch.setattr(
        service.keyword_store,
        "search",
        lambda *args, **kwargs: [RetrievalHit("keyword-only", 0.9, "keyword", "kw", {})],
    )

    assert service.search("hello", min_score=0.8)["hits"] == []


def test_in_memory_search_keeps_default_dot_ranking_and_thresholds_by_cosine(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    service.embedder = _StubEmbedder()
    chunks = [
        ChunkRecord("large-norm", "note-a", "hello semantic retrieval", {}, "hello semantic retrieval"),
        ChunkRecord("unit-norm", "note-b", "hello semantic retrieval", {}, "hello semantic retrieval"),
    ]
    service.vector_store.upsert_chunks(chunks, [[10.0, 10.0, 0.0], [1.0, 0.0, 0.0]])
    service.keyword_store.upsert_chunks(chunks)

    default_hits = service.search("hello", top_k=1)["hits"]
    threshold_hits = service.search("hello", top_k=1, min_score=0.9)["hits"]

    assert [hit["chunk_id"] for hit in default_hits] == ["large-norm"]
    assert default_hits[0]["semantic_score"] == pytest.approx(2**-0.5)
    assert [hit["chunk_id"] for hit in threshold_hits] == ["unit-norm"]
    assert threshold_hits[0]["semantic_score"] == pytest.approx(1.0)


@pytest.mark.parametrize("min_score", [float("nan"), float("inf"), -1.01, 1.01, True, "bad", 10**400, -(10**400)])
def test_search_rejects_invalid_min_score(
    tmp_path: Path, min_score: float, monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _build_service(tmp_path)

    def unexpected_work(*args, **kwargs):
        raise AssertionError("invalid min_score reached embedding/backend")

    monkeypatch.setattr(service.embedder, "embed", unexpected_work)
    monkeypatch.setattr(service.vector_store, "search", unexpected_work)
    monkeypatch.setattr(service.keyword_store, "search", unexpected_work)
    with pytest.raises(ValueError, match="min_score"):
        service.search("hello", min_score=min_score)


def test_search_applies_filter_before_semantic_candidate_limit(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    service.embedder = _StubEmbedder()
    distractors = [
        ChunkRecord(
            chunk_id=f"near-{index}",
            note_id=f"near-note-{index}",
            text="unfiltered nearby note",
            metadata={"path": f"near-{index}.md", "tags": ["other"]},
            bm25_text="unrelated wording",
        )
        for index in range(8)
    ]
    target = ChunkRecord(
        chunk_id="filtered-target",
        note_id="target-note",
        text="matching note far from the query in vector space",
        metadata={"path": "target.md", "tags": ["wanted"]},
        bm25_text="target has different wording",
    )
    service.vector_store.upsert_chunks(
        [*distractors, target], [[1000.0, 0.0, 0.0] for _ in distractors] + [[1.0, 0.0, 0.0]]
    )

    result = service.search(query="what did I do", filters={"tags": ["wanted"]}, top_k=1)

    assert [hit["chunk_id"] for hit in result["hits"]] == ["filtered-target"]


def test_unfiltered_threshold_search_requests_one_complete_vector_ranking(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    service.embedder = _StubEmbedder()
    point = SimpleNamespace(
        id="threshold-match",
        score=0.9,
        payload={"text": "hello semantic retrieval", "metadata": {"path": "match.md"}},
    )

    class _Client:
        def __init__(self) -> None:
            self.count_calls: list[dict] = []
            self.query_calls: list[dict] = []

        def count(self, **kwargs):
            self.count_calls.append(kwargs)
            return SimpleNamespace(count=1)

        def query_points(self, **kwargs):
            self.query_calls.append(kwargs)
            return SimpleNamespace(points=[point])

    client = _Client()
    vector_store = QdrantVectorStore.__new__(QdrantVectorStore)
    vector_store.collection_name = "chunks"
    vector_store.client = client
    service.vector_store = vector_store

    result = service.search("hello", top_k=1, min_score=0.5)

    assert [hit["chunk_id"] for hit in result["hits"]] == ["threshold-match"]
    assert client.count_calls == [{"collection_name": "chunks", "exact": True}]
    assert len(client.query_calls) == 1
    assert client.query_calls[0]["limit"] == 1
    assert "offset" not in client.query_calls[0]


def test_note_context_reports_backlinks_from_other_notes(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    service.keyword_store.upsert_chunks(
        [
            ChunkRecord(
                chunk_id="a1",
                note_id="note-a",
                text="note A references note B",
                metadata={"path": "A.md", "note_title": "A", "links": ["Note B"]},
                bm25_text="note A references note B",
            ),
            ChunkRecord(
                chunk_id="b1",
                note_id="note-b",
                text="note B content",
                metadata={"path": "Note B.md", "note_title": "Note B", "links": []},
                bm25_text="note B content",
            ),
        ]
    )

    context = service.note_context("Note B.md")

    assert context["backlinks"] == ["A.md"]
    assert context["outlinks"] == []


def test_build_extractive_answer_includes_chunk_text_not_just_citations() -> None:
    hits = [
        {
            "text": "Prompt engineering is about framing instructions for a model.",
            "metadata": {"path": "Notes/PE.md", "heading_path": "Intro"},
        }
    ]

    answer = RagService._build_extractive_answer(hits)

    assert "Prompt engineering is about framing instructions" in answer
    assert "Notes/PE.md :: Intro" in answer


def test_build_extractive_answer_truncates_long_chunk_text() -> None:
    hits = [
        {
            "text": "word " * 200,
            "metadata": {"path": "Long.md", "heading_path": "root"},
        }
    ]

    answer = RagService._build_extractive_answer(hits, snippet_chars=50)

    assert answer.endswith("...")
    assert len(answer.split("] ", 1)[1]) <= 53


def test_build_extractive_answer_handles_no_hits() -> None:
    assert RagService._build_extractive_answer([]) == "No relevant context found."


def test_note_context_returns_no_backlinks_when_nothing_links(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    service.keyword_store.upsert_chunks(
        [
            ChunkRecord(
                chunk_id="c1",
                note_id="note-c",
                text="isolated note",
                metadata={"path": "Isolated.md", "note_title": "Isolated", "links": []},
                bm25_text="isolated note",
            )
        ]
    )

    context = service.note_context("Isolated.md")

    assert context["backlinks"] == []


def test_note_context_falls_back_to_vector_store_when_keyword_store_is_missing_chunk(
    tmp_path: Path,
) -> None:
    # Simulates issue #12: a chunk that made it into the vector store (so
    # rag.search reports the note as indexed) but never landed in the
    # keyword store, e.g. due to a crash between the two upsert calls.
    service = _build_service(tmp_path)
    chunk = ChunkRecord(
        chunk_id="v1",
        note_id="note-v",
        text="only in the vector store",
        metadata={"path": "Drifted.md", "note_title": "Drifted", "links": []},
        bm25_text="only in the vector store",
    )
    service.vector_store.upsert_chunks([chunk], [[0.1, 0.2, 0.3]])

    context = service.note_context("Drifted.md")

    assert context["chunk_ids"] == ["v1"]
    assert context["chunk_count"] == 1


def test_note_context_deduplicates_chunk_present_in_both_stores(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    chunk = ChunkRecord(
        chunk_id="d1",
        note_id="note-d",
        text="in both stores",
        metadata={"path": "Both.md", "note_title": "Both", "links": []},
        bm25_text="in both stores",
    )
    service.keyword_store.upsert_chunks([chunk])
    service.vector_store.upsert_chunks([chunk], [[0.1, 0.2, 0.3]])

    context = service.note_context("Both.md")

    assert context["chunk_ids"] == ["d1"]
    assert context["chunk_count"] == 1
