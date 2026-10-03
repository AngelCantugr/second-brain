import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from second_brain.config import RagConfig
from second_brain.keyword_store import matches_filters
from second_brain.models import ChunkRecord
from second_brain.models import RetrievalHit
from second_brain.service import MAX_TOP_K, RagService
from second_brain.vector_store import QdrantVectorStore


class _StubEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 0.0, 0.0] for text in texts]


def _build_service(
    tmp_path: Path, *, use_in_memory_vector: bool = True, graph_enabled: bool = True
) -> RagService:
    vault = tmp_path / "vault"
    vault.mkdir(exist_ok=True)
    config = RagConfig(
        vault_path=vault,
        qdrant_path=tmp_path / "qdrant",
        fts_path=tmp_path / "fts.sqlite",
        sync_state_path=tmp_path / "sync_state.sqlite",
        graph_enabled=graph_enabled,
    )
    return RagService(config, use_in_memory_vector=use_in_memory_vector)


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


def test_query_compact_mode_keeps_citations_without_answer_draft(tmp_path: Path, monkeypatch) -> None:
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
    compact = service.query("source", verbose=False, recency_boost=0.3)

    assert verbose["chunks"][0]["metadata"]["path"] == "Notes/Source.md"
    assert compact["citations"] == verbose["citations"] == [
        {"chunk_id": "chunk-1", "path": "Notes/Source.md", "heading_path": "Evidence"}
    ]
    assert "answer_draft" not in compact
    assert "answer_draft" not in verbose
    assert compact["debug_scores"] == verbose["debug_scores"]
    assert compact["debug_scores"] == [compact["chunks"][0]["score"]]
    assert compact["chunks"][0]["path"] == "Notes/Source.md"


def test_query_with_no_hits_returns_empty_citations_chunks_and_scores(tmp_path: Path, monkeypatch) -> None:
    service = _build_service(tmp_path)
    service.embedder = _StubEmbedder()
    monkeypatch.setattr(service.vector_store, "search", lambda *args, **kwargs: [])
    monkeypatch.setattr(service.keyword_store, "search", lambda *args, **kwargs: [])

    result = service.query("missing source")

    assert result == {"citations": [], "chunks": [], "debug_scores": []}


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


@pytest.mark.parametrize("recency_boost", [float("nan"), float("inf"), -0.01, 1.01, True, "1"])
def test_search_rejects_invalid_recency_boost(tmp_path: Path, recency_boost) -> None:
    service = _build_service(tmp_path)

    with pytest.raises(ValueError, match="recency_boost"):
        service.search("hello", recency_boost=recency_boost)


@pytest.mark.parametrize("modified_since", [
    "not-a-date", "2026-10-01T00:00:00+00:99",
    "2026-10-01T00:00:00+00:00:99", "2026-10-01T00:00:00+24:00",
])
def test_search_rejects_invalid_modified_since_before_backend_search(tmp_path: Path, modified_since: str) -> None:
    service = _build_service(tmp_path)

    with pytest.raises(ValueError, match="modified_since"):
        service.search("hello", filters={"modified_since": modified_since})


def test_recency_boost_reranks_close_hits_without_swamping_two_signal_relevance(
    tmp_path: Path, monkeypatch
) -> None:
    from time import time

    service = _build_service(tmp_path)
    service.embedder = _StubEmbedder()
    now = time()
    old = RetrievalHit("old", 1.0, "semantic", "old", {"mtime": now - 90 * 86400})
    fresh = RetrievalHit("fresh", 0.9, "semantic", "fresh", {"mtime": now})
    monkeypatch.setattr(service.vector_store, "search", lambda *args, **kwargs: [old, fresh])
    monkeypatch.setattr(service.keyword_store, "search", lambda *args, **kwargs: [])

    default = service.search("hello", top_k=2)
    boosted = service.search("hello", top_k=2, recency_boost=1.0)

    assert [hit["chunk_id"] for hit in default["hits"]] == ["old", "fresh"]
    assert [hit["chunk_id"] for hit in boosted["hits"]] == ["fresh", "old"]
    assert boosted["hits"][0]["score"] == pytest.approx(1 / 61 + 1 / 62)

    older_relevant = RetrievalHit("relevant", 1.0, "semantic", "relevant", {"mtime": now - 90 * 86400})
    fresher_weak = RetrievalHit("weak", 0.9, "semantic", "weak", {"mtime": now})
    monkeypatch.setattr(
        service.vector_store, "search", lambda *args, **kwargs: [older_relevant, fresher_weak]
    )
    monkeypatch.setattr(
        service.keyword_store, "search", lambda *args, **kwargs: [older_relevant]
    )

    bounded = service.search("hello", top_k=2, recency_boost=1.0)
    assert [hit["chunk_id"] for hit in bounded["hits"]] == ["relevant", "weak"]


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


def test_search_excludes_configured_status_before_store_limits(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    service.embedder = _StubEmbedder()
    service.config.exclude_status = ["superseded"]
    chunks = [
        ChunkRecord(
            "superseded",
            "old-note",
            "project planning source",
            {"raw_frontmatter": {"status": "superseded"}},
            "project planning source",
        ),
        ChunkRecord(
            "active",
            "live-note",
            "project planning source",
            {"raw_frontmatter": {"status": "active"}},
            "project planning source",
        ),
    ]
    service.vector_store.upsert_chunks(chunks, [[100.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    service.keyword_store.upsert_chunks(chunks)

    result = service.search("project planning", top_k=1)

    assert [hit["chunk_id"] for hit in result["hits"]] == ["active"]


def test_query_exclude_status_empty_list_disables_configured_exclusion(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    service.embedder = _StubEmbedder()
    service.config.exclude_status = ["superseded"]
    chunk = ChunkRecord(
        "superseded",
        "old-note",
        "project planning source",
        {"raw_frontmatter": {"status": "superseded"}},
        "project planning source",
    )
    service.vector_store.upsert_chunks([chunk], [[100.0, 0.0, 0.0]])
    service.keyword_store.upsert_chunks([chunk])

    result = service.search("project planning", filters={"exclude_status": []})

    assert [hit["chunk_id"] for hit in result["hits"]] == ["superseded"]


def test_search_rejects_malformed_exclude_status_even_without_candidates(
    tmp_path: Path,
) -> None:
    service = _build_service(tmp_path)
    service.embedder = _StubEmbedder()

    with pytest.raises(ValueError, match="exclude_status"):
        service.search("nothing", filters={"exclude_status": "superseded"})


def test_search_composes_modified_since_with_tags_and_path_prefix(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    service.embedder = _StubEmbedder()
    chunks = [
        ChunkRecord(
            "old-good-path",
            "old-good-note",
            "quarterly planning",
            {"path": "Projects/Issue35/old.md", "tags": ["wanted"], "mtime": 1790726400.0},
            "quarterly planning",
        ),
        ChunkRecord(
            "fresh-other-path",
            "fresh-other-note",
            "quarterly planning",
            {"path": "Archive/fresh.md", "tags": ["wanted"], "mtime": 1790812800.0},
            "quarterly planning",
        ),
        ChunkRecord(
            "fresh-good-path",
            "fresh-good-note",
            "quarterly planning",
            {"path": "Projects/Issue35/fresh.md", "tags": ["wanted"], "mtime": 1790812800.0},
            "quarterly planning",
        ),
    ]
    service.vector_store.upsert_chunks(chunks, [[1.0, 0.0, 0.0]] * len(chunks))
    service.keyword_store.upsert_chunks(chunks)

    result = service.search(
        "quarterly planning",
        filters={
            "modified_since": "2026-10-01",
            "tags": ["WANTED"],
            "path_prefix": "Projects/Issue35/",
        },
    )

    assert [hit["chunk_id"] for hit in result["hits"]] == ["fresh-good-path"]


def test_modified_since_tracks_controlled_vault_file_mtime_through_qdrant_and_query(
    tmp_path: Path,
) -> None:
    vault = tmp_path / "vault"
    target_path = vault / "Projects" / "Issue35" / "boundary.md"
    other_path = vault / "Projects" / "Issue35" / "old.md"
    tag_mismatch_path = vault / "Projects" / "Issue35" / "untagged.md"
    path_mismatch_path = vault / "Archive" / "boundary.md"
    timestamp_ns = 1_790_812_800_000_000_000  # 2026-10-01T00:00:00Z

    contents = {
        target_path: "---\ntags: [wanted]\ndue: 2026-10-01\n---\n# Topic\nquarterly planning retrieval phrase boundary\n",
        other_path: "---\ntags: [wanted]\ndue: 2026-10-01\n---\n# Topic\nquarterly planning retrieval phrase old\n",
        tag_mismatch_path: "---\ntags: [other]\ndue: 2026-10-01\n---\n# Topic\nquarterly planning retrieval phrase untagged\n",
        path_mismatch_path: "---\ntags: [wanted]\ndue: 2026-10-01\n---\n# Topic\nquarterly planning retrieval phrase archive\n",
    }
    for path, content in contents.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        mtime_ns = timestamp_ns - 86_400_000_000_000 if path == other_path else timestamp_ns
        os.utime(path, ns=(mtime_ns, mtime_ns))

    service = _build_service(
        tmp_path, use_in_memory_vector=False, graph_enabled=False
    )
    service.embedder = _StubEmbedder()
    service.indexer.embedder = service.embedder
    try:
        sync_result = service.sync(mode="full")
        assert sync_result["processed"] == 4
        assert sync_result["errors"] == []

        all_indexed = service.keyword_store.search("*", limit=None)
        assert len(all_indexed) == 4
        assert {hit.metadata["path"] for hit in all_indexed} == {
            "Projects/Issue35/boundary.md",
            "Projects/Issue35/old.md",
            "Projects/Issue35/untagged.md",
            "Archive/boundary.md",
        }
        indexed_keyword = service.keyword_store.chunks_by_path(
            "Projects/Issue35/boundary.md"
        )
        assert len(indexed_keyword) == 1
        stored_points, _ = service.vector_store.client.scroll(
            collection_name=service.vector_store.collection_name,
            with_payload=True,
            limit=10,
        )
        assert len(stored_points) == 4

        filters = {
            "modified_since": "2026-10-01T00:00:00Z",
            "date_range": {"start": "2026-10-01", "end": "2026-10-01"},
            "tags": ["WANTED"],
            "path_prefix": "Projects/Issue35/",
        }
        indexed_chunks = service.vector_store.get_by_path("Projects/Issue35/boundary.md")
        assert len(indexed_chunks) == 1
        indexed_metadata = indexed_chunks[0].metadata
        assert indexed_metadata["mtime"] == pytest.approx(timestamp_ns / 1_000_000_000)
        assert indexed_metadata["tags"] == ["wanted"]
        assert matches_filters(indexed_metadata, filters)
        search = service.search("quarterly planning retrieval phrase", filters=filters)
        query = service.query("quarterly planning retrieval phrase", filters=filters)

        assert [hit["metadata"]["path"] for hit in search["hits"]] == [
            "Projects/Issue35/boundary.md"
        ]
        assert [hit["metadata"]["mtime"] for hit in search["hits"]] == [pytest.approx(1790812800.0)]
        assert [citation["path"] for citation in query["citations"]] == [
            "Projects/Issue35/boundary.md"
        ]
    finally:
        service.vector_store.client.close()


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


def test_status_reports_clean_and_stale_disk_files_without_syncing(tmp_path: Path, monkeypatch) -> None:
    service = _build_service(tmp_path, graph_enabled=False)
    service.embedder.health = lambda: True
    indexed = service.config.vault_path / "indexed.md"
    indexed.write_text("original note", encoding="utf-8")
    excluded = service.config.vault_path / "excluded.md"
    excluded.write_text("excluded before indexing", encoding="utf-8")
    service.sync(mode="full")
    tracked_before = service.sync_state.tracked_paths()
    service.config.exclude_globs = ["excluded.md"]
    service.config.watch_enabled = False

    clean = service.status()
    assert (clean["stale_files"], clean["untracked_files"], clean["missing_files"]) == (0, 0, 0)
    assert clean["watcher_last_event"] is None

    original_mtime = indexed.stat().st_mtime_ns
    indexed.write_text("edited note", encoding="utf-8")
    os.utime(indexed, ns=(original_mtime, original_mtime))
    edited = service.status()
    assert edited["stale_files"] == 1

    (service.config.vault_path / "new.md").write_text("new note", encoding="utf-8")
    indexed.unlink()
    excluded.unlink()

    status = service.status()
    assert (status["stale_files"], status["untracked_files"], status["missing_files"]) == (0, 1, 1)
    assert service.sync_state.tracked_paths() == tracked_before
