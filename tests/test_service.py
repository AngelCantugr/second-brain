from pathlib import Path

from obsidian_rag.config import RagConfig
from obsidian_rag.models import ChunkRecord
from obsidian_rag.service import RagService


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
