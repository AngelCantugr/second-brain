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
