from obsidian_rag.keyword_store import KeywordStore
from obsidian_rag.models import ChunkRecord


def _chunk(chunk_id: str, text: str, status: str = "todo") -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        note_id="note-1",
        text=text,
        metadata={
            "path": "Project.md",
            "status": status,
            "tags": ["task"],
            "raw_frontmatter": {"status": status},
            "derived_fields": {"due_date": "2026-02-24", "status": status},
        },
        bm25_text=text,
    )


def test_keyword_store_upsert_delete_and_filter(tmp_path) -> None:
    store = KeywordStore(tmp_path / "fts.sqlite")
    store.initialize()

    store.upsert_chunks([_chunk("c1", "finish quarterly planning today")])
    store.upsert_chunks([_chunk("c2", "book dentist appointment", status="done")])

    hits = store.search("quarterly planning", limit=5)
    assert [h.chunk_id for h in hits] == ["c1"]

    filtered = store.search("appointment", limit=5, filters={"status": "done"})
    assert [h.chunk_id for h in filtered] == ["c2"]

    store.delete_chunks(["c2"])
    assert store.search("appointment", limit=5) == []


def test_keyword_store_supports_date_range_and_wildcard_listing(tmp_path) -> None:
    store = KeywordStore(tmp_path / "fts.sqlite")
    store.initialize()

    store.upsert_chunks([_chunk("c1", "prepare roadmap", status="todo")])
    store.upsert_chunks(
        [
            ChunkRecord(
                chunk_id="c3",
                note_id="note-2",
                text="retrospective notes",
                metadata={
                    "path": "Logs.md",
                    "tags": ["log"],
                    "raw_frontmatter": {"status": "done"},
                    "derived_fields": {"due_date": "2026-03-10", "status": "done"},
                },
                bm25_text="retrospective notes",
            )
        ]
    )

    hits = store.search("*", limit=10, filters={"date_range": {"start": "2026-02-01", "end": "2026-02-28"}})
    assert [h.chunk_id for h in hits] == ["c1"]
