import pytest

from obsidian_rag.keyword_store import KeywordStore, matches_filters
from obsidian_rag.models import ChunkRecord


def _chunk(
    chunk_id: str,
    text: str,
    status: str = "todo",
    path: str = "Project.md",
    links: list[str] | None = None,
    note_title: str | None = None,
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        note_id="note-1",
        text=text,
        metadata={
            "path": path,
            "note_title": note_title if note_title is not None else path.removesuffix(".md"),
            "status": status,
            "tags": ["task"],
            "links": links or [],
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


def test_keyword_store_tags_filter_accepts_bare_string(tmp_path) -> None:
    store = KeywordStore(tmp_path / "fts.sqlite")
    store.initialize()

    chunk = _chunk("c1", "LangGraph agent patterns")
    chunk.metadata["tags"] = ["LangGraph"]
    store.upsert_chunks([chunk])

    hits = store.search("LangGraph agent patterns", limit=5, filters={"tags": "LangGraph"})
    assert [h.chunk_id for h in hits] == ["c1"]


def test_keyword_store_tags_filter_accepts_list(tmp_path) -> None:
    store = KeywordStore(tmp_path / "fts.sqlite")
    store.initialize()

    chunk = _chunk("c1", "LangGraph agent patterns")
    chunk.metadata["tags"] = ["LangGraph"]
    store.upsert_chunks([chunk])

    hits = store.search("LangGraph agent patterns", limit=5, filters={"tags": ["LangGraph"]})
    assert [h.chunk_id for h in hits] == ["c1"]


def test_keyword_store_tags_filter_accepts_none(tmp_path) -> None:
    store = KeywordStore(tmp_path / "fts.sqlite")
    store.initialize()

    chunk = _chunk("c1", "LangGraph agent patterns")
    chunk.metadata["tags"] = ["LangGraph"]
    store.upsert_chunks([chunk])

    hits = store.search("LangGraph agent patterns", limit=5, filters={"tags": None})
    assert [h.chunk_id for h in hits] == ["c1"]


def test_keyword_store_tags_filter_bare_string_excludes_non_matching_tag(tmp_path) -> None:
    store = KeywordStore(tmp_path / "fts.sqlite")
    store.initialize()

    chunk = _chunk("c1", "LangGraph agent patterns")
    chunk.metadata["tags"] = ["LangGraph"]
    store.upsert_chunks([chunk])

    hits = store.search("LangGraph agent patterns", limit=5, filters={"tags": "NoSuchTag"})
    assert hits == []


def test_keyword_store_tags_filter_is_case_insensitive(tmp_path) -> None:
    store = KeywordStore(tmp_path / "fts.sqlite")
    store.initialize()

    chunk = _chunk("c1", "LangGraph agent patterns")
    chunk.metadata["tags"] = ["langgraph", "ai-agents"]
    store.upsert_chunks([chunk])

    hits = store.search("LangGraph agent patterns", limit=5, filters={"tags": "LangGraph"})
    assert [h.chunk_id for h in hits] == ["c1"]


def test_matches_filters_rejects_non_dict_date_range() -> None:
    metadata = {"derived_fields": {"due_date": "2026-02-24"}}
    with pytest.raises(ValueError):
        matches_filters(metadata, {"date_range": "2026-02-24"})


def test_matches_filters_rejects_non_dict_frontmatter_contains() -> None:
    metadata = {"raw_frontmatter": {"status": "done"}}
    with pytest.raises(ValueError):
        matches_filters(metadata, {"frontmatter_contains": "done"})


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


def test_keyword_store_date_range_matches_plain_date_key(tmp_path) -> None:
    store = KeywordStore(tmp_path / "fts.sqlite")
    store.initialize()

    store.upsert_chunks(
        [
            ChunkRecord(
                chunk_id="c1",
                note_id="note-1",
                text="daily journal wins",
                metadata={
                    "path": "Daily/Journal/2026-06-16.md",
                    "tags": [],
                    "raw_frontmatter": {"date": "2026-06-16"},
                    "derived_fields": {"date_date": "2026-06-16"},
                },
                bm25_text="daily journal wins",
            )
        ]
    )

    hits = store.search(
        "daily journal wins",
        limit=5,
        filters={"date_range": {"start": "2026-06-16", "end": "2026-06-16"}},
    )
    assert [h.chunk_id for h in hits] == ["c1"]


def test_keyword_store_search_sanitizes_fts5_special_characters(tmp_path) -> None:
    store = KeywordStore(tmp_path / "fts.sqlite")
    store.initialize()

    store.upsert_chunks([_chunk("c1", "finish quarterly planning today")])

    hits = store.search("quarterly planning?", limit=5)
    assert [h.chunk_id for h in hits] == ["c1"]

    hits = store.search('"quarterly" (planning):', limit=5)
    assert [h.chunk_id for h in hits] == ["c1"]

    assert store.search("???", limit=5) == []


def test_backlinks_for_title_finds_linking_note(tmp_path) -> None:
    store = KeywordStore(tmp_path / "fts.sqlite")
    store.initialize()

    store.upsert_chunks(
        [
            _chunk("a1", "note A references note B", path="A.md", links=["Note B"]),
            _chunk("b1", "note B has no outlinks", path="B.md"),
        ]
    )

    assert store.backlinks_for_title("Note B") == ["A.md"]


def test_backlinks_for_title_matches_case_insensitively_and_strips_heading(tmp_path) -> None:
    store = KeywordStore(tmp_path / "fts.sqlite")
    store.initialize()

    store.upsert_chunks(
        [
            _chunk("a1", "note A links to a heading in B", path="A.md", links=["note b#Status"]),
            _chunk("b1", "note B", path="B.md"),
        ]
    )

    assert store.backlinks_for_title("Note B") == ["A.md"]


def test_backlinks_for_title_returns_empty_when_no_match(tmp_path) -> None:
    store = KeywordStore(tmp_path / "fts.sqlite")
    store.initialize()

    store.upsert_chunks([_chunk("a1", "note A links elsewhere", path="A.md", links=["Note C"])])

    assert store.backlinks_for_title("Note B") == []


def test_backlinks_for_title_excludes_self_link(tmp_path) -> None:
    store = KeywordStore(tmp_path / "fts.sqlite")
    store.initialize()

    store.upsert_chunks([_chunk("a1", "note A links to itself", path="A.md", links=["Note A"])])

    assert store.backlinks_for_title("Note A", exclude_path="A.md") == []
