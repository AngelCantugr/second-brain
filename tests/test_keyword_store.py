import pytest

from second_brain.keyword_store import KeywordStore, matches_filters
from second_brain.models import ChunkRecord


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


@pytest.mark.parametrize(
    ("status", "excluded", "expected"),
    [("superseded", ["superseded"], False), ("active", ["superseded"], True), (None, ["superseded"], True)],
)
def test_matches_filters_excludes_frontmatter_status(
    status: str | None, excluded: list[str], expected: bool
) -> None:
    metadata = {"raw_frontmatter": ({"status": status} if status is not None else {})}

    assert matches_filters(metadata, {"exclude_status": excluded}) is expected


def test_matches_filters_rejects_malformed_exclude_status() -> None:
    with pytest.raises(ValueError, match="exclude_status"):
        matches_filters({}, {"exclude_status": "superseded"})


@pytest.mark.parametrize(
    ("modified_since", "mtime", "matches"),
    [
        ("2026-10-01", 1790812800.0, True),
        ("2026-10-01T00:00:00Z", 1790812800.0, True),
        ("2026-10-01T00:00:00", 1790812800.0, True),
        ("2026-10-01T00:00:01+00:00", 1790812800.0, False),
        ("2026-10-01T03:00:00+03:00", 1790812800.0, True),
        ("2026-10-01T00:00:31+00:00:30", 1790812800.0, False),
        ("2026-10-01T00:00:31+00:00:30.5", 1790812800.0, False),
    ],
)
def test_modified_since_matches_note_mtime_inclusively(
    modified_since: str, mtime: float, matches: bool
) -> None:
    assert matches_filters({"mtime": mtime}, {"modified_since": modified_since}) is matches


def test_modified_since_composes_with_other_metadata_filters() -> None:
    metadata = {"mtime": 1790812800.0, "path": "Projects/Issue35/note.md", "tags": ["RAG"]}

    assert matches_filters(
        metadata,
        {
            "modified_since": "2026-10-01",
            "tags": ["rag"],
            "path_prefix": "Projects/Issue35/",
        },
    )
    assert not matches_filters(
        metadata,
        {"modified_since": "2026-10-01", "path_prefix": "Archive/"},
    )


def test_keyword_search_composes_modified_since_with_date_range_tags_and_path(tmp_path) -> None:
    store = KeywordStore(tmp_path / "fts.sqlite")
    store.initialize()
    old = _chunk("old", "quarterly planning", path="Projects/Issue35/old.md")
    old.metadata.update(mtime=1790726400.0, tags=["RAG"])
    fresh = _chunk("fresh", "quarterly planning", path="Projects/Issue35/fresh.md")
    fresh.metadata.update(mtime=1790812800.0, tags=["RAG"])
    fresh.metadata["derived_fields"]["due_date"] = "2026-10-01"
    store.upsert_chunks([old, fresh])

    hits = store.search(
        "quarterly planning",
        filters={
            "modified_since": "2026-10-01",
            "date_range": {"start": "2026-10-01", "end": "2026-10-01"},
            "tags": ["rag"],
            "path_prefix": "Projects/Issue35/",
        },
    )

    assert [hit.chunk_id for hit in hits] == ["fresh"]


@pytest.mark.parametrize("modified_since", [
    "not-a-date", "2026-13-01", "2026-10-01T25:00:00Z", None, 5,
    "2026-10-01T00:00:00+00:99",
    "2026-10-01T00:00:00+00:00:99",
    "2026-10-01T00:00:00+24:00",
])
def test_matches_filters_rejects_invalid_modified_since(modified_since) -> None:
    with pytest.raises(ValueError, match="modified_since"):
        matches_filters({}, {"modified_since": modified_since})


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


def test_keyword_search_filters_before_candidate_limit(tmp_path) -> None:
    store = KeywordStore(tmp_path / "fts.sqlite")
    store.initialize()
    chunks = [_chunk(f"near-{index}", "gym exercise check-in") for index in range(12)]
    target = _chunk("filtered-target", "gym exercise check-in")
    for chunk in chunks:
        chunk.metadata["tags"] = ["other"]
    target.metadata["tags"] = ["wanted"]
    store.upsert_chunks([*chunks, target])

    hits = store.search("gym exercise check-in", limit=1, filters={"tags": ["wanted"]})

    assert [hit.chunk_id for hit in hits] == ["filtered-target"]


def test_keyword_search_unbounded_limit_returns_all_matching_candidates(tmp_path) -> None:
    store = KeywordStore(tmp_path / "fts.sqlite")
    store.initialize()
    chunks = [_chunk(f"candidate-{index}", "matching search candidates") for index in range(37)]
    store.upsert_chunks(chunks)

    hits = store.search("matching search", limit=None)

    assert len(hits) == 37


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
