import json
import sqlite3

import pytest

from second_brain.keyword_store import KeywordStore, build_fts_match_query, matches_filters
from second_brain.models import ChunkRecord


def _chunk(
    chunk_id: str,
    text: str,
    status: str = "todo",
    path: str = "Project.md",
    links: list[str] | None = None,
    note_title: str | None = None,
    heading_path: str = "",
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
        heading_path=heading_path,
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


def test_build_fts_match_query_drops_stopwords_and_ors_remaining_terms() -> None:
    assert build_fts_match_query("What are my notes on system design?") == '"notes" OR "system" OR "design"'


def test_build_fts_match_query_falls_back_to_all_tokens_when_only_stopwords() -> None:
    assert build_fts_match_query("what is it") == '"what" OR "is" OR "it"'


def test_build_fts_match_query_returns_empty_for_punctuation_only() -> None:
    assert build_fts_match_query("???") == ""


def test_keyword_search_natural_language_question_matches_partial_terms(tmp_path) -> None:
    store = KeywordStore(tmp_path / "fts.sqlite")
    store.initialize()
    store.upsert_chunks([_chunk("partial", "my notes on design patterns")])

    hits = store.search("What are my notes on system design?", limit=5)

    assert [h.chunk_id for h in hits] == ["partial"]


def test_keyword_search_matches_note_title_when_body_never_says_it(tmp_path) -> None:
    store = KeywordStore(tmp_path / "fts.sqlite")
    store.initialize()
    store.upsert_chunks([_chunk("kafka-note", "partitions and consumer groups", note_title="Kafka")])

    hits = store.search("kafka", limit=5)

    assert [h.chunk_id for h in hits] == ["kafka-note"]


def test_keyword_search_ranks_title_and_heading_matches_above_body_mention(tmp_path) -> None:
    store = KeywordStore(tmp_path / "fts.sqlite")
    store.initialize()
    shared_text = "kafka mentioned in passing"
    store.upsert_chunks(
        [
            _chunk("body-only", shared_text, note_title="Journal"),
            _chunk("heading-match", shared_text, note_title="Journal", heading_path="Kafka"),
            _chunk("title-match", shared_text, note_title="Kafka"),
        ]
    )

    hits = store.search("kafka", limit=5)

    assert [h.chunk_id for h in hits] == ["title-match", "heading-match", "body-only"]


def _write_pre_title_schema_db(db_path) -> None:
    """Write the FTS layout that shipped before title/heading columns existed."""

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            note_id TEXT NOT NULL,
            text TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, text)")
    rows = [
        ("old-kafka", "event streaming notes", {"note_title": "Kafka", "heading_path": "Partitions"}),
        ("old-groceries", "milk eggs bread", {"note_title": "Groceries", "heading_path": "root"}),
    ]
    for chunk_id, text, metadata in rows:
        conn.execute(
            "INSERT INTO chunks(chunk_id, note_id, text, metadata_json) VALUES(?, ?, ?, ?)",
            (chunk_id, "note-1", text, json.dumps(metadata)),
        )
        conn.execute("INSERT INTO chunks_fts(chunk_id, text) VALUES(?, ?)", (chunk_id, text))
    conn.commit()
    conn.close()


def test_initialize_migrates_pre_title_fts_table_and_is_idempotent(tmp_path) -> None:
    db_path = tmp_path / "fts.sqlite"
    _write_pre_title_schema_db(db_path)
    store = KeywordStore(db_path)

    store.initialize()

    conn = sqlite3.connect(db_path)
    columns = [row[1] for row in conn.execute("PRAGMA table_info(chunks_fts)")]
    assert columns == ["chunk_id", "text", "title", "heading"]
    assert conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 2
    # Mark the migrated row so a second initialize() that rebuilds the table would overwrite it.
    conn.execute("UPDATE chunks_fts SET title = 'sentinel' WHERE chunk_id = 'old-kafka'")
    conn.commit()
    conn.close()

    assert [h.chunk_id for h in store.search("kafka", limit=5)] == []
    assert [h.chunk_id for h in store.search("partitions", limit=5)] == ["old-kafka"]

    store.initialize()

    conn = sqlite3.connect(db_path)
    assert conn.execute(
        "SELECT title FROM chunks_fts WHERE chunk_id = 'old-kafka'"
    ).fetchone()[0] == "sentinel"
    assert conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == 2
    conn.close()


def test_initialize_migration_skips_corrupt_metadata_row(tmp_path) -> None:
    db_path = tmp_path / "fts.sqlite"
    _write_pre_title_schema_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO chunks(chunk_id, note_id, text, metadata_json) VALUES(?, ?, ?, ?)",
        ("corrupt", "note-2", "orphan text", "{not valid json"),
    )
    conn.execute("INSERT INTO chunks_fts(chunk_id, text) VALUES(?, ?)", ("corrupt", "orphan text"))
    conn.commit()
    conn.close()

    KeywordStore(db_path).initialize()

    conn = sqlite3.connect(db_path)
    assert [r[1] for r in conn.execute("PRAGMA table_info(chunks_fts)")] == [
        "chunk_id", "text", "title", "heading",
    ]
    assert sorted(r[0] for r in conn.execute("SELECT chunk_id FROM chunks_fts")) == [
        "corrupt", "old-groceries", "old-kafka",
    ]
    conn.close()
    store = KeywordStore(db_path)
    assert [h.chunk_id for h in store.search("kafka", limit=5)] == ["old-kafka"]


def test_upsert_stores_empty_heading_for_root_and_root_is_not_searchable(tmp_path) -> None:
    store = KeywordStore(tmp_path / "fts.sqlite")
    store.initialize()
    store.upsert_chunks([_chunk("top", "intro paragraph before any heading", heading_path="root")])

    conn = sqlite3.connect(tmp_path / "fts.sqlite")
    heading = conn.execute("SELECT heading FROM chunks_fts WHERE chunk_id = 'top'").fetchone()[0]
    conn.close()
    assert heading == ""
    assert store.search("root", limit=5) == []


def test_build_fts_match_query_drops_single_character_tokens() -> None:
    assert build_fts_match_query("don't panic") == '"don" OR "panic"'


def test_build_fts_match_query_keeps_single_character_tokens_when_nothing_else_remains() -> None:
    assert build_fts_match_query("x y") == '"x" OR "y"'


def test_filtered_search_pulls_rows_lazily_and_stops_at_limit(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = KeywordStore(tmp_path / "fts.sqlite")
    store.initialize()
    store.upsert_chunks([_chunk(f"c{i}", "shared keyword text") for i in range(20)])

    pulled = 0
    real_connect = store._connect

    class CountingCursor:
        def __init__(self, cursor: sqlite3.Cursor) -> None:
            self._cursor = cursor

        def __iter__(self):
            nonlocal pulled
            for row in self._cursor:
                pulled += 1
                yield row

        def fetchall(self):
            nonlocal pulled
            rows = self._cursor.fetchall()
            pulled += len(rows)
            return rows

    class CountingConnection:
        def __init__(self, conn: sqlite3.Connection) -> None:
            self._conn = conn

        def __enter__(self):
            self._conn.__enter__()
            return self

        def __exit__(self, *exc_info):
            return self._conn.__exit__(*exc_info)

        def execute(self, *args, **kwargs):
            return CountingCursor(self._conn.execute(*args, **kwargs))

    monkeypatch.setattr(store, "_connect", lambda: CountingConnection(real_connect()))

    hits = store.search("keyword", limit=2, filters={"status": "todo"})

    assert len(hits) == 2
    assert pulled == 2


def _write_pre_path_schema_db(db_path) -> None:
    """Seed the layout before ``chunks.path``, with FTS rowids unrelated to chunks rowids."""

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            note_id TEXT NOT NULL,
            text TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, text, title, heading)"
    )
    rows = [
        ("kafka-1", "event streaming notes", "Kafka.md"),
        ("kafka-2", "consumer group rebalancing", "Kafka.md"),
        ("groceries-1", "milk eggs bread", "Groceries.md"),
    ]
    for chunk_id, text, path in rows:
        conn.execute(
            "INSERT INTO chunks(chunk_id, note_id, text, metadata_json) VALUES(?, ?, ?, ?)",
            (chunk_id, "note-1", text, json.dumps({"path": path, "note_title": path[:-3]})),
        )
    # Old upserts deleted and re-inserted FTS rows, so their rowids drifted away
    # from the chunks rowids; reversed insertion reproduces that mismatch.
    for chunk_id, text, path in reversed(rows):
        conn.execute(
            "INSERT INTO chunks_fts(chunk_id, text, title, heading) VALUES(?, ?, ?, '')",
            (chunk_id, text, path[:-3]),
        )
    conn.commit()
    conn.close()


def _fts_rows_match_chunks(db_path) -> bool:
    """Every chunk points at exactly one FTS row for itself, and no FTS row is orphaned."""

    conn = sqlite3.connect(db_path)
    try:
        pointed = conn.execute(
            "SELECT c.chunk_id, f.chunk_id FROM chunks c "
            "LEFT JOIN chunks_fts f ON f.rowid = c.fts_rowid"
        ).fetchall()
        fts_count = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    finally:
        conn.close()
    return all(own == fts for own, fts in pointed) and fts_count == len(pointed)


def test_initialize_migrates_pre_path_schema_and_links_fts_rows(tmp_path) -> None:
    db_path = tmp_path / "fts.sqlite"
    _write_pre_path_schema_db(db_path)
    store = KeywordStore(db_path)

    store.initialize()
    store.initialize()

    assert _fts_rows_match_chunks(db_path)
    assert store.chunk_ids_by_path("Kafka.md") == ["kafka-1", "kafka-2"]
    assert [hit.chunk_id for hit in store.chunks_by_path("Groceries.md")] == ["groceries-1"]
    assert [hit.chunk_id for hit in store.search("rebalancing", limit=5)] == ["kafka-2"]

    store.upsert_chunks([_chunk("kafka-2", "partition leader election", path="Kafka.md")])
    store.delete_chunks(["groceries-1"])

    assert _fts_rows_match_chunks(db_path)
    assert store.search("rebalancing", limit=5) == []
    assert [hit.chunk_id for hit in store.search("election", limit=5)] == ["kafka-2"]
    assert store.search("milk", limit=5) == []


def test_upsert_replaces_a_chunks_fts_row_instead_of_adding_one(tmp_path) -> None:
    db_path = tmp_path / "fts.sqlite"
    store = KeywordStore(db_path)
    store.initialize()

    store.upsert_chunks([_chunk("c1", "first draft"), _chunk("c2", "other chunk")])
    store.upsert_chunks([_chunk("c1", "second draft", path="Moved.md")])

    assert _fts_rows_match_chunks(db_path)
    assert store.search("first", limit=5) == []
    assert [hit.chunk_id for hit in store.search("second", limit=5)] == ["c1"]
    assert store.chunk_ids_by_path("Project.md") == ["c2"]
    assert store.chunk_ids_by_path("Moved.md") == ["c1"]


def test_chunk_ids_by_path_is_an_indexed_lookup(tmp_path) -> None:
    db_path = tmp_path / "fts.sqlite"
    KeywordStore(db_path).initialize()

    conn = sqlite3.connect(db_path)
    plan = " ".join(
        row[-1]
        for row in conn.execute(
            "EXPLAIN QUERY PLAN SELECT chunk_id FROM chunks WHERE path = ? ORDER BY rowid",
            ("a.md",),
        )
    )
    conn.close()
    assert "idx_chunks_path" in plan


def test_fts_rows_stay_linked_after_vacuum_renumbers_chunk_rowids(tmp_path) -> None:
    db_path = tmp_path / "fts.sqlite"
    store = KeywordStore(db_path)
    store.initialize()
    store.upsert_chunks([_chunk(f"c{i}", f"word{i} body") for i in range(5)])
    store.delete_chunks(["c0", "c2"])
    conn = sqlite3.connect(db_path)
    conn.execute("VACUUM")
    conn.close()

    store.upsert_chunks([_chunk("c4", "replacement body")])
    store.delete_chunks(["c3"])

    assert _fts_rows_match_chunks(db_path)
    assert store.search("word4", limit=5) == []
    assert [hit.chunk_id for hit in store.search("replacement", limit=5)] == ["c4"]
    assert [hit.chunk_id for hit in store.search("word1", limit=5)] == ["c1"]
