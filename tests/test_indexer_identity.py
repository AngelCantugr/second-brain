"""Real-store regressions for path ownership and retryable legacy migration."""

from dataclasses import replace
from pathlib import Path
import sqlite3
import uuid

import pytest

from second_brain.chunker import chunk_note
from second_brain.config import RagConfig
from second_brain.graph import GraphStore
from second_brain.indexer import Indexer
from second_brain.keyword_store import KeywordStore
from second_brain.parser import parse_note
from second_brain.sync_state import SyncStateStore
from second_brain.vector_store import QdrantVectorStore


class _FixtureEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, float(len(text))] for text in texts]


@pytest.fixture
def real_indexer(tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    config = RagConfig(
        vault_path=vault,
        qdrant_path=tmp_path / "qdrant",
        fts_path=tmp_path / "fts.sqlite",
        sync_state_path=tmp_path / "sync.sqlite",
        chunk_size=20,
        chunk_overlap=0,
    )
    vectors = QdrantVectorStore(config.qdrant_path, "chunks")
    indexer = Indexer(
        config, _FixtureEmbedder(), vectors, KeywordStore(config.fts_path),
        SyncStateStore(config.sync_state_path), GraphStore(config.fts_path),
    )
    indexer.initialize()
    try:
        yield indexer
    finally:
        vectors.client.close()


def _assert_note(indexer: Indexer, path: str, text: str) -> set[str]:
    vectors = indexer.vector_store.get_by_path(path)
    keywords = indexer.keyword_store.chunks_by_path(path)
    assert [hit.text for hit in vectors] == [text]
    assert [hit.text for hit in keywords] == [text]
    ids = {hit.chunk_id for hit in vectors}
    assert ids == {hit.chunk_id for hit in keywords}
    assert indexer.graph_store.note_meta_for(path) is not None
    return ids


def _legacy_population(indexer: Indexer, paths: list[str]) -> set[str]:
    """Seed the old content-only schema/IDs, including future runtime metadata.

    This deliberately reconstructs the old identity contract independently
    of the new identity code, so changing that code cannot mask legacy data.
    """
    db_path = indexer.config.sync_state_path
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TABLE note_state")
        conn.execute(
            "CREATE TABLE note_state (path TEXT PRIMARY KEY, content_hash TEXT NOT NULL, "
            "mtime REAL NOT NULL, updated_at REAL DEFAULT (strftime('%s', 'now')))"
        )
        conn.execute(
            "CREATE TABLE runtime_metadata (key TEXT PRIMARY KEY, value REAL NOT NULL)"
        )
        conn.execute(
            "INSERT INTO runtime_metadata VALUES ('watcher_last_event', 123.5)"
        )
        legacy_ids = set()
        for path in paths:
            note = parse_note(indexer.config.vault_path / path, indexer.config.vault_path)
            chunks = chunk_note(note, 20, 0)
            # Fixture notes have one section/window, so the historical index is 0.
            legacy_chunks = [
                replace(
                    chunk, note_id=note.content_hash[:16],
                    chunk_id=str(uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{note.content_hash[:16]}:{chunk.heading_path}:0:{chunk.text}",
                    )),
                )
                for chunk in chunks
            ]
            indexer.vector_store.ensure_collection(2)
            indexer.vector_store.upsert_chunks(
                legacy_chunks, indexer.embedder.embed([chunk.text for chunk in legacy_chunks])
            )
            indexer.keyword_store.upsert_chunks(legacy_chunks)
            indexer.graph_store.upsert_note_meta(path, note.title, note.tags, note.links, [1.0, 2.0])
            conn.execute(
                "INSERT INTO note_state(path, content_hash, mtime) VALUES (?, ?, ?)",
                (path, note.content_hash, note.mtime),
            )
            legacy_ids.update(chunk.chunk_id for chunk in legacy_chunks)
        state_before = conn.execute("SELECT * FROM note_state ORDER BY path").fetchall()
    indexer.sync_state.initialize()
    indexer.sync_state.initialize()
    with sqlite3.connect(db_path) as conn:
        assert conn.execute(
            "SELECT path, content_hash, mtime, updated_at FROM note_state ORDER BY path"
        ).fetchall() == state_before
    return legacy_ids


def test_identical_notes_have_independent_chunks_and_note_ids(real_indexer: Indexer) -> None:
    indexer = real_indexer
    for path in ["a.md", "b.md"]:
        (indexer.config.vault_path / path).write_text("shared content marker")
    result = indexer.sync()
    assert result.errors == []
    a_ids = _assert_note(indexer, "a.md", "shared content marker")
    b_ids = _assert_note(indexer, "b.md", "shared content marker")
    assert a_ids.isdisjoint(b_ids)
    with sqlite3.connect(indexer.config.fts_path) as conn:
        assert conn.execute("SELECT COUNT(DISTINCT note_id) FROM chunks").fetchone()[0] == 2
    assert indexer.keyword_store.count_chunks() == 2


def test_converging_then_diverging_notes_preserve_unchanged_owner(real_indexer: Indexer) -> None:
    indexer = real_indexer
    vault = indexer.config.vault_path
    (vault / "a.md").write_text("original a content")
    (vault / "b.md").write_text("shared content marker")
    assert indexer.sync().errors == []
    b_ids = _assert_note(indexer, "b.md", "shared content marker")
    b_meta = indexer.graph_store.note_meta_for("b.md")
    for content in ["shared content marker", "new a content"]:
        (vault / "a.md").write_text(content)
        result = indexer.sync()
        assert (result.processed, result.skipped, result.errors) == (1, 1, [])
        assert _assert_note(indexer, "b.md", "shared content marker") == b_ids
        _assert_note(indexer, "a.md", content)
        assert indexer.graph_store.note_meta_for("b.md") == b_meta


def test_removing_one_duplicate_preserves_surviving_note(real_indexer: Indexer) -> None:
    indexer = real_indexer
    vault = indexer.config.vault_path
    for path in ["a.md", "b.md"]:
        (vault / path).write_text("shared content marker")
    assert indexer.sync().errors == []
    b_ids = {hit.chunk_id for hit in indexer.vector_store.get_by_path("b.md")}
    b_meta = indexer.graph_store.note_meta_for("b.md")
    (vault / "a.md").unlink()
    result = indexer.sync()
    assert (result.deleted, result.errors) == (1, [])
    assert _assert_note(indexer, "b.md", "shared content marker") == b_ids
    assert indexer.graph_store.note_meta_for("b.md") == b_meta
    assert indexer.vector_store.get_by_path("a.md") == []
    assert indexer.keyword_store.chunk_ids_by_path("a.md") == []
    assert indexer.sync_state.tracked_paths() == ["b.md"]


def test_unchanged_legacy_duplicates_migrate_without_resetting_metadata(real_indexer: Indexer) -> None:
    indexer = real_indexer
    for path in ["a.md", "b.md"]:
        (indexer.config.vault_path / path).write_text("shared content marker")
    legacy_ids = _legacy_population(indexer, ["a.md", "b.md"])
    result = indexer.sync()
    assert (result.processed, result.skipped, result.errors) == (2, 0, [])
    a_ids = _assert_note(indexer, "a.md", "shared content marker")
    b_ids = _assert_note(indexer, "b.md", "shared content marker")
    assert a_ids.isdisjoint(b_ids)
    assert legacy_ids.isdisjoint(a_ids | b_ids)
    assert indexer.vector_store.client.retrieve("chunks", list(legacy_ids)) == []
    assert indexer.keyword_store.count_chunks() == 2
    assert indexer.sync().skipped == 2
    with sqlite3.connect(indexer.config.sync_state_path) as conn:
        assert conn.execute("SELECT value FROM runtime_metadata WHERE key='watcher_last_event'").fetchone() == (123.5,)
        rows = conn.execute("SELECT path, content_hash FROM note_state ORDER BY path").fetchall()
    assert rows == [(path, parse_note(indexer.config.vault_path / path).content_hash) for path in ["a.md", "b.md"]]


@pytest.mark.parametrize("failure", ["embed", "vector_delete", "keyword_delete", "vector_upsert", "keyword_upsert", "metadata"])
@pytest.mark.parametrize("legacy", [False, True], ids=["edit", "legacy"])
@pytest.mark.parametrize("failure_timing", ["before", "after"])
def test_failure_retry_converges_both_stores_without_marking_current(
    real_indexer: Indexer, monkeypatch: pytest.MonkeyPatch, failure: str, legacy: bool,
    failure_timing: str,
) -> None:
    indexer = real_indexer
    path = indexer.config.vault_path / "a.md"
    path.write_text("old content marker")
    if legacy:
        old_ids = _legacy_population(indexer, ["a.md"])
    else:
        assert indexer.sync().errors == []
        old_ids = _assert_note(indexer, "a.md", "old content marker")
        path.write_text("new content marker")
    with sqlite3.connect(indexer.config.sync_state_path) as conn:
        state_before = conn.execute("SELECT * FROM note_state").fetchall()
    current_hash = parse_note(path).content_hash
    target, method = {
        "embed": (indexer.embedder, "embed"),
        "vector_delete": (indexer.vector_store, "delete_chunks"),
        "keyword_delete": (indexer.keyword_store, "delete_chunks"),
        "vector_upsert": (indexer.vector_store, "upsert_chunks"),
        "keyword_upsert": (indexer.keyword_store, "upsert_chunks"),
        "metadata": (indexer.graph_store, "upsert_note_meta"),
    }[failure]
    original = getattr(target, method)

    def fail(*args, **kwargs):
        if failure_timing == "after":
            original(*args, **kwargs)
        raise RuntimeError("injected indexing failure")

    with monkeypatch.context() as patch:
        patch.setattr(target, method, fail)
        failed = indexer.sync()
        assert failed.processed == 0
        assert len(failed.errors) == 1
        assert "injected indexing failure" in failed.errors[0]
    with sqlite3.connect(indexer.config.sync_state_path) as conn:
        assert conn.execute("SELECT * FROM note_state").fetchall() == state_before
    assert indexer.sync_state.should_reindex("a.md", current_hash)
    if failure == "embed":
        assert indexer.sync_state.pending_paths() == []
    else:
        assert indexer.sync_state.pending_paths() == ["a.md"]
    if failure == "embed":
        assert {hit.chunk_id for hit in indexer.vector_store.get_by_path("a.md")} == old_ids
        assert set(indexer.keyword_store.chunk_ids_by_path("a.md")) == old_ids
    retry = indexer.sync()
    assert (retry.processed, retry.errors) == (1, [])
    new_ids = _assert_note(indexer, "a.md", "old content marker" if legacy else "new content marker")
    assert old_ids.isdisjoint(new_ids)
    assert indexer.vector_store.client.retrieve("chunks", list(old_ids)) == []
    assert indexer.keyword_store.count_chunks() == 1
    assert not indexer.sync_state.should_reindex("a.md", current_hash)
    assert indexer.sync_state.pending_paths() == []
    assert indexer.sync().skipped == 1


def test_checkpoint_failure_keeps_pending_marker_until_retry(
    real_indexer: Indexer, monkeypatch: pytest.MonkeyPatch,
) -> None:
    indexer = real_indexer
    path = indexer.config.vault_path / "a.md"
    path.write_text("old checkpoint content")
    assert indexer.sync().errors == []
    with sqlite3.connect(indexer.config.sync_state_path) as conn:
        checkpoint_before = conn.execute("SELECT * FROM note_state").fetchall()
    path.write_text("new indexed content")

    def fail_checkpoint(*args, **kwargs) -> None:
        raise RuntimeError("checkpoint unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(indexer.sync_state, "record_note", fail_checkpoint)
        failed = indexer.sync()
    assert failed.processed == 0
    assert len(failed.errors) == 1
    assert "checkpoint unavailable" in failed.errors[0]
    assert indexer.sync_state.pending_paths() == ["a.md"]
    with sqlite3.connect(indexer.config.sync_state_path) as conn:
        assert conn.execute("SELECT * FROM note_state").fetchall() == checkpoint_before

    restarted_state = SyncStateStore(indexer.config.sync_state_path)
    restarted_state.initialize()
    indexer.sync_state = restarted_state
    retry = indexer.sync()

    assert (retry.processed, retry.skipped, retry.errors) == (1, 0, [])
    assert [hit.text for hit in indexer.vector_store.get_by_path("a.md")] == ["new indexed content"]
    assert [hit.text for hit in indexer.keyword_store.chunks_by_path("a.md")] == ["new indexed content"]
    assert restarted_state.pending_paths() == []


def test_file_sync_migrates_only_selected_legacy_note(real_indexer: Indexer) -> None:
    indexer = real_indexer
    for path in ["a.md", "b.md"]:
        (indexer.config.vault_path / path).write_text("shared content marker")
    legacy_ids = _legacy_population(indexer, ["a.md", "b.md"])
    result = indexer.sync(mode="file", file_path="a.md")
    assert (result.processed, result.errors) == (1, [])
    a_ids = _assert_note(indexer, "a.md", "shared content marker")
    assert indexer.sync_state.should_reindex(
        "b.md", parse_note(indexer.config.vault_path / "b.md").content_hash
    )
    _assert_note(indexer, "b.md", "shared content marker")
    result = indexer.sync(mode="file", file_path="b.md")
    assert (result.processed, result.errors) == (1, [])
    assert _assert_note(indexer, "a.md", "shared content marker") == a_ids
    assert indexer.vector_store.client.retrieve("chunks", list(legacy_ids)) == []
    assert indexer.sync().skipped == 2


def test_vector_only_legacy_leftover_is_cleaned_on_retry(real_indexer: Indexer) -> None:
    indexer = real_indexer
    (indexer.config.vault_path / "a.md").write_text("old content marker")
    legacy_ids = _legacy_population(indexer, ["a.md"])
    indexer.keyword_store.delete_chunks(list(legacy_ids))
    assert indexer.sync().errors == []
    assert indexer.vector_store.client.retrieve("chunks", list(legacy_ids)) == []
    _assert_note(indexer, "a.md", "old content marker")


def test_keyword_only_legacy_note_recovers_without_vector_collection(real_indexer: Indexer) -> None:
    indexer = real_indexer
    (indexer.config.vault_path / "a.md").write_text("old content marker")
    legacy_ids = _legacy_population(indexer, ["a.md"])
    indexer.vector_store.client.delete_collection("chunks")
    result = indexer.sync()
    assert (result.processed, result.errors) == (1, [])
    assert indexer.vector_store.client.retrieve("chunks", list(legacy_ids)) == []
    _assert_note(indexer, "a.md", "old content marker")


def test_caller_built_notes_with_shared_note_id_still_have_path_owned_chunks(
    real_indexer: Indexer,
) -> None:
    path = real_indexer.config.vault_path / "a.md"
    path.write_text("content marker")
    a = parse_note(path, real_indexer.config.vault_path)
    b = replace(a, path="b.md")
    assert chunk_note(a, 20, 0)[0].chunk_id != chunk_note(b, 20, 0)[0].chunk_id


@pytest.mark.parametrize("failure", ["vector_delete", "keyword_delete"])
@pytest.mark.parametrize("failure_timing", ["before", "after"])
def test_removed_note_cleanup_retries_after_partial_failure(
    real_indexer: Indexer, monkeypatch: pytest.MonkeyPatch, failure: str,
    failure_timing: str,
) -> None:
    indexer = real_indexer
    path = indexer.config.vault_path / "a.md"
    path.write_text("removed content marker")
    assert indexer.sync().errors == []
    path.unlink()
    target = indexer.vector_store if failure == "vector_delete" else indexer.keyword_store
    original = target.delete_chunks

    def fail(*args, **kwargs):
        if failure_timing == "after":
            original(*args, **kwargs)
        raise RuntimeError("injected deletion failure")

    with monkeypatch.context() as patch:
        patch.setattr(target, "delete_chunks", fail)
        failed = indexer.sync()
        assert failed.deleted == 0
        assert len(failed.errors) == 1
    assert indexer.sync_state.tracked_paths() == ["a.md"]
    retry = indexer.sync()
    assert (retry.deleted, retry.errors) == (1, [])
    assert indexer.vector_store.get_by_path("a.md") == []
    assert indexer.keyword_store.count_chunks() == 0
    assert indexer.sync_state.tracked_paths() == []


def test_first_empty_note_needs_no_vector_collection(real_indexer: Indexer) -> None:
    indexer = real_indexer
    (indexer.config.vault_path / "a.md").write_text("---\ntags: [project]\n---\n")
    assert indexer.sync().errors == []
    assert indexer.vector_store.get_by_path("a.md") == []
    meta = indexer.graph_store.note_meta_for("a.md")
    assert meta["tags"] == ["project"]
    assert meta["centroid"] is None


def test_empty_transition_keeps_tags_and_clears_centroid(real_indexer: Indexer) -> None:
    indexer = real_indexer
    path = indexer.config.vault_path / "a.md"
    path.write_text("---\ntags: [project]\n---\ncontent marker")
    assert indexer.sync().errors == []
    assert indexer.graph_store.note_meta_for("a.md")["centroid"] is not None
    path.write_text("---\ntags: [project]\n---\n")
    assert indexer.sync().errors == []
    assert indexer.vector_store.get_by_path("a.md") == []
    assert indexer.keyword_store.count_chunks() == 0
    meta = indexer.graph_store.note_meta_for("a.md")
    assert meta["tags"] == ["project"]
    assert meta["centroid"] is None
    assert meta["dim"] is None


def test_pending_replacement_survives_revert_and_sync_state_restart(
    real_indexer: Indexer, monkeypatch: pytest.MonkeyPatch,
) -> None:
    indexer = real_indexer
    path = indexer.config.vault_path / "a.md"
    original_text = "original uniqueold"
    edited_text = "edited uniquenew"
    path.write_text(original_text)
    assert indexer.sync().errors == []
    with sqlite3.connect(indexer.config.sync_state_path) as conn:
        checkpoint_before = conn.execute("SELECT * FROM note_state").fetchall()

    path.write_text(edited_text)

    def fail_keyword_write(chunks) -> None:
        raise RuntimeError("keyword write unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(indexer.keyword_store, "upsert_chunks", fail_keyword_write)
        failed = indexer.sync()
    assert failed.processed == 0
    assert len(failed.errors) == 1
    assert "keyword write unavailable" in failed.errors[0]
    assert [hit.text for hit in indexer.vector_store.get_by_path("a.md")] == [edited_text]
    assert indexer.keyword_store.chunks_by_path("a.md") == []
    with sqlite3.connect(indexer.config.sync_state_path) as conn:
        assert conn.execute("SELECT * FROM note_state").fetchall() == checkpoint_before

    path.write_text(original_text)
    restarted_state = SyncStateStore(indexer.config.sync_state_path)
    restarted_state.initialize()
    indexer.sync_state = restarted_state
    retry = indexer.sync()

    assert (retry.processed, retry.skipped, retry.errors) == (1, 0, [])
    assert [hit.text for hit in indexer.vector_store.get_by_path("a.md")] == [original_text]
    assert [hit.text for hit in indexer.keyword_store.chunks_by_path("a.md")] == [original_text]
    assert restarted_state.pending_paths() == []
    assert restarted_state.should_reindex("a.md", parse_note(path).content_hash) is False
    assert indexer.sync().skipped == 1


def test_deleted_pending_only_path_is_cleaned_without_checkpoint(
    real_indexer: Indexer, monkeypatch: pytest.MonkeyPatch,
) -> None:
    indexer = real_indexer
    path = indexer.config.vault_path / "a.md"
    path.write_text("new uncheckpointed note")

    def fail_keyword_write(chunks) -> None:
        raise RuntimeError("keyword write unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(indexer.keyword_store, "upsert_chunks", fail_keyword_write)
        failed = indexer.sync()
    assert failed.processed == 0
    assert len(failed.errors) == 1
    assert indexer.sync_state.tracked_paths() == ["a.md"]
    with sqlite3.connect(indexer.config.sync_state_path) as conn:
        assert conn.execute("SELECT * FROM note_state").fetchall() == []

    path.unlink()
    cleaned = indexer.sync()

    assert (cleaned.deleted, cleaned.errors) == (1, [])
    assert indexer.vector_store.get_by_path("a.md") == []
    assert indexer.keyword_store.chunks_by_path("a.md") == []
    assert indexer.sync_state.tracked_paths() == []
