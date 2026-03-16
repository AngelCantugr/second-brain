from pathlib import Path

from obsidian_rag.sync_state import SyncStateStore


def test_sync_state_tracks_hash_and_detects_changes(tmp_path: Path) -> None:
    store = SyncStateStore(tmp_path / "sync_state.sqlite")
    store.initialize()

    assert store.should_reindex("notes/A.md", "h1") is True
    store.record_note("notes/A.md", "h1", 1.23)
    assert store.should_reindex("notes/A.md", "h1") is False
    assert store.should_reindex("notes/A.md", "h2") is True

    store.remove_note("notes/A.md")
    assert store.should_reindex("notes/A.md", "h2") is True
