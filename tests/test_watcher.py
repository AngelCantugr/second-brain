from pathlib import Path
from types import SimpleNamespace

from watchfiles import Change

from second_brain.config import RagConfig
from second_brain.service import RagService
from second_brain.watcher import VaultWatcher


def test_watcher_modified_event_respects_scanner_exclusions(
    tmp_path: Path, monkeypatch
) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    config = RagConfig(
        vault_path=vault,
        qdrant_path=tmp_path / "qdrant",
        fts_path=tmp_path / "fts.sqlite",
        sync_state_path=tmp_path / "sync_state.sqlite",
    )
    service = RagService(config, use_in_memory_vector=True)
    note = vault / "CLAUDE.md"
    note.write_text("# Instructions\nsearchable needle", encoding="utf-8")

    monkeypatch.setattr(
        "second_brain.watcher.watch",
        lambda root: iter([{(Change.modified, str(note))}]),
    )

    VaultWatcher(service, debounce_seconds=0).run()

    assert service.indexer.sync_state.tracked_paths() == []
    assert service.indexer.vector_store.get_by_path("CLAUDE.md") == []
    assert service.indexer.keyword_store.chunk_ids_by_path("CLAUDE.md") == []


def test_watcher_records_batch_timestamp_visible_to_another_service(tmp_path: Path, monkeypatch) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    config = RagConfig(
        vault_path=vault,
        qdrant_path=tmp_path / "qdrant",
        fts_path=tmp_path / "fts.sqlite",
        sync_state_path=tmp_path / "sync_state.sqlite",
        graph_enabled=False,
    )
    service = RagService(config, use_in_memory_vector=True)
    service.embedder.health = lambda: True
    second_service = RagService(config, use_in_memory_vector=True)
    second_service.embedder.health = lambda: True
    assert second_service.status()["watcher_last_event"] is None
    file_path = vault / "note.md"
    file_path.write_text("note", encoding="utf-8")
    monkeypatch.setattr("second_brain.watcher.watch", lambda _: iter([{(Change.added, str(file_path))}]))
    ticks = iter([0.0, 2.0])
    monkeypatch.setattr(
        "second_brain.watcher.time",
        SimpleNamespace(monotonic=lambda: next(ticks), time=lambda: 1234.5),
    )
    monkeypatch.setattr(service, "sync", lambda **_: None)

    VaultWatcher(service, debounce_seconds=1.0).run()

    assert second_service.status()["watcher_last_event"] == 1234.5
