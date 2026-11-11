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
        lambda root, **_: iter([{(Change.modified, str(note))}]),
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
    monkeypatch.setattr("second_brain.watcher.watch", lambda _root, **_: iter([{(Change.added, str(file_path))}]))
    ticks = iter([0.0, 2.0])
    monkeypatch.setattr(
        "second_brain.watcher.time",
        SimpleNamespace(monotonic=lambda: next(ticks), time=lambda: 1234.5),
    )
    monkeypatch.setattr(service, "sync", lambda **_: None)

    VaultWatcher(service, debounce_seconds=1.0).run()

    assert second_service.status()["watcher_last_event"] == 1234.5


def test_watcher_records_deletion_before_failed_sync(tmp_path: Path, monkeypatch) -> None:
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
    other_store_service = RagService(config, use_in_memory_vector=True)
    missing_path = vault / "deleted.md"
    monkeypatch.setattr(
        "second_brain.watcher.watch",
        lambda _root, **_: iter([{(Change.deleted, str(missing_path))}]),
    )
    monkeypatch.setattr(
        "second_brain.watcher.time",
        SimpleNamespace(monotonic=lambda: 10.0, time=lambda: 5678.0),
    )

    def fail_sync(**_: object) -> None:
        raise RuntimeError("indexing failed")

    monkeypatch.setattr(service, "sync", fail_sync)

    # A failed sync is logged, not raised: one bad file must not end the watch loop.
    VaultWatcher(service).run()

    assert other_store_service.sync_state.watcher_last_event() == 5678.0


class _StubEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]

    def health(self) -> bool:
        return True


def _service(tmp_path: Path) -> RagService:
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
    service.embedder = _StubEmbedder()
    service.indexer.embedder = service.embedder
    return service


def _script(monkeypatch, steps):
    """Feed the watcher ``(clock, changes)`` steps; return what was synced by each step.

    ``synced_before[i]`` is the list of sync calls made before step ``i`` was
    delivered, so a test can tell a flush inside the loop from the final one.
    """

    calls: list[tuple[str, str | None]] = []
    synced_before: list[list[tuple[str, str | None]]] = []
    clock = {"now": 0.0}

    def fake_watch(_root, **_options):
        for now, changes in steps:
            synced_before.append(list(calls))
            clock["now"] = now
            yield changes

    monkeypatch.setattr("second_brain.watcher.watch", fake_watch)
    monkeypatch.setattr(
        "second_brain.watcher.time",
        SimpleNamespace(monotonic=lambda: clock["now"], time=lambda: 1000.0 + clock["now"]),
    )
    return calls, synced_before


def _record_sync(service: RagService, monkeypatch, calls, fail_on: str | None = None) -> None:
    def sync(mode: str = "incremental", file_path: str | None = None) -> dict:
        calls.append((mode, None if file_path is None else Path(file_path).name))
        if fail_on is not None and file_path is not None and file_path.endswith(fail_on):
            raise RuntimeError("embedding failed")
        return {"errors": []}

    monkeypatch.setattr(service, "sync", sync)


def test_watcher_flushes_after_a_quiet_period_without_waiting_for_another_event(
    tmp_path: Path, monkeypatch
) -> None:
    service = _service(tmp_path)
    a, b = service.config.vault_path / "a.md", service.config.vault_path / "b.md"
    calls, synced_before = _script(
        monkeypatch,
        [
            (0.0, {(Change.modified, str(a))}),
            (0.2, {(Change.modified, str(b))}),
            (1.0, set()),  # idle tick, still inside the quiet period
            (2.5, set()),  # idle tick after it: both files flush here
            (3.0, set()),
        ],
    )
    _record_sync(service, monkeypatch, calls)

    VaultWatcher(service, debounce_seconds=2.0).run()

    assert synced_before[3] == []
    assert synced_before[4] == [("file", "a.md"), ("file", "b.md")]
    assert calls == [("file", "a.md"), ("file", "b.md")]


def test_watcher_syncs_a_note_once_for_a_burst_of_saves(tmp_path: Path, monkeypatch) -> None:
    service = _service(tmp_path)
    a = service.config.vault_path / "a.md"
    calls, _ = _script(
        monkeypatch,
        [(t, {(Change.modified, str(a))}) for t in (0.0, 1.0, 2.0, 3.0)] + [(6.0, set())],
    )
    _record_sync(service, monkeypatch, calls)

    VaultWatcher(service, debounce_seconds=2.0).run()

    assert calls == [("file", "a.md")]


def test_watcher_keeps_going_after_a_file_fails_to_sync(tmp_path: Path, monkeypatch) -> None:
    service = _service(tmp_path)
    a, b = service.config.vault_path / "a.md", service.config.vault_path / "b.md"
    calls, _ = _script(
        monkeypatch,
        [
            (0.0, {(Change.modified, str(a)), (Change.modified, str(b))}),
            (5.0, set()),
            (6.0, {(Change.modified, str(b))}),
            (12.0, set()),
        ],
    )
    _record_sync(service, monkeypatch, calls, fail_on="a.md")

    VaultWatcher(service, debounce_seconds=2.0).run()

    assert calls == [("file", "a.md"), ("file", "b.md"), ("file", "b.md")]


def test_watcher_removes_a_deleted_note_with_a_single_file_sync(
    tmp_path: Path, monkeypatch
) -> None:
    service = _service(tmp_path)
    vault = service.config.vault_path
    (vault / "keep.md").write_text("# Keep\nstays", encoding="utf-8")
    gone = vault / "gone.md"
    gone.write_text("# Gone\nremoved soon", encoding="utf-8")
    service.sync(mode="full")
    gone.unlink()
    modes: list[str] = []
    real_sync = service.sync

    def sync(mode: str = "incremental", file_path: str | None = None) -> dict:
        modes.append(mode)
        return real_sync(mode=mode, file_path=file_path)

    monkeypatch.setattr(service, "sync", sync)
    _script(monkeypatch, [(0.0, {(Change.deleted, str(gone))})])

    VaultWatcher(service, debounce_seconds=2.0).run()

    assert modes == ["file"]
    assert service.sync_state.tracked_paths() == ["keep.md"]


def test_watcher_ignores_a_note_created_and_removed_before_the_flush(
    tmp_path: Path, monkeypatch
) -> None:
    service = _service(tmp_path)
    draft = service.config.vault_path / "draft.md"
    _script(monkeypatch, [(0.0, {(Change.added, str(draft))}), (9.0, set())])

    VaultWatcher(service, debounce_seconds=2.0).run()

    assert service.sync_state.tracked_paths() == []


def test_watcher_rescans_when_a_folder_is_added_or_removed(tmp_path: Path, monkeypatch) -> None:
    """Moving a folder reports the folder, not the notes inside it."""

    service = _service(tmp_path)
    vault = service.config.vault_path
    moved_in = vault / "Projects"
    moved_in.mkdir()
    calls, _ = _script(
        monkeypatch,
        [
            (0.0, {(Change.added, str(moved_in))}),
            (5.0, set()),
            (6.0, {(Change.deleted, str(vault / "Archive"))}),
            (12.0, set()),
            (13.0, {(Change.added, str(vault / ".obsidian"))}),
            (20.0, set()),
        ],
    )
    (vault / ".obsidian").mkdir()
    _record_sync(service, monkeypatch, calls)

    VaultWatcher(service, debounce_seconds=2.0).run()

    assert calls == [("incremental", None), ("incremental", None)]


def test_watcher_thread_starts_reports_running_and_stops(tmp_path: Path, monkeypatch) -> None:
    service = _service(tmp_path)

    def fake_watch(_root, stop_event, **_options):
        stop_event.wait(timeout=5)
        return
        yield  # pragma: no cover - makes this a generator

    monkeypatch.setattr("second_brain.watcher.watch", fake_watch)
    watcher = VaultWatcher(service)

    assert service.status()["watcher_running"] is False
    watcher.start()
    assert watcher.running
    assert service.status()["watcher_running"] is True
    watcher.stop()

    assert not watcher.running
    assert service.status()["watcher_running"] is False


def test_watcher_ignores_index_stores_kept_inside_the_vault(tmp_path: Path, monkeypatch) -> None:
    """The default config puts ``data/`` in the vault; a sync must not trigger a rescan."""

    vault = tmp_path / "vault"
    vault.mkdir()
    data = vault / "data"
    config = RagConfig(
        vault_path=vault,
        qdrant_path=data / "qdrant",
        fts_path=data / "fts.sqlite",
        sync_state_path=data / "sync_state.sqlite",
        graph_enabled=False,
    )
    service = RagService(config, use_in_memory_vector=True)
    segment = data / "qdrant" / "collection" / "chunks"
    segment.mkdir(parents=True)
    calls, _ = _script(
        monkeypatch,
        [
            (0.0, {(Change.added, str(segment)), (Change.deleted, str(data / "fts.sqlite-journal"))}),
            (9.0, set()),
        ],
    )
    _record_sync(service, monkeypatch, calls)

    VaultWatcher(service, debounce_seconds=2.0).run()

    assert calls == []
    assert service.sync_state.watcher_last_event() is None
