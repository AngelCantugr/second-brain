"""Filesystem watcher for near real-time incremental indexing."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from watchfiles import Change, watch

from second_brain.scanner import is_eligible_source_path, is_excluded_path
from second_brain.service import RagService

logger = logging.getLogger(__name__)

# How often the watch loop wakes with no change, to flush once the vault is quiet.
_IDLE_TICK_MS = 1000
# A SQLite database and the files it writes beside itself.
_SQLITE_SIDECARS = ("", "-journal", "-wal", "-shm")


class VaultWatcher:
    """Re-index vault notes shortly after they change on disk.

    Changes are collected until the vault has been quiet for
    ``debounce_seconds``, then each changed note is synced once. Waiting for
    quiet, rather than syncing on a timer, keeps a note that an editor
    autosaves every couple of seconds from being re-embedded on every save.
    """

    def __init__(self, service: RagService, debounce_seconds: float = 5.0) -> None:
        self.service = service
        self.debounce_seconds = debounce_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Run the watch loop on a daemon thread, so it ends with the process."""

        if self.running:
            return
        self._stop.clear()
        self.service.watcher = self
        self._thread = threading.Thread(
            target=self._run_logged, name="second-brain-watcher", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        """Ask the loop to end and wait for it; pending changes are flushed first."""

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def _run_logged(self) -> None:
        try:
            self.run()
        # The thread has no caller to raise to; without this the watcher
        # would stop silently (a vault folder that does not exist, say).
        except Exception:  # noqa: BLE001
            logger.exception("vault watcher stopped; run a sync to pick up changes")

    def run(self) -> None:
        """Watch the vault until stopped, syncing changed notes after each quiet period."""

        config = self.service.config
        vault_root = config.vault_path.resolve()
        pending: set[Path] = set()
        rescan = False
        last_change = 0.0

        # yield_on_timeout makes the loop wake on idle ticks too. Without it a
        # change that arrives inside the quiet period would wait for the next
        # filesystem event, however long that takes, before being synced.
        for changes in watch(
            config.vault_path,
            stop_event=self._stop,
            rust_timeout=_IDLE_TICK_MS,
            yield_on_timeout=True,
        ):
            relevant = False
            for change, path_str in changes:
                path = Path(path_str)
                try:
                    relative = path.resolve().relative_to(vault_root)
                except ValueError:
                    continue
                if self._is_index_data(path.resolve()):
                    continue
                # Folders first: one named like a note (``Projects.md``) would
                # otherwise be synced as a single file and its notes missed.
                if self._is_folder_move(change, path, relative, vault_root):
                    rescan = True
                    relevant = True
                elif is_eligible_source_path(
                    relative, config.exclude_globs, config.include_extensions
                ):
                    pending.add(path)
                    relevant = True

            if relevant:
                self.service.sync_state.record_watcher_event(time.time())
                last_change = time.monotonic()
            if (pending or rescan) and time.monotonic() - last_change >= self.debounce_seconds:
                self._flush(pending, rescan)
                pending, rescan = set(), False

        if pending or rescan:
            self._flush(pending, rescan)

    def _is_index_data(self, resolved: Path) -> bool:
        """Whether a path belongs to the index stores, which may sit inside the vault.

        The default config keeps ``data/`` under the vault. A sync writes
        there, and treating a new Qdrant segment folder as a moved folder of
        notes would make every sync trigger another one.
        """

        config = self.service.config
        if resolved.is_relative_to(config.qdrant_path.resolve()):
            return True
        # Exact names only: a note such as ``fts.sqlite-notes.md`` shares the
        # database's prefix and must still be synced.
        return any(
            str(resolved) == f"{database.resolve()}{sidecar}"
            for database in (config.fts_path, config.sync_state_path)
            for sidecar in _SQLITE_SIDECARS
        )

    def _is_folder_move(
        self, change: Change, path: Path, relative: Path, vault_root: Path
    ) -> bool:
        """Whether a change may add or remove notes without naming them.

        Moving, renaming or deleting a folder is reported for the folder
        alone. A removed path can no longer be inspected, and its name says
        nothing (``Projects.v1`` is a folder), so it counts as a folder when
        an indexed note sits under it.
        """

        if is_excluded_path(path.resolve(), vault_root, self.service.config.exclude_globs):
            return False
        if change == Change.added:
            return path.is_dir()
        if change != Change.deleted:
            return False
        prefix = relative.as_posix() + "/"
        return any(
            tracked.startswith(prefix) for tracked in self.service.sync_state.tracked_paths()
        )

    def _flush(self, pending: set[Path], rescan: bool) -> None:
        if rescan:
            # A vault scan finds every changed note, the pending ones included.
            self._sync(mode="incremental")
            return
        # A deleted note goes through file mode too: it removes the note
        # without the full vault scan an incremental sync would cost.
        for file_path in sorted(pending):
            self._sync(mode="file", file_path=str(file_path))

    def _sync(self, **kwargs: str) -> None:
        """Run one sync; a failure is logged so the next change is still handled."""

        try:
            result = self.service.sync(**kwargs)
        except FileNotFoundError as exc:
            # Usually a file created and removed again before the flush.
            logger.info("vault watcher skipped %s: %s", kwargs.get("file_path", "scan"), exc)
        except Exception:  # noqa: BLE001
            logger.warning(
                "vault watcher could not sync %s; it is retried on its next change or sync",
                kwargs.get("file_path", "the vault"),
                exc_info=True,
            )
        else:
            for error in (result or {}).get("errors", []):
                logger.warning("vault watcher sync: %s", error)
