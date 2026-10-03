"""Filesystem watcher for near real-time incremental indexing."""

from __future__ import annotations

import time
from pathlib import Path

from watchfiles import Change, watch

from second_brain.scanner import is_eligible_markdown_path
from second_brain.service import RagService


class VaultWatcher:
    """Monitor vault changes and trigger debounced sync operations."""

    def __init__(self, service: RagService, debounce_seconds: float = 1.0) -> None:
        self.service = service
        self.debounce_seconds = debounce_seconds

    def run(self) -> None:
        """Start watch loop and process create/update/delete events."""

        pending: set[Path] = set()
        last_flush = time.monotonic()

        for changes in watch(self.service.config.vault_path):
            observed_markdown_event = False
            for change, path_str in changes:
                path = Path(path_str)
                try:
                    relative_path = path.resolve().relative_to(
                        self.service.config.vault_path.resolve()
                    )
                except ValueError:
                    continue
                if not is_eligible_markdown_path(
                    relative_path, self.service.config.exclude_globs
                ):
                    continue
                observed_markdown_event = True
                if change == Change.deleted:
                    # Full incremental run handles deletions safely against state db.
                    self.service.sync(mode="incremental")
                    continue
                pending.add(path)

            if observed_markdown_event:
                self.service.sync_state.record_watcher_event(time.time())

            now = time.monotonic()
            if now - last_flush < self.debounce_seconds:
                continue

            for file_path in sorted(pending):
                self.service.sync(mode="file", file_path=str(file_path))
            pending.clear()
            last_flush = now
