"""Filesystem watcher for near real-time incremental indexing."""

from __future__ import annotations

import time
from pathlib import Path

from watchfiles import Change, watch

from obsidian_rag.service import RagService


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
            for change, path_str in changes:
                path = Path(path_str)
                if path.suffix != ".md":
                    continue
                if change == Change.deleted:
                    # Full incremental run handles deletions safely against state db.
                    self.service.sync(mode="incremental")
                    continue
                pending.add(path)

            now = time.monotonic()
            if now - last_flush < self.debounce_seconds:
                continue

            for file_path in sorted(pending):
                self.service.sync(mode="file", file_path=str(file_path))
            pending.clear()
            last_flush = now
