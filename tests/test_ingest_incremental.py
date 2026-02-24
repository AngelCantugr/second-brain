from __future__ import annotations

import os
from pathlib import Path

from vault_mcp.config import Settings
from vault_mcp.ingest import run_ingestion


class FakeStore:
    def __init__(self) -> None:
        self.replaced_files: list[str] = []
        self.upsert_count = 0

    def replace_file_chunks(self, file_path: str, chunks, embeddings, note) -> None:
        self.replaced_files.append(file_path)
        self.upsert_count += len(chunks)


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[0.1, 0.2, 0.3] for _ in texts]


def _make_vault(vault_root: Path) -> None:
    (vault_root / "Topic").mkdir(parents=True)
    (vault_root / "Topic" / "a.md").write_text("# A\nalpha", encoding="utf-8")
    (vault_root / "Topic" / "b.md").write_text("# B\nbeta", encoding="utf-8")


def test_incremental_ingestion_only_reindexes_changed_files(tmp_path: Path) -> None:
    vault_root = tmp_path / "Obsidian"
    index_dir = tmp_path / "index"
    _make_vault(vault_root)

    settings = Settings(vault_root=vault_root, index_dir=index_dir)
    store = FakeStore()
    embedder = FakeEmbedder()

    first = run_ingestion(settings, embedder=embedder, store=store)
    assert first["indexed_files"] == 2
    assert sorted(store.replaced_files) == ["Topic/a.md", "Topic/b.md"]

    store.replaced_files.clear()
    second = run_ingestion(settings, embedder=embedder, store=store)
    assert second["indexed_files"] == 0
    assert store.replaced_files == []

    target = vault_root / "Topic" / "b.md"
    target.write_text("# B\nbeta updated", encoding="utf-8")
    st = target.stat()
    os.utime(target, (st.st_atime, st.st_mtime + 10))

    third = run_ingestion(settings, embedder=embedder, store=store)
    assert third["indexed_files"] == 1
    assert store.replaced_files == ["Topic/b.md"]
