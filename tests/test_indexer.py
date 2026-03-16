from pathlib import Path

import pytest

from obsidian_rag.config import RagConfig
from obsidian_rag.indexer import Indexer
from obsidian_rag.keyword_store import KeywordStore
from obsidian_rag.sync_state import SyncStateStore
from obsidian_rag.vector_store import InMemoryVectorStore


class _StubEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]


def _build_indexer(tmp_path: Path, vault_path: Path) -> Indexer:
    config = RagConfig(
        vault_path=vault_path,
        qdrant_path=tmp_path / "qdrant",
        fts_path=tmp_path / "fts.sqlite",
        sync_state_path=tmp_path / "sync_state.sqlite",
        chunk_size=20,
        chunk_overlap=0,
    )
    indexer = Indexer(
        config=config,
        embedder=_StubEmbedder(),
        vector_store=InMemoryVectorStore(),
        keyword_store=KeywordStore(config.fts_path),
        sync_state=SyncStateStore(config.sync_state_path),
    )
    indexer.initialize()
    return indexer


def test_file_mode_requires_file_path(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("hello", encoding="utf-8")
    indexer = _build_indexer(tmp_path, vault)

    with pytest.raises(ValueError, match="file_path"):
        indexer.sync(mode="file")


def test_file_mode_accepts_path_relative_to_vault(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# A\nhello world", encoding="utf-8")
    (vault / "b.md").write_text("# B\nother note", encoding="utf-8")
    indexer = _build_indexer(tmp_path, vault)

    result = indexer.sync(mode="file", file_path="a.md")

    assert result.processed == 1
    assert result.skipped == 0
    assert result.deleted == 0
    assert result.errors == []
    assert indexer.sync_state.tracked_paths() == ["a.md"]
