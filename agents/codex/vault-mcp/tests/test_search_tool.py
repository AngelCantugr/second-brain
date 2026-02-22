from __future__ import annotations

from pathlib import Path

from vault_mcp.config import Settings
from vault_mcp.server import VaultService


class FakeStore:
    def __init__(self) -> None:
        self.last_query_top_k = None

    def search(self, query_embedding: list[float], top_k: int = 5):
        self.last_query_top_k = top_k
        return [
            {
                "note": "Rate Limiting",
                "excerpt": "Use token buckets",
                "score": 0.88,
                "path": "Engineering/rate-limiting.md",
            }
        ]


class FakeEmbedder:
    def embed_query(self, query: str) -> list[float]:
        assert query
        return [0.1, 0.2, 0.3]


def test_search_vault_contract(tmp_path: Path) -> None:
    vault_root = tmp_path / "Obsidian"
    vault_root.mkdir()

    settings = Settings(vault_root=vault_root, index_dir=tmp_path / "index")
    service = VaultService(settings=settings, store=FakeStore(), embedder=FakeEmbedder())

    out = service.search_vault("rate limiting", top_k=3)

    assert isinstance(out, list)
    assert set(out[0].keys()) == {"note", "excerpt", "score", "path"}


def test_get_note_list_topics_recent_activity(tmp_path: Path) -> None:
    vault_root = tmp_path / "Obsidian"
    (vault_root / "Engineering").mkdir(parents=True)
    (vault_root / "Engineering" / "rate-limiting.md").write_text("# RL\nBody", encoding="utf-8")

    settings = Settings(vault_root=vault_root, index_dir=tmp_path / "index")
    service = VaultService(settings=settings, store=FakeStore(), embedder=FakeEmbedder())

    note = service.get_note("Engineering/rate-limiting.md")
    topics = service.list_topics()
    recent = service.recent_activity(days=7)

    assert note.startswith("# RL")
    assert topics["Engineering"] == 1
    assert recent[0]["path"] == "Engineering/rate-limiting.md"
