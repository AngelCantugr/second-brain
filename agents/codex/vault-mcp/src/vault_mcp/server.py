from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

import ollama

from vault_mcp.config import Settings
from vault_mcp.index_store import ChromaIndexStore
from vault_mcp.metadata import iter_markdown_files
from vault_mcp.topic_tree import count_notes_by_top_level


class SearchStore(Protocol):
    def search(self, query_embedding: list[float], top_k: int = 5):
        ...


class QueryEmbedder(Protocol):
    def embed_query(self, query: str) -> list[float]:
        ...


class OllamaQueryEmbedder:
    def __init__(self, model: str) -> None:
        self.model = model

    def embed_query(self, query: str) -> list[float]:
        response = ollama.embed(model=self.model, input=[query])
        return response["embeddings"][0]


class VaultService:
    def __init__(
        self,
        settings: Settings,
        store: SearchStore | None = None,
        embedder: QueryEmbedder | None = None,
    ) -> None:
        self.settings = settings
        self.vault_root = settings.resolved_vault_root()
        self.store = store or ChromaIndexStore(
            str(settings.resolved_index_dir()), settings.collection_name
        )
        self.embedder = embedder or OllamaQueryEmbedder(settings.embedding_model)

    def search_vault(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        vector = self.embedder.embed_query(query)
        raw_results = self.store.search(vector, top_k=top_k)

        results: list[dict[str, Any]] = []
        for item in raw_results:
            if is_dataclass(item):
                payload = asdict(item)
            else:
                payload = dict(item)
            results.append(
                {
                    "note": payload.get("note", ""),
                    "excerpt": payload.get("excerpt", ""),
                    "score": float(payload.get("score", 0.0)),
                    "path": payload.get("path", ""),
                }
            )
        return results

    def get_note(self, path: str) -> str:
        candidate = (self.vault_root / path).resolve()
        if not str(candidate).startswith(str(self.vault_root.resolve())):
            raise ValueError("path escapes vault root")
        if not candidate.exists() or candidate.suffix.lower() != ".md":
            raise FileNotFoundError(path)
        return candidate.read_text(encoding="utf-8")

    def list_topics(self) -> dict[str, int]:
        paths = iter_markdown_files(self.vault_root)
        return count_notes_by_top_level(self.vault_root, paths)

    def recent_activity(self, days: int = 7) -> list[dict[str, Any]]:
        if days <= 0:
            raise ValueError("days must be > 0")
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        recent: list[dict[str, Any]] = []
        for path in iter_markdown_files(self.vault_root):
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            if modified < cutoff:
                continue
            recent.append(
                {
                    "path": path.relative_to(self.vault_root).as_posix(),
                    "date_modified": modified.isoformat(),
                }
            )
        return sorted(recent, key=lambda item: item["date_modified"], reverse=True)


def build_server(service: VaultService):
    from mcp.server.fastmcp import FastMCP

    app = FastMCP("vault")

    @app.tool()
    def search_vault(query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return service.search_vault(query=query, top_k=top_k)

    @app.tool()
    def get_note(path: str) -> str:
        return service.get_note(path=path)

    @app.tool()
    def list_topics() -> dict[str, int]:
        return service.list_topics()

    @app.tool()
    def recent_activity(days: int = 7) -> list[dict[str, Any]]:
        return service.recent_activity(days=days)

    return app


def main() -> None:
    service = VaultService(Settings())
    app = build_server(service)
    app.run(transport="stdio")


if __name__ == "__main__":
    main()
