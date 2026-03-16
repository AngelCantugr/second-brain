"""Embedding providers.

Embeddings convert text into vectors that support semantic similarity search.
"""

from __future__ import annotations

from typing import Protocol

import requests


class Embedder(Protocol):
    """Interface expected by the indexer/service for embedding text."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OllamaEmbedder:
    """Local embedding adapter for Ollama's ``/api/embeddings`` endpoint."""

    def __init__(self, base_url: str, model: str, timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed input texts with the configured Ollama model."""

        vectors: list[list[float]] = []
        for text in texts:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            vectors.append(payload["embedding"])
        return vectors

    def health(self) -> bool:
        """Return ``True`` if Ollama appears reachable."""

        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.ok
        except requests.RequestException:
            return False
