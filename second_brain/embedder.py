"""Embedding providers.

Embeddings convert text into vectors that support semantic similarity search.
"""

from __future__ import annotations

import logging
from typing import Protocol

import requests

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Interface expected by the indexer/service for embedding text."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class EmbeddingError(RuntimeError):
    """An embedding batch failed; the message names the batch and Ollama's reason."""


class OllamaEmbedder:
    """Local embedding adapter for Ollama's batch ``/api/embed`` endpoint.

    ``/api/embed`` returns L2-normalized vectors, where the legacy
    ``/api/embeddings`` endpoint returned unnormalized ones. Their directions
    match, and Qdrant and the note graph compare vectors by cosine, so search
    and edge scores do not depend on the switch.
    """

    def __init__(
        self, base_url: str, model: str, timeout: int = 120, batch_size: int = 32
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in order, ``batch_size`` texts per request.

        A failed batch raises ``EmbeddingError`` at once, without sending later
        batches. The indexer embeds each note before writing anything for it,
        so the note is reported in the sync errors, keeps its previous index
        entries, and is retried by the next sync.

        The one retry is a batch Ollama rejects for exceeding the model's
        context length. Chunks are sized in words and the limit is in tokens,
        so code-dense text (500 words of fenced Python can pass 2,048 tokens)
        can overflow. That batch is resent once with ``truncate=True`` and a
        warning is logged: the vector then covers the chunk's head, while
        keyword search still indexes the full text.
        """

        vectors: list[list[float]] = []
        batch_count = -(-len(texts) // self.batch_size)
        for batch_number, start in enumerate(
            range(0, len(texts), self.batch_size), start=1
        ):
            batch = texts[start : start + self.batch_size]
            where = (
                f"batch {batch_number}/{batch_count} "
                f"(texts {start}-{start + len(batch) - 1})"
            )
            vectors.extend(self._embed_batch(batch, where))
        return vectors

    def _embed_batch(self, batch: list[str], where: str) -> list[list[float]]:
        # truncate=False first, so an over-length chunk is detected and logged
        # instead of silently cut short.
        response = self._post(batch, where, truncate=False)
        if _is_context_overflow(response):
            logger.warning(
                "Ollama /api/embed %s: an input exceeds the context length of %s; "
                "resending with truncation, so its vector covers only its start",
                where,
                self.model,
            )
            # Truncation only cuts inputs over the limit; the others in the
            # batch get the same vectors as before.
            response = self._post(batch, where, truncate=True)

        if not response.ok:
            raise EmbeddingError(
                f"Ollama /api/embed {where} failed with HTTP {response.status_code}: "
                f"{_error_detail(response)}"
            )
        try:
            embeddings = response.json().get("embeddings")
        except (ValueError, AttributeError):
            embeddings = None
        if not isinstance(embeddings, list) or len(embeddings) != len(batch):
            got = len(embeddings) if isinstance(embeddings, list) else 0
            raise EmbeddingError(
                f"Ollama /api/embed {where} returned {got} embeddings for {len(batch)} inputs"
            )
        return embeddings

    def _post(self, batch: list[str], where: str, truncate: bool) -> requests.Response:
        try:
            return requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": batch, "truncate": truncate},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise EmbeddingError(f"Ollama /api/embed {where} failed: {exc}") from exc

    def health(self) -> bool:
        """Return ``True`` if Ollama appears reachable."""

        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.ok
        except requests.RequestException:
            return False


def _is_context_overflow(response: requests.Response) -> bool:
    # Ollama answers 400 "the input length exceeds the context length".
    return response.status_code == 400 and "context length" in _error_detail(response)


def _error_detail(response: requests.Response) -> str:
    """Prefer Ollama's JSON ``error`` field; fall back to the raw body."""

    try:
        return str(response.json()["error"])
    except (ValueError, KeyError, TypeError):
        return response.text[:500]
