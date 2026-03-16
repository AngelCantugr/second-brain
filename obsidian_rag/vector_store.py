"""Vector storage backends for semantic search."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from obsidian_rag.models import ChunkRecord, RetrievalHit


@dataclass(slots=True)
class VectorSearchItem:
    """Legacy/auxiliary search result shape (not currently used)."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float


class InMemoryVectorStore:
    """Simple in-memory vector store for tests/local development."""

    def __init__(self) -> None:
        self._vectors: dict[str, tuple[list[float], ChunkRecord]] = {}

    def ensure_collection(self, vector_size: int) -> None:
        """No-op for in-memory backend (kept for API compatibility)."""

        _ = vector_size

    def upsert_chunks(self, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        """Insert/update chunk vectors in memory."""

        for chunk, vec in zip(chunks, embeddings, strict=True):
            self._vectors[chunk.chunk_id] = (vec, chunk)

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        """Delete vectors by chunk id."""

        for cid in chunk_ids:
            self._vectors.pop(cid, None)

    def search(self, query_vector: list[float], limit: int = 10) -> list[RetrievalHit]:
        """Return top-k chunks ranked by dot-product similarity."""

        scored: list[RetrievalHit] = []
        for chunk_id, (vec, chunk) in self._vectors.items():
            score = _dot(query_vector, vec)
            scored.append(
                RetrievalHit(
                    chunk_id=chunk_id,
                    score=score,
                    source="semantic",
                    text=chunk.text,
                    metadata=chunk.metadata,
                )
            )
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:limit]


class QdrantVectorStore:
    """Persistent vector store backed by local Qdrant."""

    def __init__(self, path: Path, collection_name: str) -> None:
        from qdrant_client import QdrantClient

        self.collection_name = collection_name
        self.client = QdrantClient(path=str(path))

    def ensure_collection(self, vector_size: int) -> None:
        """Create Qdrant collection when it does not exist."""

        from qdrant_client.http import models

        collections = self.client.get_collections().collections
        existing = {c.name for c in collections}
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )

    def upsert_chunks(self, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        """Insert/update vectors and payload metadata in Qdrant."""

        from qdrant_client.http import models

        points = []
        for chunk, vector in zip(chunks, embeddings, strict=True):
            points.append(
                models.PointStruct(
                    id=chunk.chunk_id,
                    vector=vector,
                    payload={"text": chunk.text, "metadata": chunk.metadata},
                )
            )
        self.client.upsert(collection_name=self.collection_name, points=points)

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        """Delete Qdrant points by ids."""

        from qdrant_client.http import models

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=chunk_ids),
        )

    def search(self, query_vector: list[float], limit: int = 10) -> list[RetrievalHit]:
        """Return top-k semantic matches from Qdrant."""

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        points = response.points
        hits: list[RetrievalHit] = []
        for p in points:
            payload = p.payload or {}
            hits.append(
                RetrievalHit(
                    chunk_id=str(p.id),
                    score=float(p.score),
                    source="semantic",
                    text=str(payload.get("text", "")),
                    metadata=dict(payload.get("metadata", {})),
                )
            )
        return hits


def _dot(a: list[float], b: list[float]) -> float:
    """Compute dot product between two vectors."""

    return sum(x * y for x, y in zip(a, b, strict=False))
