"""Vector storage backends for semantic search."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable

from second_brain.models import ChunkRecord, RetrievalHit


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

    def upsert_chunks(
        self, chunks: list[ChunkRecord], embeddings: list[list[float]]
    ) -> None:
        """Insert/update chunk vectors in memory."""

        for chunk, vec in zip(chunks, embeddings, strict=True):
            self._vectors[chunk.chunk_id] = (vec, chunk)

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        """Delete vectors by chunk id."""

        for cid in chunk_ids:
            self._vectors.pop(cid, None)

    def search(
        self,
        query_vector: list[float],
        limit: int | None = 10,
        metadata_filter: Callable[[dict[str, Any]], bool] | None = None,
    ) -> list[RetrievalHit]:
        """Return legacy dot-ranked chunks with separate cosine similarities."""

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
                    semantic_score=_cosine_similarity(query_vector, vec),
                )
            )
        scored.sort(key=lambda h: h.score, reverse=True)
        if metadata_filter is not None:
            scored = [hit for hit in scored if metadata_filter(hit.metadata)]
        return scored if limit is None else scored[:limit]

    def get_by_path(self, rel_path: str) -> list[RetrievalHit]:
        """Return all chunks for a note path, independent of the keyword store."""

        hits: list[RetrievalHit] = []
        for chunk_id, (_vec, chunk) in self._vectors.items():
            if chunk.metadata.get("path") != rel_path:
                continue
            hits.append(
                RetrievalHit(
                    chunk_id=chunk_id,
                    score=0.0,
                    source="semantic",
                    text=chunk.text,
                    metadata=chunk.metadata,
                )
            )
        return hits


class QdrantVectorStore:
    """Persistent vector store backed by local Qdrant."""

    def __init__(self, path: Path, collection_name: str) -> None:
        from qdrant_client import QdrantClient

        self.collection_name = collection_name
        # Remove any stale lock file left by a previously crashed process.
        # Qdrant's local storage uses a text-marker lock (not an OS flock),
        # so it won't be cleared automatically on unclean shutdown.
        lock_file = path / ".lock"
        lock_file.unlink(missing_ok=True)
        self.client = QdrantClient(path=str(path))

    def ensure_collection(self, vector_size: int) -> None:
        """Create Qdrant collection when it does not exist."""

        from qdrant_client.http import models

        collections = self.client.get_collections().collections
        existing = {c.name for c in collections}
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            )

    def upsert_chunks(
        self, chunks: list[ChunkRecord], embeddings: list[list[float]]
    ) -> None:
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

        # Independent store recovery may leave only keyword IDs. An absent
        # collection already satisfies vector deletion and must not block retry.
        if not chunk_ids or not self.client.collection_exists(self.collection_name):
            return

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=chunk_ids),
        )

    def search(
        self,
        query_vector: list[float],
        limit: int | None = 10,
        metadata_filter: Callable[[dict[str, Any]], bool] | None = None,
    ) -> list[RetrievalHit]:
        """Return ranked semantic matches, filtering before the requested limit.

        Python predicates support arbitrary frontmatter filters, so filtered
        searches rank the full collection once and apply the reference predicate.
        """
        if metadata_filter is not None:
            point_count = self.client.count(
                collection_name=self.collection_name,
                exact=True,
            ).count
            if point_count == 0:
                return []
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=point_count,
                with_payload=True,
            )
            points = response.points
        elif limit is None:
            points = []
            offset = 0
            page_size = 32
            while True:
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=page_size,
                    with_payload=True,
                    offset=offset,
                )
                page = response.points
                points.extend(page)
                if not page:
                    break
                offset += len(page)
        else:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True,
            )
            points = response.points

        hits: list[RetrievalHit] = []
        for point in points:
            payload = point.payload or {}
            metadata = dict(payload.get("metadata", {}))
            if metadata_filter is not None and not metadata_filter(metadata):
                continue
            semantic_score = float(point.score)
            hits.append(
                RetrievalHit(
                    chunk_id=str(point.id),
                    score=semantic_score,
                    source="semantic",
                    text=str(payload.get("text", "")),
                    metadata=metadata,
                    semantic_score=semantic_score,
                )
            )
            if limit is not None and len(hits) >= limit:
                return hits
        return hits

    def get_by_path(self, rel_path: str) -> list[RetrievalHit]:
        """Return all chunks for a note path via a payload filter (not similarity ranked)."""

        from qdrant_client.http import models

        # A first sync may inspect ownership before any embedded note has
        # created the collection, including a vault containing only empty notes.
        if not self.client.collection_exists(self.collection_name):
            return []

        query_filter = models.Filter(
            must=[models.FieldCondition(key="metadata.path", match=models.MatchValue(value=rel_path))]
        )

        hits: list[RetrievalHit] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                with_payload=True,
                limit=256,
                offset=offset,
            )
            for p in points:
                payload = p.payload or {}
                hits.append(
                    RetrievalHit(
                        chunk_id=str(p.id),
                        score=0.0,
                        source="semantic",
                        text=str(payload.get("text", "")),
                        metadata=dict(payload.get("metadata", {})),
                    )
                )
            if offset is None:
                break
        return hits


def _dot(a: list[float], b: list[float]) -> float:
    """Compute the legacy in-memory ranking score."""

    return sum(x * y for x, y in zip(a, b, strict=False))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity, returning zero for zero-length vectors."""

    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_product = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    return dot / norm_product if norm_product else 0.0
