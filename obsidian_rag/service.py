"""High-level service API used by CLI and MCP layers."""

from __future__ import annotations

from dataclasses import asdict

from obsidian_rag.config import RagConfig
from obsidian_rag.embedder import OllamaEmbedder
from obsidian_rag.indexer import Indexer
from obsidian_rag.keyword_store import KeywordStore, matches_filters
from obsidian_rag.retrieval import normalize_query, reciprocal_rank_fusion
from obsidian_rag.sync_state import SyncStateStore
from obsidian_rag.vector_store import InMemoryVectorStore, QdrantVectorStore


class RagService:
    """Facade around indexing, retrieval, and health/status endpoints."""

    def __init__(self, config: RagConfig, use_in_memory_vector: bool = False) -> None:
        self.config = config
        self.embedder = OllamaEmbedder(config.ollama_url, config.embedding_model)
        if use_in_memory_vector:
            self.vector_store = InMemoryVectorStore()
        else:
            self.vector_store = QdrantVectorStore(config.qdrant_path, config.collection_name)
        self.keyword_store = KeywordStore(config.fts_path)
        self.sync_state = SyncStateStore(config.sync_state_path)
        self.indexer = Indexer(
            config=config,
            embedder=self.embedder,
            vector_store=self.vector_store,
            keyword_store=self.keyword_store,
            sync_state=self.sync_state,
        )
        self.indexer.initialize()

    def sync(self, mode: str = "incremental", file_path: str | None = None) -> dict:
        """Trigger synchronization from vault files into indexes."""

        result = self.indexer.sync(mode=mode, file_path=file_path)
        return asdict(result)

    def search(self, query: str, filters: dict | None = None, top_k: int = 10) -> dict:
        """Return hybrid retrieval hits for a query."""

        normalized = normalize_query(query)
        effective_filters = dict(filters or {})
        query_vec = self.embedder.embed([normalized])[0]
        semantic_hits = [
            h
            for h in self.vector_store.search(query_vec, limit=top_k * 3)
            if matches_filters(h.metadata, effective_filters)
        ][:top_k]
        keyword_hits = self.keyword_store.search(normalized, limit=top_k, filters=effective_filters)
        merged = reciprocal_rank_fusion(semantic_hits, keyword_hits)

        return {
            "query": query,
            "hits": [
                {
                    "chunk_id": h.chunk_id,
                    "score": h.score,
                    "source": h.source,
                    "text": h.text,
                    "metadata": h.metadata,
                }
                for h in merged[:top_k]
            ],
        }

    def query(self, query: str, filters: dict | None = None, top_k: int = 8) -> dict:
        """Return answer draft + citations + debug fields for a query."""

        results = self.search(query=query, filters=filters, top_k=top_k)
        citations = []
        for hit in results["hits"]:
            metadata = hit["metadata"]
            citations.append(
                {
                    "chunk_id": hit["chunk_id"],
                    "path": metadata.get("path"),
                    "heading_path": metadata.get("heading_path", "root"),
                }
            )

        answer_lines = [f"- {c['path']} :: {c['heading_path']}" for c in citations]
        answer = "\n".join(answer_lines) if answer_lines else "No relevant context found."

        return {
            "answer_draft": answer,
            "citations": citations,
            "chunks": results["hits"],
            "debug_scores": [h["score"] for h in results["hits"]],
        }

    def note_context(self, note_path: str) -> dict:
        """Return chunk/context summary for one note path."""

        matching = self.keyword_store.chunks_by_path(note_path)
        links = []
        for hit in matching:
            links.extend(hit.metadata.get("links", []))

        return {
            "note_path": note_path,
            "chunk_ids": [h.chunk_id for h in matching],
            "chunk_count": len(matching),
            "outlinks": sorted(set(links)),
            "backlinks": [],
        }

    def status(self) -> dict:
        """Return index/model runtime status for operational visibility."""

        return {
            "watch_enabled": self.config.watch_enabled,
            "max_context_chunks": self.config.max_context_chunks,
            "model": self.config.embedding_model,
            "index_size": self.keyword_store.count_chunks(),
            "last_sync_timestamp": self.sync_state.last_sync_timestamp(),
            "watcher_state": "enabled" if self.config.watch_enabled else "disabled",
            "last_tracked_files": len(self.sync_state.tracked_paths()),
            "model_available": self.embedder.health(),
        }

    def health(self) -> dict:
        """Return health checks for vector store, embedder, and keyword db."""

        qdrant_ok = True
        try:
            if hasattr(self.vector_store, "client"):
                self.vector_store.client.get_collections()
        except Exception:
            qdrant_ok = False

        return {
            "qdrant": qdrant_ok,
            "ollama": self.embedder.health(),
            "fts": self.config.fts_path.exists(),
        }
