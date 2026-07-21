"""High-level service API used by CLI and MCP layers."""

from __future__ import annotations

from dataclasses import asdict

from obsidian_rag.config import RagConfig
from obsidian_rag.embedder import OllamaEmbedder
from obsidian_rag.graph import (
    GraphStore,
    articulation_points,
    build_nx_graph,
    canonical_pair,
    compute_clusters,
    shortest_evidence_path,
)
from obsidian_rag.indexer import Indexer
from obsidian_rag.keyword_store import KeywordStore, matches_filters
from obsidian_rag.models import RetrievalHit
from obsidian_rag.retrieval import normalize_query, reciprocal_rank_fusion
from obsidian_rag.sync_state import SyncStateStore
from obsidian_rag.vector_store import InMemoryVectorStore, QdrantVectorStore


MAX_TOP_K = 50


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
        self.graph_store = GraphStore(config.fts_path)
        self.indexer = Indexer(
            config=config,
            embedder=self.embedder,
            vector_store=self.vector_store,
            keyword_store=self.keyword_store,
            sync_state=self.sync_state,
            graph_store=self.graph_store,
        )
        self.indexer.initialize()

    def sync(self, mode: str = "incremental", file_path: str | None = None) -> dict:
        """Trigger synchronization from vault files into indexes."""

        result = self.indexer.sync(mode=mode, file_path=file_path)
        return asdict(result)

    def search(self, query: str, filters: dict | None = None, top_k: int = 10) -> dict:
        """Return hybrid retrieval hits for a query."""

        normalized = normalize_query(query)
        if not normalized:
            raise ValueError("query must not be empty or whitespace-only")
        if top_k < 1 or top_k > MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}, got {top_k}")
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

        answer = self._build_extractive_answer(results["hits"])

        return {
            "answer_draft": answer,
            "citations": citations,
            "chunks": results["hits"],
            "debug_scores": [h["score"] for h in results["hits"]],
        }

    @staticmethod
    def _build_extractive_answer(hits: list[dict], max_chunks: int = 3, snippet_chars: int = 320) -> str:
        """Build a naive extractive answer from the top hits' actual chunk text."""

        if not hits:
            return "No relevant context found."

        snippets = []
        for hit in hits[:max_chunks]:
            text = " ".join((hit.get("text") or "").split())
            if len(text) > snippet_chars:
                text = text[:snippet_chars].rstrip() + "..."
            metadata = hit.get("metadata") or {}
            path = metadata.get("path") or "unknown"
            heading_path = metadata.get("heading_path") or "root"
            snippets.append(f"[{path} :: {heading_path}] {text}")

        return "\n\n".join(snippets)

    def note_context(self, note_path: str) -> dict:
        """Return chunk/context summary for one note path.

        Reads both stores (like ``search`` does) so a note whose keyword-store
        write is missing or drifted from the vector store is still reported.
        """

        by_id: dict[str, RetrievalHit] = {}
        for hit in [*self.keyword_store.chunks_by_path(note_path), *self.vector_store.get_by_path(note_path)]:
            by_id.setdefault(hit.chunk_id, hit)
        matching = list(by_id.values())
        links = []
        for hit in matching:
            links.extend(hit.metadata.get("links", []))

        note_title = matching[0].metadata.get("note_title") if matching else None
        backlinks = (
            self.keyword_store.backlinks_for_title(note_title, exclude_path=note_path)
            if note_title
            else []
        )

        return {
            "note_path": note_path,
            "chunk_ids": [h.chunk_id for h in matching],
            "chunk_count": len(matching),
            "outlinks": sorted(set(links)),
            "backlinks": backlinks,
        }

    def related(self, note_path: str, top_k: int = 10) -> dict:
        """Return this note's graph neighbors, ranked by composite score.

        Each neighbor reports the per-signal breakdown (semantic/link/tag/
        comention) and structural evidence so a caller can see *why* two
        notes are associated, not just that they are.
        """

        if not note_path or not note_path.strip():
            raise ValueError("note_path must not be empty or whitespace-only")
        if top_k < 1 or top_k > MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}, got {top_k}")

        if self.graph_store.note_meta_for(note_path) is None:
            return {"note_path": note_path, "found": False, "neighbors": []}

        edges = self.graph_store.edges_for(note_path)
        edges.sort(key=lambda e: e.composite, reverse=True)
        top_edges = edges[:top_k]

        other_paths = {e.dst if e.src == note_path else e.src for e in top_edges}
        meta_by_path = self.graph_store.note_meta_for_many(other_paths)

        neighbors = []
        for edge in top_edges:
            other = edge.dst if edge.src == note_path else edge.src
            other_meta = meta_by_path.get(other)
            if edge.src == note_path:
                links_to, linked_from = edge.link_src_to_dst, edge.link_dst_to_src
            else:
                links_to, linked_from = edge.link_dst_to_src, edge.link_src_to_dst
            neighbors.append(
                {
                    "path": other,
                    "title": other_meta["title"] if other_meta else other,
                    "composite": edge.composite,
                    "signals": {
                        "semantic": edge.semantic,
                        "link": edge.link,
                        "tag": edge.tag,
                        "comention": edge.comention,
                    },
                    "evidence": {
                        "links_to": links_to,
                        "linked_from": linked_from,
                        "shared_tags": edge.shared_tags,
                        "comention_count": edge.comention_count,
                    },
                }
            )

        return {"note_path": note_path, "found": True, "neighbors": neighbors}

    def connections(self, note_a: str, note_b: str) -> dict:
        """Return how closely two notes are associated, direct or via a path.

        Closeness is the direct edge's composite score when one exists;
        otherwise it's the product of composite scores along the shortest
        evidentiary path (an unbroken chain of nonzero relatedness), or 0.0
        if the two notes aren't connected in the graph at all.

        When no direct edge exists, this loads every edge in the vault to
        search for a path -- O(edge count) per call, with no caching between
        calls. Fine at typical vault scale; worth revisiting if it becomes a
        hot path on very large graphs.
        """

        if not note_a or not note_a.strip() or not note_b or not note_b.strip():
            raise ValueError("note_a and note_b must not be empty or whitespace-only")
        if note_a == note_b:
            raise ValueError("note_a and note_b must be different notes")

        found_a = self.graph_store.note_meta_for(note_a) is not None
        found_b = self.graph_store.note_meta_for(note_b) is not None

        direct = self.graph_store.edge_between(note_a, note_b)
        direct_payload = None
        if direct is not None:
            direct_payload = {
                "composite": direct.composite,
                "signals": {
                    "semantic": direct.semantic,
                    "link": direct.link,
                    "tag": direct.tag,
                    "comention": direct.comention,
                },
                "evidence": {
                    "shared_tags": direct.shared_tags,
                    "comention_count": direct.comention_count,
                    "link_src_to_dst": direct.link_src_to_dst,
                    "link_dst_to_src": direct.link_dst_to_src,
                },
            }

        connected = False
        closeness = 0.0
        path: list[str] | None = None
        path_edges: list[dict] = []

        if direct is not None:
            connected = True
            closeness = direct.composite
            path = [note_a, note_b]
        elif found_a and found_b:
            edges = self.graph_store.all_edges()
            g = build_nx_graph(edges)
            node_path = shortest_evidence_path(g, note_a, note_b)
            if node_path is not None:
                # Look hops up in the same edge snapshot used to build g and
                # find node_path, rather than re-querying the store per hop:
                # avoids both a redundant round trip per hop and a race where
                # a concurrent sync could delete a hop's edge between the
                # path search and a separate re-fetch.
                edge_lookup = {canonical_pair(e.src, e.dst): e for e in edges}
                connected = True
                path = node_path
                closeness = 1.0
                for src, dst in zip(node_path, node_path[1:]):
                    hop = edge_lookup[canonical_pair(src, dst)]
                    closeness *= hop.composite
                    path_edges.append(
                        {
                            "src": src,
                            "dst": dst,
                            "composite": hop.composite,
                            "signals": {
                                "semantic": hop.semantic,
                                "link": hop.link,
                                "tag": hop.tag,
                                "comention": hop.comention,
                            },
                        }
                    )

        return {
            "note_a": note_a,
            "note_b": note_b,
            "found_a": found_a,
            "found_b": found_b,
            "connected": connected,
            "direct_edge": direct_payload,
            "closeness": closeness,
            "path": path,
            "path_edges": path_edges,
        }

    def graph_map(self, min_score: float | None = None) -> dict:
        """Summarize the note graph as clusters, orphans, and bridge notes.

        Clusters come from greedy modularity community detection on the
        edge set thresholded at ``min_score`` (default: the configured
        ``graph_min_edge_score``); each is labeled by its most common member
        tag, falling back to its highest-degree note's title. Bridges are
        articulation points -- notes whose removal would split their
        neighborhood apart.

        Loads every note and edge in the vault to build the graph -- O(note
        count + edge count) per call, with no caching between calls. Fine at
        typical vault scale; worth revisiting if it becomes a hot path on
        very large graphs.
        """

        if min_score is not None and not (0.0 <= min_score <= 1.0):
            raise ValueError(f"min_score must be between 0.0 and 1.0, got {min_score}")
        threshold = min_score if min_score is not None else self.config.graph_min_edge_score

        all_meta = self.graph_store.all_note_meta()
        all_paths = {m["path"] for m in all_meta}
        meta_by_path = {m["path"]: m for m in all_meta}
        edges = self.graph_store.all_edges(min_composite=threshold)

        g = build_nx_graph(edges, min_score=threshold)
        for p in all_paths:
            if p not in g:
                g.add_node(p)

        raw_clusters = compute_clusters(g)
        orphans = sorted(p for p in all_paths if g.degree(p) == 0)

        clusters = []
        node_to_cluster: dict[str, int] = {}
        for cluster_id, members in enumerate(raw_clusters):
            if len(members) < 2:
                continue
            degrees = {m: g.degree(m) for m in members}
            hub = max(degrees, key=degrees.get)
            tag_counts: dict[str, int] = {}
            for m in members:
                for tag in meta_by_path.get(m, {}).get("tags", []):
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            label = (
                max(tag_counts, key=tag_counts.get)
                if tag_counts
                else meta_by_path.get(hub, {}).get("title", hub)
            )
            sorted_members = sorted(members, key=lambda m: degrees[m], reverse=True)
            top_tags = sorted(tag_counts, key=tag_counts.get, reverse=True)[:5]
            clusters.append(
                {
                    "id": cluster_id,
                    "label": label,
                    "size": len(members),
                    "notes": sorted_members[:25],
                    "top_tags": top_tags,
                    "hub": hub,
                }
            )
            for m in members:
                node_to_cluster[m] = cluster_id

        bridges = []
        for node in articulation_points(g):
            neighbor_clusters = sorted(
                {
                    node_to_cluster[n]
                    for n in g.neighbors(node)
                    if n in node_to_cluster
                }
            )
            bridges.append(
                {
                    "path": node,
                    "note": meta_by_path.get(node, {}).get("title", node),
                    "connects_clusters": neighbor_clusters,
                }
            )

        return {
            "note_count": len(all_paths),
            "edge_count": len(edges),
            "clusters": clusters,
            "orphans": orphans,
            "bridges": bridges,
        }

    def status(self) -> dict:
        """Return index/model runtime status for operational visibility."""

        graph_counts = self.graph_store.counts()
        return {
            "watch_enabled": self.config.watch_enabled,
            "max_context_chunks": self.config.max_context_chunks,
            "model": self.config.embedding_model,
            "index_size": self.keyword_store.count_chunks(),
            "last_sync_timestamp": self.sync_state.last_sync_timestamp(),
            "watcher_state": "enabled" if self.config.watch_enabled else "disabled",
            "last_tracked_files": len(self.sync_state.tracked_paths()),
            "model_available": self.embedder.health(),
            "graph_nodes": graph_counts["nodes"],
            "graph_edges": graph_counts["edges"],
            "graph_last_built": graph_counts["last_built"],
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
