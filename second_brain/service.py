"""High-level service API used by CLI and MCP layers."""

from __future__ import annotations

from dataclasses import asdict
import math

from second_brain.config import RagConfig
from second_brain.embedder import OllamaEmbedder
from second_brain.graph import (
    GraphStore,
    articulation_points,
    build_nx_graph,
    canonical_pair,
    compute_clusters,
    shortest_evidence_path,
)
from second_brain.indexer import Indexer
from second_brain.keyword_store import KeywordStore, matches_filters, parse_modified_since
from second_brain.models import RetrievalHit
from second_brain.retrieval import normalize_query, reciprocal_rank_fusion, validate_recency_boost
from second_brain.scanner import path_is_excluded
from second_brain.sync_state import SyncStateStore
from second_brain.vector_store import InMemoryVectorStore, QdrantVectorStore


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

    def search(
        self,
        query: str,
        filters: dict | None = None,
        top_k: int = 10,
        min_score: float | None = None,
        verbose: bool = True,
        recency_boost: float = 0.0,
    ) -> dict:
        """Return hybrid hits, retaining full metadata unless compact mode is requested."""

        normalized = normalize_query(query)
        if not normalized:
            raise ValueError("query must not be empty or whitespace-only")
        if top_k < 1 or top_k > MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}, got {top_k}")
        if min_score is not None:
            valid_number = isinstance(min_score, (int, float)) and not isinstance(min_score, bool)
            # Compare bounds before isfinite converts integers to floats; huge
            # integers must raise the same ValueError as other invalid cutoffs.
            if not valid_number or min_score < -1.0 or min_score > 1.0 or not math.isfinite(min_score):
                raise ValueError("min_score must be a finite cosine similarity in [-1.0, 1.0]")
        recency_boost = validate_recency_boost(recency_boost)
        effective_filters = dict(filters or {})
        if "exclude_status" not in effective_filters and self.config.exclude_status:
            effective_filters["exclude_status"] = self.config.exclude_status
        # Validate an override even when the vault currently has no search candidates.
        exclude_status = effective_filters.get("exclude_status", [])
        if not isinstance(exclude_status, list) or any(
            not isinstance(status, str) for status in exclude_status
        ):
            raise ValueError("filters['exclude_status'] must be an array of strings")
        if "modified_since" in effective_filters:
            parse_modified_since(effective_filters["modified_since"])
        query_vec = self.embedder.embed([normalized])[0]
        semantic_hits = self.vector_store.search(
            query_vec,
            limit=None if min_score is not None else top_k,
            metadata_filter=(
                (lambda metadata: matches_filters(metadata, effective_filters))
                if effective_filters
                else None
            ),
        )
        keyword_hits = self.keyword_store.search(
            normalized,
            limit=None if min_score is not None else top_k,
            filters=effective_filters,
        )
        if min_score is not None:
            # A keyword-only result has no cosine value that can satisfy this cutoff.
            semantic_hits = [
                hit
                for hit in semantic_hits
                if (hit.semantic_score if hit.semantic_score is not None else hit.score) >= min_score
            ]
            qualifying_ids = {hit.chunk_id for hit in semantic_hits}
            keyword_hits = [hit for hit in keyword_hits if hit.chunk_id in qualifying_ids]
        merged = reciprocal_rank_fusion(
            semantic_hits, keyword_hits, recency_boost=recency_boost
        )

        hits = []
        for hit in merged[:top_k]:
            if verbose:
                hits.append(
                    {
                        "chunk_id": hit.chunk_id,
                        "score": hit.score,
                        "semantic_score": hit.semantic_score,
                        "keyword_score": hit.keyword_score,
                        "source": hit.source,
                        "text": hit.text,
                        "metadata": hit.metadata,
                    }
                )
            else:
                metadata = hit.metadata
                hits.append(
                    {
                        "chunk_id": hit.chunk_id,
                        "score": hit.score,
                        "semantic_score": hit.semantic_score,
                        "keyword_score": hit.keyword_score,
                        "text": hit.text,
                        "path": metadata.get("path"),
                        "note_title": metadata.get("note_title"),
                        "heading_path": metadata.get("heading_path", "root"),
                    }
                )
        return {"query": query, "hits": hits}

    def query(
        self,
        query: str,
        filters: dict | None = None,
        top_k: int = 8,
        min_score: float | None = None,
        verbose: bool = True,
        recency_boost: float = 0.0,
    ) -> dict:
        """Return retrieved chunks and citations with optional compact metadata."""

        results = self.search(
            query=query,
            filters=filters,
            top_k=top_k,
            min_score=min_score,
            verbose=verbose,
            recency_boost=recency_boost,
        )
        citations = []
        for hit in results["hits"]:
            metadata = hit.get("metadata") or hit
            citations.append(
                {
                    "chunk_id": hit["chunk_id"],
                    "path": metadata.get("path"),
                    "heading_path": metadata.get("heading_path", "root"),
                }
            )

        return {
            "citations": citations,
            "chunks": results["hits"],
            # Preserve the debug field while making its rank-fusion semantics explicit.
            "debug_scores": [h["score"] for h in results["hits"]],
        }

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

    def related(self, note_path: str, top_k: int = 10, verbose: bool = True) -> dict:
        """Return this note's graph neighbors, ranked by composite score.

        Verbose mode includes per-signal breakdowns; compact mode omits them
        while retaining structural evidence.
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
            neighbor = {
                "path": other,
                "title": other_meta["title"] if other_meta else other,
                "composite": edge.composite,
                "evidence": {
                    "links_to": links_to,
                    "linked_from": linked_from,
                    "shared_tags": edge.shared_tags,
                    "comention_count": edge.comention_count,
                },
            }
            if verbose:
                neighbor["signals"] = {
                    "semantic": edge.semantic,
                    "link": edge.link,
                    "tag": edge.tag,
                    "comention": edge.comention,
                }
            neighbors.append(neighbor)

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

        Clusters come from greedy modularity community detection on a transient
        graph thresholded at ``min_score`` (default: ``graph_min_edge_score``).
        Paths matching ``exclude_globs`` are removed before candidate generation
        and scoring, including co-mention sources and semantic neighbor slots.
        The view uses stored note metadata/centroids without modifying the index.
        Each cluster uses its most common unused tag, then the hub title or
        path; numeric suffixes resolve any remaining label collisions.
        Bridges are articulation points -- notes whose removal would split
        their neighborhood apart.

        Recomputes candidates/scores on each call without caching. The cosine
        matrix needs O(n squared) memory and pair comparisons for n eligible
        centroids, so large vaults may see increased map latency.
        """

        if min_score is not None and not (0.0 <= min_score <= 1.0):
            raise ValueError(f"min_score must be between 0.0 and 1.0, got {min_score}")
        threshold = min_score if min_score is not None else self.config.graph_min_edge_score

        all_meta = [
            meta
            for meta in self.graph_store.all_note_meta()
            if not path_is_excluded(meta["path"], self.config.exclude_globs)
        ]
        all_paths = {m["path"] for m in all_meta}
        meta_by_path = {m["path"]: m for m in all_meta}
        edges = [
            edge
            for edge in self.indexer.graph_builder.edges_from_metadata(all_meta)
            if edge.composite >= threshold
        ]

        g = build_nx_graph(edges, min_score=threshold)
        for p in sorted(all_paths):
            if p not in g:
                g.add_node(p)

        raw_clusters = sorted(
            compute_clusters(g), key=lambda members: (-len(members), tuple(sorted(members)))
        )
        orphans = sorted(p for p in all_paths if g.degree(p) == 0)

        clusters = []
        used_labels: set[str] = set()
        node_to_cluster: dict[str, int] = {}
        for cluster_id, members in enumerate(raw_clusters):
            if len(members) < 2:
                continue
            degrees = {m: g.degree(m) for m in members}
            hub = min(members, key=lambda member: (-degrees[member], member))
            tag_counts: dict[str, int] = {}
            for m in members:
                for tag in meta_by_path.get(m, {}).get("tags", []):
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            ordered_tags = sorted(tag_counts, key=lambda tag: (-tag_counts[tag], tag.casefold(), tag))
            title = meta_by_path.get(hub, {}).get("title", "")
            candidates = [*ordered_tags, title or hub]
            label = next(
                (candidate for candidate in candidates if candidate.casefold() not in used_labels),
                title or hub,
            )
            base_label = label
            suffix = 2
            while label.casefold() in used_labels:
                label = f"{base_label} ({suffix})"
                suffix += 1
            used_labels.add(label.casefold())
            sorted_members = sorted(members, key=lambda member: (-degrees[member], member))
            top_tags = ordered_tags[:5]
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
