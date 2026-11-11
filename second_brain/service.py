"""High-level service API used by CLI and MCP layers."""

from __future__ import annotations

import hashlib
import itertools
import math
import threading
from dataclasses import asdict

from second_brain.code_graph import EDGE_TYPES, LINKS
from second_brain.config import RagConfig
from second_brain.doc_roles import DOC_ROLES, HISTORY, RECORD, REFERENCE, doc_role
from second_brain.drift import (
    DANGLING,
    FILE,
    LEVELS,
    STALE,
    STATUSES,
    SYMBOL,
    mention_level,
    record_summaries,
    rename_candidates,
)
from second_brain.embedder import OllamaEmbedder
from second_brain.graph import (
    GraphStore,
    articulation_points,
    build_nx_graph,
    canonical_pair,
    compute_clusters,
    ghost_targets,
    shortest_evidence_path,
    vault_metas,
)
from second_brain.indexer import Indexer
from second_brain.keyword_store import (
    KeywordStore,
    matches_filters,
    parse_modified_since,
)
from second_brain.models import RetrievalHit
from second_brain.retrieval import (
    apply_code_source_weight,
    apply_graph_boost,
    graph_boost_seeds,
    normalize_query,
    reciprocal_rank_fusion,
    validate_code_source_weight,
    validate_graph_boost,
    validate_recency_boost,
)
from second_brain.routing import build_repo_terms, repo_intent_terms
from second_brain.scanner import (
    is_eligible_source_path,
    path_is_excluded,
)
from second_brain.sources import (
    VAULT_SOURCE,
    is_eligible_code_source_path,
    iter_all_source_files,
    split_code_source_path,
)
from second_brain.sync_state import SyncStateStore
from second_brain.vector_store import InMemoryVectorStore, QdrantVectorStore

MAX_TOP_K = 50
# Each side of a drift pair is excerpted, not returned whole: enough for an
# agent to judge, short enough that 20 pairs fit in one tool result.
DRIFT_EXCERPT_CHARS = 1500
DRIFT_NOTE_EXCERPTS = 2
MAX_GHOSTS = 25
# Distinct top notes whose graph neighbors graph_boost lifts.
GRAPH_BOOST_SEEDS = 5


def _excerpt_around(text: str, needle: str, size: int = DRIFT_EXCERPT_CHARS) -> str:
    """Up to ``size`` characters of ``text`` centered on the first ``needle``.

    A chunk can run past ``size``; cutting its prefix would drop the mention
    the chunk was chosen for. Cut ends are marked with an ellipsis.
    """

    if len(text) <= size:
        return text
    at = text.find(needle)
    start = max(0, min(at - (size - len(needle)) // 2, len(text) - size))
    end = start + size
    return ("…" if start else "") + text[start:end] + ("…" if end < len(text) else "")


class RagService:
    """Facade around indexing, retrieval, and health/status endpoints."""

    def __init__(self, config: RagConfig, use_in_memory_vector: bool = False) -> None:
        self.config = config
        self.embedder = OllamaEmbedder(
            config.ollama_url, config.embedding_model, batch_size=config.embed_batch_size
        )
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
        # Built on first use and dropped by every sync; see _repo_terms().
        self._cached_repo_terms: frozenset[str] | None = None
        # The watcher syncs on its own thread while tool calls search on
        # theirs; the local Qdrant client is shared, so writes and vector
        # reads take turns. Reentrant because query() calls search().
        self._store_lock = threading.RLock()
        # Set by VaultWatcher.start(); None when no watcher runs in this process.
        self.watcher = None

    def sync(self, mode: str = "incremental", file_path: str | None = None) -> dict:
        """Trigger synchronization from vault files into indexes."""

        with self._store_lock:
            try:
                result = self.indexer.sync(mode=mode, file_path=file_path)
            finally:
                self._cached_repo_terms = None
        return asdict(result)

    def _repo_terms(self) -> frozenset[str]:
        """Return the code-source names that route a query to the repo.

        Every sync, including the watcher's, goes through ``sync``, which
        clears the cache, so the terms follow renamed files and symbols.
        """

        # Under the lock sync() holds while it clears the cache: a build that
        # read the index before a watcher sync must not store its result
        # after that sync's invalidation.
        with self._store_lock:
            terms = self._cached_repo_terms
            if terms is None:
                terms = build_repo_terms(self.keyword_store.iter_chunk_metadata_and_text())
                self._cached_repo_terms = terms
            return terms

    def search(
        self,
        query: str,
        filters: dict | None = None,
        top_k: int = 10,
        min_score: float | None = None,
        verbose: bool = True,
        recency_boost: float = 0.0,
        graph_boost: float = 0.0,
        code_source_weight: float | None = None,
    ) -> dict:
        """Return hybrid hits, retaining full metadata unless compact mode is requested.

        ``code_source_weight`` overrides the configured value for this call.
        It scales code-source hits only when the query names nothing
        distinctive in a code source (see ``second_brain.routing``).
        """

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
        graph_boost = validate_graph_boost(graph_boost)
        code_source_weight = validate_code_source_weight(
            self.config.code_source_weight if code_source_weight is None else code_source_weight
        )
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
        metadata_filter = self._metadata_filter(effective_filters)
        query_vec = self.embedder.embed([normalized])[0]
        # Both reads sit under one lock: a watcher sync between them would
        # replace a note's chunks, and the two legs would then return the old
        # and new text of the same passage under different chunk ids.
        with self._store_lock:
            semantic_hits = self.vector_store.search(
                query_vec,
                limit=None if min_score is not None else top_k,
                metadata_filter=metadata_filter,
            )
            keyword_hits = self.keyword_store.search(
                normalized,
                limit=None if min_score is not None else top_k,
                filters=effective_filters,
                predicate=metadata_filter,
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
        if (
            code_source_weight < 1.0
            and self.config.code_sources
            and not repo_intent_terms(normalized, self._repo_terms())
        ):
            merged = apply_code_source_weight(merged, code_source_weight)
        if graph_boost and self.config.graph_enabled:
            seeds = graph_boost_seeds(merged, GRAPH_BOOST_SEEDS)
            merged = apply_graph_boost(
                merged, self._neighbor_strength(seeds), graph_boost=graph_boost
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

    def _neighbor_strength(self, seeds: list[str]) -> dict[str, float]:
        """Map each non-seed graph neighbor of ``seeds`` to its strongest edge composite.

        Seeds are not lifted even when linked to each other: on the golden
        eval, letting linked seeds lift each other reshuffled the top ranks
        and cost MRR, while lifting only lower-ranked neighbors did not.
        """

        seed_set = set(seeds)
        strength: dict[str, float] = {}
        for edge in self.graph_store.edges_for_many(seed_set):
            for seed, other in ((edge.src, edge.dst), (edge.dst, edge.src)):
                if seed in seed_set and other not in seed_set:
                    strength[other] = max(strength.get(other, 0.0), edge.composite)
        return strength

    def query(
        self,
        query: str,
        filters: dict | None = None,
        top_k: int = 8,
        min_score: float | None = None,
        verbose: bool = True,
        recency_boost: float = 0.0,
        graph_boost: float = 0.0,
        code_source_weight: float | None = None,
    ) -> dict:
        """Return retrieved chunks and citations with optional compact metadata."""

        results = self.search(
            query=query,
            filters=filters,
            top_k=top_k,
            min_score=min_score,
            verbose=verbose,
            recency_boost=recency_boost,
            graph_boost=graph_boost,
            code_source_weight=code_source_weight,
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
        # One lock for both reads, as in search(): a watcher sync between
        # them would list a note's old and new chunk ids together.
        with self._store_lock:
            keyword_hits = self.keyword_store.chunks_by_path(note_path)
            vector_hits = self.vector_store.get_by_path(note_path)
        for hit in [*keyword_hits, *vector_hits]:
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

    def related(
        self,
        note_path: str,
        top_k: int = 10,
        verbose: bool = True,
        edge_types: list[str] | None = None,
    ) -> dict:
        """Return this note's graph neighbors, ranked by composite score.

        Verbose mode includes per-signal breakdowns; compact mode omits them
        while retaining structural evidence. ``edge_types`` keeps only edges
        of those types (``links``, ``mentions_code``, ``imports``).
        """

        if not note_path or not note_path.strip():
            raise ValueError("note_path must not be empty or whitespace-only")
        if top_k < 1 or top_k > MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}, got {top_k}")
        if edge_types is not None and (
            not isinstance(edge_types, list)
            or not edge_types
            or any(t not in EDGE_TYPES for t in edge_types)
        ):
            raise ValueError(f"edge_types must be a non-empty list drawn from {list(EDGE_TYPES)}")

        if self.graph_store.note_meta_for(note_path) is None:
            return {"note_path": note_path, "found": False, "neighbors": []}

        edges = self.graph_store.edges_for(note_path)
        if edge_types is not None:
            edges = [edge for edge in edges if edge.type in edge_types]
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
                "type": edge.type,
                "composite": edge.composite,
                "evidence": {
                    "links_to": links_to,
                    "linked_from": linked_from,
                    "shared_tags": edge.shared_tags,
                    "comention_count": edge.comention_count,
                },
            }
            if edge.type != LINKS:
                # What matched: the inline code, path or import, as written.
                neighbor["evidence"]["code_refs"] = edge.code_refs
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
                "type": direct.type,
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
            if direct.type != LINKS:
                direct_payload["evidence"]["code_refs"] = direct.code_refs

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
                for src, dst in itertools.pairwise(node_path):
                    hop = edge_lookup[canonical_pair(src, dst)]
                    closeness *= hop.composite
                    path_edges.append(
                        {
                            "src": src,
                            "dst": dst,
                            "type": hop.type,
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

    def _metadata_filter(self, filters: dict):
        """One predicate for both retrieval legs, or None when nothing filters.

        ``doc_role`` is resolved here rather than in ``matches_filters``: a role
        comes from the configured globs, not from stored chunk metadata. Code
        chunks are not docs and match no role.
        """

        rest = {key: value for key, value in filters.items() if key != "doc_role"}
        if "doc_role" not in filters:
            return (lambda metadata: matches_filters(metadata, rest)) if rest else None
        expected = filters["doc_role"]
        roles = [expected] if isinstance(expected, str) else expected
        if not isinstance(roles, list) or not roles or any(role not in DOC_ROLES for role in roles):
            raise ValueError(
                f"filters['doc_role'] must be a role or a list of roles from {list(DOC_ROLES)}"
            )
        role_by_path: dict[str, str] = {}

        def keep(metadata: dict) -> bool:
            if metadata.get("kind") == "code" or not matches_filters(metadata, rest):
                return False
            path = metadata.get("path", "")
            if path not in role_by_path:
                role_by_path[path] = doc_role(path, self.config)
            return role_by_path[path] in roles

        return keep

    def drift(
        self,
        status: list[str] | None = None,
        note_path: str | None = None,
        repo: str | None = None,
        levels: list[str] | None = None,
        limit: int = 20,
        roles: list[str] | None = None,
    ) -> dict:
        """Return note/code pairs that may have drifted apart, with both texts.

        A pair is ``stale`` when the code a note mentions changed after the
        note last did, and ``dangling`` when a mention that used to resolve
        no longer does. The server does not judge whether the texts disagree:
        it returns the note excerpts around the mention and the code's
        outline so the caller can. ``status`` defaults to both; ``ok`` lists
        tracked pairs with no change. ``note_path`` and ``repo`` narrow the
        pairs to one note or one code source. ``levels`` keeps mentions of a
        ``symbol``, a ``file`` or a whole ``repo``; repo mentions are left out
        by default because the README they resolve to changes constantly.

        ``roles`` picks which docs' pairs are listed, by ``doc_roles``:
        ``reference`` (the default: living docs, where a change means the
        doc may be out of date) and ``record`` (designs, plans, specs). Records
        are also summarized one row each in ``records``, most moved first,
        whatever ``roles`` says. ``history`` docs (changelogs) are not tracked.
        """

        statuses = [STALE, DANGLING] if status is None else status
        if not isinstance(statuses, list) or not statuses or any(s not in STATUSES for s in statuses):
            raise ValueError(f"status must be a non-empty list drawn from {list(STATUSES)}")
        wanted_levels = [SYMBOL, FILE] if levels is None else levels
        if not isinstance(wanted_levels, list) or not wanted_levels or any(
            level not in LEVELS for level in wanted_levels
        ):
            raise ValueError(f"levels must be a non-empty list drawn from {list(LEVELS)}")
        if limit < 1 or limit > MAX_TOP_K:
            raise ValueError(f"limit must be between 1 and {MAX_TOP_K}, got {limit}")
        wanted_roles = [REFERENCE] if roles is None else roles
        listable = [role for role in DOC_ROLES if role != HISTORY]
        if not isinstance(wanted_roles, list) or not wanted_roles or any(
            role not in listable for role in wanted_roles
        ):
            raise ValueError(f"roles must be a non-empty list drawn from {listable}")

        # Roles come from the current globs, so a changed glob applies at once;
        # history rows are also dropped when a sync writes them.
        stored = list(self.graph_store.code_mentions().values())
        role_of = {src: doc_role(src, self.config) for src in {state.src for state in stored}}
        mentions = [
            state
            for state in stored
            if role_of[state.src] != HISTORY
            and (note_path is None or state.src == note_path)
            and (repo is None or split_code_source_path(state.target)[0] == repo)
        ]
        level_of = {(s.src, s.mention): mention_level(s) for s in mentions}
        counts = {name: 0 for name in STATUSES}
        for state in mentions:
            if role_of[state.src] in wanted_roles:
                counts[state.status] += 1
        records = record_summaries(
            (state for state in mentions if role_of[state.src] == RECORD), wanted_levels
        )
        selected = [
            state
            for state in mentions
            if state.status in statuses
            and role_of[state.src] in wanted_roles
            and level_of[(state.src, state.mention)] in wanted_levels
        ]
        # Dangling first (a name that points nowhere is drift for sure), then
        # symbol mentions (a body changed under that exact name) before file
        # ones, then the most recent code change or loss.
        selected.sort(
            key=lambda s: (
                s.status != DANGLING,
                LEVELS.index(level_of[(s.src, s.mention)]),
                -(s.lost_at or s.changed_at or 0.0),
                s.src,
                s.mention,
            )
        )
        page = selected[:limit]

        metas = self.graph_store.all_note_meta() if any(s.status == DANGLING for s in page) else []
        meta_by_path = self.graph_store.note_meta_for_many({s.src for s in page} | {s.target for s in page})
        sources = {source.name: source.path for source in self.config.code_sources}
        pairs = []
        for state in page:
            note_meta = meta_by_path.get(state.src) or {}
            target_meta = meta_by_path.get(state.target)
            note_chunks = self.keyword_store.chunks_by_path(state.src)
            excerpts = [
                _excerpt_around(chunk.text, state.mention)
                for chunk in note_chunks
                if state.mention in chunk.text
            ]
            code = {
                "path": state.target,
                "symbol": state.symbol,
                "exists": target_meta is not None,
                "changed_at": state.changed_at,
                "mtime": (target_meta or {}).get("code", {}).get("mtime"),
            }
            repo_name, rel = split_code_source_path(state.target)
            if repo_name in sources:
                code["local_path"] = str(sources[repo_name] / rel)
            if target_meta is not None:
                code.update(self._drift_code_text(state.target, state.symbol))
            evidence: dict = {
                "mention": state.mention,
                "level": level_of[(state.src, state.mention)],
                "role": role_of[state.src],
            }
            if state.status == DANGLING:
                evidence["reason"] = state.reason
                evidence["lost_at"] = state.lost_at
                evidence["candidates"] = rename_candidates(state, metas)
            pairs.append(
                {
                    "status": state.status,
                    "note": {
                        "path": state.src,
                        "title": note_meta.get("title", state.src),
                        "mtime": note_meta.get("code", {}).get("mtime"),
                        "excerpts": (
                            excerpts or [c.text[:DRIFT_EXCERPT_CHARS] for c in note_chunks[:1]]
                        )[:DRIFT_NOTE_EXCERPTS],
                    },
                    "code": code,
                    "evidence": evidence,
                }
            )
        return {
            "counts": counts,
            "total": len(selected),
            "pairs": pairs,
            "record_total": len(records),
            "records": records[:limit],
        }

    def _drift_code_text(self, path: str, symbol: str | None) -> dict:
        """The indexed outline of a symbol, or a file's first chunk, with its line span.

        A symbol that is gone (a dangling mention whose file remains) has no
        text: another declaration's outline under its name would mislead.
        """

        chunks = self.keyword_store.chunks_by_path(path)
        if symbol is not None:
            chunks = [c for c in chunks if c.metadata.get("symbol") == symbol]
        if not chunks:
            return {"text": None}
        chunk = chunks[0]
        found: dict = {"text": chunk.text[:DRIFT_EXCERPT_CHARS]}
        for key in ("start_line", "end_line"):
            if key in chunk.metadata:
                found[key] = chunk.metadata[key]
        return found

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
        their neighborhood apart. Ghosts are wikilink targets that match no
        eligible note, ranked by how many notes link to them: notes the vault
        keeps referencing but has not written (or has excluded).

        Recomputes candidates/scores on each call without caching. The cosine
        matrix needs O(n squared) memory and pair comparisons for n eligible
        centroids, so large vaults may see increased map latency.
        """

        if min_score is not None and not (0.0 <= min_score <= 1.0):
            raise ValueError(f"min_score must be between 0.0 and 1.0, got {min_score}")
        threshold = min_score if min_score is not None else self.config.graph_min_edge_score

        all_meta = [
            meta
            for meta in vault_metas(self.graph_store.all_note_meta())
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
            "ghosts": (
                ghost_targets(all_meta, limit=MAX_GHOSTS) if self.config.graph_enabled else []
            ),
        }

    def status(self) -> dict:
        """Return runtime, index, watcher, graph, and read-only staleness status.

        ``stale_files`` counts eligible checkpointed files on disk whose UTF-8
        content hash or mtime changed, or whose checkpoint needs recovery or identity
        migration; unreadable checkpointed files count as stale. ``untracked_files``
        counts eligible files without a successful checkpoint, while
        ``missing_files`` counts eligible checkpointed or pending paths absent from
        disk. Eligibility follows scanner rules, including hidden-path and
        configured-glob exclusions. ``watcher_last_event`` is the last observed
        eligible file event time in Unix epoch seconds, or ``None`` before an
        event; it is not a heartbeat. Status reads and hashes tracked files but does
        not sync or mutate note or sync-state data. Incremental sync may leave an
        mtime-only warning until a full sync refreshes recorded mtimes.
        """

        graph_counts = self.graph_store.counts()
        tracked_state = self.sync_state.tracked_file_state()
        disk_files, missing_sources = iter_all_source_files(self.config)
        disk_paths = {file.rel_path: file.abs_path for file in disk_files}
        eligible_tracked = {
            path: state
            for path, state in tracked_state.items()
            if self._eligible_tracked_path(path, missing_sources)
        }
        stale_files = 0
        for relative_path, path in disk_paths.items():
            tracked = eligible_tracked.get(relative_path)
            if tracked is None or tracked.content_hash is None:
                # Pending-only paths have no successful checkpoint: present ones
                # remain untracked, while absent ones are reported as missing below.
                continue
            try:
                content = path.read_text(encoding="utf-8")
                content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                mtime = path.stat().st_mtime
            except (OSError, UnicodeError):
                # Do not report an unreadable candidate as clean or expose file data.
                stale_files += 1
                continue
            if (
                tracked.requires_reindex
                or content_hash != tracked.content_hash
                or mtime != tracked.mtime
            ):
                stale_files += 1
        checkpointed_paths = {
            path
            for path, state in eligible_tracked.items()
            if state.content_hash is not None
        }
        untracked_files = len(disk_paths.keys() - checkpointed_paths)
        missing_files = len(eligible_tracked.keys() - disk_paths.keys())
        return {
            "watch_enabled": self.config.watch_enabled,
            "max_context_chunks": self.config.max_context_chunks,
            "model": self.config.embedding_model,
            "index_size": self.keyword_store.count_chunks(),
            "last_sync_timestamp": self.sync_state.last_sync_timestamp(),
            "watcher_state": "enabled" if self.config.watch_enabled else "disabled",
            "watcher_running": self.watcher is not None and self.watcher.running,
            "last_tracked_files": len(tracked_state),
            "stale_files": stale_files,
            "untracked_files": untracked_files,
            "missing_files": missing_files,
            "watcher_last_event": self.sync_state.watcher_last_event(),
            "model_available": self.embedder.health(),
            "graph_nodes": graph_counts["nodes"],
            "graph_edges": graph_counts["edges"],
            "graph_last_built": graph_counts["last_built"],
        }

    def _eligible_tracked_path(self, path: str, missing_sources: dict[str, str]) -> bool:
        """Apply scanner eligibility rules to paths already stored in sync state.

        A code-source path is judged by its own source's rules. One from a
        source that is no longer configured, or whose root is missing, is not
        counted: sync keeps the second and deletes the first. A source whose
        listing failed counts as missing.
        """

        repo_path = split_code_source_path(path)
        if repo_path is None:
            # An unavailable vault is like a missing code-source root: its
            # notes are kept by sync, so they are not reported as missing.
            return VAULT_SOURCE not in missing_sources and is_eligible_source_path(
                path, self.config.exclude_globs, self.config.include_extensions
            )
        repo, rel = repo_path
        source = next((s for s in self.config.code_sources if s.name == repo), None)
        return (
            source is not None
            and repo not in missing_sources
            and is_eligible_code_source_path(rel, source)
        )

    def health(self) -> dict:
        """Return health checks for vector store, embedder, and keyword db."""

        qdrant_ok = True
        try:
            if hasattr(self.vector_store, "client"):
                self.vector_store.client.get_collections()
        # Any client failure means "not healthy"; health() must report, not raise.
        except Exception:  # noqa: BLE001
            qdrant_ok = False

        return {
            "qdrant": qdrant_ok,
            "ollama": self.embedder.health(),
            "fts": self.config.fts_path.exists(),
        }
