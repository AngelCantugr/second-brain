"""Note-association graph: pure scoring functions, storage, and builder.

Edges are undirected facts about pairs of notes, scored from four blended
signals (semantic similarity, wikilinks, shared tags, co-mentions) and
persisted alongside the keyword index so graph queries stay cheap. This
module intentionally never talks to the vector store directly -- note
centroid vectors are handed in by the indexer, which keeps both vector-store
backends (Qdrant and the in-memory test double) untouched.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from obsidian_rag.config import RagConfig
from obsidian_rag.models import Edge

_ONE_WAY_LINK_SCORE = 0.7
_MUTUAL_LINK_SCORE = 1.0


def normalize_link_title(link: str) -> str:
    """Normalize a raw wikilink target to the title-matching key.

    Mirrors ``obsidian_rag.keyword_store._chunk_links_target``: strip any
    ``#heading`` fragment, trim whitespace, and lowercase.
    """

    return link.split("#", 1)[0].strip().lower()


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors, 0.0 if either is degenerate."""

    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity matrix for a (n, dim) array of row vectors.

    Rows with zero norm produce zero similarity against every other row
    (rather than NaN from a division by zero).
    """

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    normalized = vectors / safe_norms
    normalized[norms.squeeze(-1) == 0.0] = 0.0
    return normalized @ normalized.T


def tag_jaccard(tags_a: list[str], tags_b: list[str]) -> float:
    """Jaccard similarity of two tag sets, case-insensitive. 0.0 if both empty."""

    set_a = {t.lower() for t in tags_a}
    set_b = {t.lower() for t in tags_b}
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def link_component(link_src_to_dst: bool, link_dst_to_src: bool) -> float:
    """Score the wikilink signal: mutual > one-way > none."""

    if link_src_to_dst and link_dst_to_src:
        return _MUTUAL_LINK_SCORE
    if link_src_to_dst or link_dst_to_src:
        return _ONE_WAY_LINK_SCORE
    return 0.0


def comention_component(count: int, cap: int) -> float:
    """Score the co-mention signal, saturating at ``cap`` mentions."""

    if cap <= 0:
        return 0.0
    return min(1.0, count / cap)


def composite_score(
    semantic: float,
    link: float,
    tag: float,
    comention: float,
    weights: tuple[float, float, float, float],
) -> float:
    """Weighted sum of the four normalized signal components."""

    w_semantic, w_link, w_tag, w_comention = weights
    return (
        w_semantic * semantic
        + w_link * link
        + w_tag * tag
        + w_comention * comention
    )


def comention_pairs(
    resolved_links: list[str], max_fanout: int
) -> set[tuple[str, str]]:
    """Canonical (min, max) path pairs co-mentioned by one note's outlinks.

    ``resolved_links`` are the *target note paths* a single note links to
    (already resolved from raw wikilink titles, deduplicated). If the
    mentioning note's fanout exceeds ``max_fanout`` it contributes no pairs,
    to keep hub/MOC notes from generating O(fanout^2) edges.
    """

    unique = sorted(set(resolved_links))
    if len(unique) < 2 or len(unique) > max_fanout:
        return set()
    pairs: set[tuple[str, str]] = set()
    for i, a in enumerate(unique):
        for b in unique[i + 1 :]:
            pairs.add((a, b) if a < b else (b, a))
    return pairs


def canonical_pair(a: str, b: str) -> tuple[str, str]:
    """Order two note paths so an undirected pair has one canonical key."""

    return (a, b) if a < b else (b, a)


def _encode_centroid(centroid: list[float] | None) -> tuple[bytes | None, int | None]:
    """Serialize a centroid vector to a float32 blob + its dimension."""

    if centroid is None:
        return None, None
    arr = np.asarray(centroid, dtype=np.float32)
    return arr.tobytes(), int(arr.shape[0])


def _decode_centroid(blob: bytes | None) -> list[float] | None:
    """Deserialize a float32 blob back to a plain list of floats."""

    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32).tolist()


def _row_to_edge(row: sqlite3.Row) -> Edge:
    """Build an ``Edge`` from an ``edges`` table row."""

    return Edge(
        src=row["src"],
        dst=row["dst"],
        semantic=row["semantic"],
        link=row["link"],
        tag=row["tag"],
        comention=row["comention"],
        comention_count=row["comention_count"],
        link_src_to_dst=bool(row["link_src_to_dst"]),
        link_dst_to_src=bool(row["link_dst_to_src"]),
        shared_tags=json.loads(row["shared_tags_json"]),
        composite=row["composite"],
    )


class GraphStore:
    """Owns the ``note_meta`` and ``edges`` tables inside the fts.sqlite DB.

    Follows the same connection and schema-creation conventions as
    ``KeywordStore``: a fresh per-call connection with ``sqlite3.Row`` row
    access, used as a context manager for auto-commit, and idempotent
    ``CREATE TABLE IF NOT EXISTS`` statements with no migration framework.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        """Create the graph tables if missing."""

        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS note_meta (
                    path TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    title_key TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    links_json TEXT NOT NULL,
                    centroid BLOB,
                    dim INTEGER,
                    updated_at REAL DEFAULT (strftime('%s','now'))
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_note_meta_title_key ON note_meta(title_key)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS edges (
                    src TEXT NOT NULL,
                    dst TEXT NOT NULL,
                    semantic REAL NOT NULL DEFAULT 0,
                    link REAL NOT NULL DEFAULT 0,
                    tag REAL NOT NULL DEFAULT 0,
                    comention REAL NOT NULL DEFAULT 0,
                    comention_count INTEGER NOT NULL DEFAULT 0,
                    link_src_to_dst INTEGER NOT NULL DEFAULT 0,
                    link_dst_to_src INTEGER NOT NULL DEFAULT 0,
                    shared_tags_json TEXT NOT NULL DEFAULT '[]',
                    composite REAL NOT NULL,
                    updated_at REAL DEFAULT (strftime('%s','now')),
                    PRIMARY KEY (src, dst)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_edges_composite ON edges(composite)"
            )

    # -- note_meta -----------------------------------------------------

    def upsert_note_meta(
        self,
        path: str,
        title: str,
        tags: list[str],
        links: list[str],
        centroid: list[float] | None,
    ) -> dict | None:
        """Insert/update a note's graph metadata; return the previous row (if any).

        The previous row is returned as a plain dict with ``tags``/``links``
        already JSON-decoded, so callers (the incremental builder) can diff
        old vs. new links/tags without a second read.
        """

        previous = self.note_meta_for(path)
        title_key = title.strip().lower()
        blob, dim = _encode_centroid(centroid)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO note_meta(path, title, title_key, tags_json, links_json, centroid, dim, updated_at)
                VALUES(?, ?, ?, ?, ?, ?, ?, strftime('%s','now'))
                ON CONFLICT(path) DO UPDATE SET
                    title=excluded.title,
                    title_key=excluded.title_key,
                    tags_json=excluded.tags_json,
                    links_json=excluded.links_json,
                    centroid=excluded.centroid,
                    dim=excluded.dim,
                    updated_at=excluded.updated_at
                """,
                (
                    path,
                    title,
                    title_key,
                    json.dumps(sorted(tags), sort_keys=True),
                    json.dumps(sorted(links), sort_keys=True),
                    blob,
                    dim,
                ),
            )
        return previous

    def note_meta_for(self, path: str) -> dict | None:
        """Return one note's graph metadata as a plain dict, or None."""

        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM note_meta WHERE path = ?", (path,)
            ).fetchone()
        if row is None:
            return None
        return {
            "path": row["path"],
            "title": row["title"],
            "title_key": row["title_key"],
            "tags": json.loads(row["tags_json"]),
            "links": json.loads(row["links_json"]),
            "centroid": _decode_centroid(row["centroid"]),
            "dim": row["dim"],
        }

    def delete_note_meta(self, path: str) -> dict | None:
        """Delete a note's graph metadata; return the row that was removed."""

        previous = self.note_meta_for(path)
        with self._connect() as conn:
            conn.execute("DELETE FROM note_meta WHERE path = ?", (path,))
        return previous

    def all_note_meta(self) -> list[dict]:
        """Return graph metadata for every indexed note."""

        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM note_meta").fetchall()
        return [
            {
                "path": row["path"],
                "title": row["title"],
                "title_key": row["title_key"],
                "tags": json.loads(row["tags_json"]),
                "links": json.loads(row["links_json"]),
                "centroid": _decode_centroid(row["centroid"]),
                "dim": row["dim"],
            }
            for row in rows
        ]

    def paths_by_title_key(self, title_key: str) -> list[str]:
        """Return note paths whose title matches ``title_key`` (may be >1)."""

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT path FROM note_meta WHERE title_key = ?", (title_key,)
            ).fetchall()
        return [row["path"] for row in rows]

    def all_centroids(self) -> tuple[list[str], np.ndarray]:
        """Return (paths, matrix) for every note that has a centroid."""

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT path, centroid FROM note_meta WHERE centroid IS NOT NULL"
            ).fetchall()
        paths = [row["path"] for row in rows]
        if not paths:
            return [], np.empty((0, 0), dtype=np.float32)
        vectors = np.stack(
            [np.frombuffer(row["centroid"], dtype=np.float32) for row in rows]
        )
        return paths, vectors

    def centroid_for(self, path: str) -> list[float] | None:
        """Return one note's centroid vector, or None if absent/not embedded."""

        with self._connect() as conn:
            row = conn.execute(
                "SELECT centroid FROM note_meta WHERE path = ?", (path,)
            ).fetchone()
        if row is None:
            return None
        return _decode_centroid(row["centroid"])

    # -- edges ----------------------------------------------------------

    def replace_edges_for(self, path: str, edges: list[Edge]) -> None:
        """Replace every edge touching ``path`` with the given edge list."""

        with self._connect() as conn:
            conn.execute(
                "DELETE FROM edges WHERE src = ? OR dst = ?", (path, path)
            )
            for edge in edges:
                self._insert_edge(conn, edge)

    def replace_all_edges(self, edges: list[Edge]) -> None:
        """Replace the entire edge set (used by a full rebuild)."""

        with self._connect() as conn:
            conn.execute("DELETE FROM edges")
            for edge in edges:
                self._insert_edge(conn, edge)

    def delete_edges_touching(self, paths: set[str]) -> None:
        """Delete every edge with either endpoint in ``paths``."""

        if not paths:
            return
        plist = list(paths)
        placeholders = ",".join("?" for _ in plist)
        with self._connect() as conn:
            conn.execute(
                f"DELETE FROM edges WHERE src IN ({placeholders}) OR dst IN ({placeholders})",
                plist + plist,
            )

    def insert_edges(self, edges: list[Edge]) -> None:
        """Insert/update edges without first clearing existing rows."""

        with self._connect() as conn:
            for edge in edges:
                self._insert_edge(conn, edge)

    @staticmethod
    def _insert_edge(conn: sqlite3.Connection, edge: Edge) -> None:
        conn.execute(
            """
            INSERT INTO edges(
                src, dst, semantic, link, tag, comention, comention_count,
                link_src_to_dst, link_dst_to_src, shared_tags_json, composite,
                updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, strftime('%s','now'))
            ON CONFLICT(src, dst) DO UPDATE SET
                semantic=excluded.semantic,
                link=excluded.link,
                tag=excluded.tag,
                comention=excluded.comention,
                comention_count=excluded.comention_count,
                link_src_to_dst=excluded.link_src_to_dst,
                link_dst_to_src=excluded.link_dst_to_src,
                shared_tags_json=excluded.shared_tags_json,
                composite=excluded.composite,
                updated_at=excluded.updated_at
            """,
            (
                edge.src,
                edge.dst,
                edge.semantic,
                edge.link,
                edge.tag,
                edge.comention,
                edge.comention_count,
                int(edge.link_src_to_dst),
                int(edge.link_dst_to_src),
                json.dumps(sorted(edge.shared_tags)),
                edge.composite,
            ),
        )

    def edges_for(self, path: str) -> list[Edge]:
        """Return every edge touching ``path``, unsorted."""

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM edges WHERE src = ? OR dst = ?", (path, path)
            ).fetchall()
        return [_row_to_edge(row) for row in rows]

    def edge_between(self, a: str, b: str) -> Edge | None:
        """Return the edge between two notes, if one exists."""

        src, dst = canonical_pair(a, b)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM edges WHERE src = ? AND dst = ?", (src, dst)
            ).fetchone()
        return _row_to_edge(row) if row is not None else None

    def all_edges(self, min_composite: float = 0.0) -> list[Edge]:
        """Return every edge with ``composite >= min_composite``."""

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM edges WHERE composite >= ?", (min_composite,)
            ).fetchall()
        return [_row_to_edge(row) for row in rows]

    def min_semantic_edge(self, path: str) -> float:
        """Return the lowest positive semantic score among ``path``'s edges.

        Used by the incremental builder's reverse-kNN heuristic: a changed
        note X should also be considered a candidate neighbor of any note Y
        whose weakest current semantic edge X could plausibly displace.
        Returns 0.0 when ``path`` has no semantic edges yet.
        """

        with self._connect() as conn:
            row = conn.execute(
                "SELECT MIN(semantic) AS m FROM edges "
                "WHERE (src = ? OR dst = ?) AND semantic > 0",
                (path, path),
            ).fetchone()
        return float(row["m"]) if row and row["m"] is not None else 0.0

    def min_semantic_edges(self) -> dict[str, float]:
        """Return every note's lowest positive semantic edge score, batched.

        Equivalent to calling ``min_semantic_edge`` for every note, but in
        one query instead of one round trip per note -- used by the
        incremental builder's reverse-kNN heuristic, which otherwise needs
        this value for every candidate note in the vault per changed note.
        """

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT src, dst, semantic FROM edges WHERE semantic > 0"
            ).fetchall()
        mins: dict[str, float] = {}
        for row in rows:
            semantic = float(row["semantic"])
            for node in (row["src"], row["dst"]):
                if node not in mins or semantic < mins[node]:
                    mins[node] = semantic
        return mins

    def note_meta_for_many(self, paths) -> dict[str, dict]:
        """Batch lookup of note_meta rows for a set of paths, keyed by path.

        One query instead of one round trip per path -- used where a caller
        needs metadata for several notes at once (e.g. ``rag.related``'s
        neighbor titles).
        """

        plist = list(paths)
        if not plist:
            return {}
        placeholders = ",".join("?" for _ in plist)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM note_meta WHERE path IN ({placeholders})", plist
            ).fetchall()
        return {
            row["path"]: {
                "path": row["path"],
                "title": row["title"],
                "title_key": row["title_key"],
                "tags": json.loads(row["tags_json"]),
                "links": json.loads(row["links_json"]),
                "centroid": _decode_centroid(row["centroid"]),
                "dim": row["dim"],
            }
            for row in rows
        }

    def counts(self) -> dict:
        """Return graph size stats for ``rag.status``."""

        with self._connect() as conn:
            nodes = conn.execute("SELECT COUNT(*) FROM note_meta").fetchone()[0]
            edges = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
            last_built = conn.execute("SELECT MAX(updated_at) FROM edges").fetchone()[0]
        return {
            "nodes": int(nodes),
            "edges": int(edges),
            "last_built": float(last_built) if last_built is not None else None,
        }


@dataclass
class _BuildContext:
    """Precomputed per-build state shared by candidate generation and scoring."""

    meta_by_path: dict[str, dict]
    title_index: dict[str, list[str]]
    resolved_links: dict[str, list[str]]
    backlinks_index: dict[str, set[str]]
    comention_counts: dict[tuple[str, str], int]
    paths: list[str]
    sims: np.ndarray
    path_to_idx: dict[str, int]


class GraphBuilder:
    """Computes note-association edges from ``GraphStore`` metadata.

    ``rebuild_full`` recomputes every edge from scratch (the correctness
    baseline, used for ``rag_sync(mode="full")``). ``update_for_changes``
    recomputes only the edges plausibly affected by a set of changed/deleted
    notes, using a reverse-kNN heuristic for the semantic signal (a superset
    approximation of exact mutual-kNN -- ``mode="full"`` remains the exact
    fallback).
    """

    def __init__(self, store: GraphStore, config: RagConfig) -> None:
        self.store = store
        self.config = config

    @property
    def _weights(self) -> tuple[float, float, float, float]:
        return (
            self.config.graph_weight_semantic,
            self.config.graph_weight_link,
            self.config.graph_weight_tag,
            self.config.graph_weight_comention,
        )

    def rebuild_full(self) -> dict:
        """Recompute every edge in the graph from current note_meta rows."""

        metas = self.store.all_note_meta()
        if len(metas) < 2:
            self.store.replace_all_edges([])
            return {"nodes": len(metas), "edges": 0, "edges_updated": 0}

        ctx = self._build_context(metas)
        candidate_pairs: set[tuple[str, str]] = set(ctx.comention_counts.keys())
        for path in ctx.meta_by_path:
            candidate_pairs |= self._candidates_for(path, ctx)

        edges = self._score_pairs(candidate_pairs, ctx)
        self.store.replace_all_edges(edges)
        return {"nodes": len(metas), "edges": len(edges), "edges_updated": len(edges)}

    def update_for_changes(
        self,
        changed: set[str],
        deleted: set[str],
        old_meta: dict[str, dict | None],
    ) -> dict:
        """Recompute edges affected by a set of changed/deleted note paths.

        ``old_meta`` maps each path in ``changed | deleted`` to its previous
        ``note_meta`` row (as returned by ``GraphStore.upsert_note_meta`` /
        ``delete_note_meta``), or ``None`` for a brand-new note. It is used
        to detect co-mention pairs that lost a contributing note and link
        targets that were dropped, neither of which is visible from the
        current (post-update) metadata alone.
        """

        if not changed and not deleted:
            counts = self.store.counts()
            return {
                "nodes": counts["nodes"],
                "edges": counts["edges"],
                "edges_updated": 0,
            }

        metas = self.store.all_note_meta()
        if len(metas) < 2:
            self.store.replace_all_edges([])
            return {"nodes": len(metas), "edges": 0, "edges_updated": 0}

        ctx = self._build_context(metas)
        affected = self._affected_paths(changed, deleted, old_meta, ctx)

        candidate_pairs: set[tuple[str, str]] = set()
        for path in affected:
            if path in ctx.meta_by_path:
                candidate_pairs |= self._candidates_for(path, ctx)
        for pair in ctx.comention_counts:
            if pair[0] in affected or pair[1] in affected:
                candidate_pairs.add(pair)

        self.store.delete_edges_touching(affected | deleted)
        edges = self._score_pairs(candidate_pairs, ctx)
        self.store.insert_edges(edges)

        counts = self.store.counts()
        return {
            "nodes": counts["nodes"],
            "edges": counts["edges"],
            "edges_updated": len(edges),
        }

    # -- context building ------------------------------------------------

    def _build_context(self, metas: list[dict]) -> _BuildContext:
        meta_by_path = {m["path"]: m for m in metas}

        title_index: dict[str, list[str]] = {}
        for m in metas:
            title_index.setdefault(m["title_key"], []).append(m["path"])

        resolved_links = {
            m["path"]: self._resolve_titles(m["links"], title_index, exclude=m["path"])
            for m in metas
        }

        backlinks_index: dict[str, set[str]] = {}
        for source, targets in resolved_links.items():
            for target in targets:
                backlinks_index.setdefault(target, set()).add(source)

        comention_counts: dict[tuple[str, str], int] = {}
        for links in resolved_links.values():
            for pair in comention_pairs(links, self.config.graph_comention_max_fanout):
                comention_counts[pair] = comention_counts.get(pair, 0) + 1

        paths, matrix = self.store.all_centroids()
        sims = cosine_matrix(matrix) if paths else np.empty((0, 0))
        path_to_idx = {p: i for i, p in enumerate(paths)}

        return _BuildContext(
            meta_by_path=meta_by_path,
            title_index=title_index,
            resolved_links=resolved_links,
            backlinks_index=backlinks_index,
            comention_counts=comention_counts,
            paths=paths,
            sims=sims,
            path_to_idx=path_to_idx,
        )

    @staticmethod
    def _resolve_titles(
        link_titles: list[str], title_index: dict[str, list[str]], exclude: str
    ) -> list[str]:
        resolved: set[str] = set()
        for link_title in link_titles:
            key = normalize_link_title(link_title)
            for target_path in title_index.get(key, []):
                if target_path != exclude:
                    resolved.add(target_path)
        return sorted(resolved)

    # -- candidate generation ---------------------------------------------

    def _semantic_candidates(self, path: str, ctx: _BuildContext) -> list[str]:
        if path not in ctx.path_to_idx:
            return []
        i = ctx.path_to_idx[path]
        row = ctx.sims[i]
        order = np.argsort(-row)
        result: list[str] = []
        for j in order:
            if j == i:
                continue
            sim = float(row[j])
            if sim < self.config.graph_semantic_min:
                break
            result.append(ctx.paths[j])
            if len(result) >= self.config.graph_knn_k:
                break
        return result

    def _candidates_for(self, path: str, ctx: _BuildContext) -> set[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        for target in self._semantic_candidates(path, ctx):
            pairs.add(canonical_pair(path, target))
        for target in ctx.resolved_links.get(path, []):
            pairs.add(canonical_pair(path, target))
        for source in ctx.backlinks_index.get(path, []):
            pairs.add(canonical_pair(path, source))
        return pairs

    def _affected_paths(
        self,
        changed: set[str],
        deleted: set[str],
        old_meta: dict[str, dict | None],
        ctx: _BuildContext,
    ) -> set[str]:
        affected: set[str] = set(changed)
        max_fanout = self.config.graph_comention_max_fanout

        for path in changed | deleted:
            old = old_meta.get(path)
            if old:
                old_targets = self._resolve_titles(
                    old.get("links", []), ctx.title_index, exclude=path
                )
                affected.update(old_targets)
                for a, b in comention_pairs(old_targets, max_fanout):
                    affected.add(a)
                    affected.add(b)

            new_targets = ctx.resolved_links.get(path, [])
            affected.update(new_targets)
            for a, b in comention_pairs(new_targets, max_fanout):
                affected.add(a)
                affected.add(b)

            affected.update(ctx.backlinks_index.get(path, []))

        # Reverse-kNN heuristic: a changed note may now belong in another
        # note's semantic neighborhood even if it wasn't there before. Fetch
        # every note's current weakest semantic edge in one query up front
        # instead of one round trip per (changed note, candidate) pair --
        # otherwise this loop is O(len(changed) * vault size) individual
        # SQLite connections, which dominates every incremental sync.
        semantic_min = self.config.graph_semantic_min
        min_semantic_edges = (
            self.store.min_semantic_edges() if any(p in ctx.path_to_idx for p in changed) else {}
        )
        for path in changed:
            if path not in ctx.path_to_idx:
                continue
            i = ctx.path_to_idx[path]
            row = ctx.sims[i]
            for j, sim in enumerate(row):
                if j == i or sim < semantic_min:
                    continue
                candidate = ctx.paths[j]
                threshold = max(semantic_min, min_semantic_edges.get(candidate, 0.0))
                if float(sim) >= threshold:
                    affected.add(candidate)

        return affected

    # -- scoring -----------------------------------------------------------

    def _score_pairs(
        self, pairs: set[tuple[str, str]], ctx: _BuildContext
    ) -> list[Edge]:
        edges: list[Edge] = []
        for src, dst in pairs:
            edge = self._score_pair(src, dst, ctx)
            if edge is not None:
                edges.append(edge)
        return edges

    def _score_pair(self, src: str, dst: str, ctx: _BuildContext) -> Edge | None:
        meta_a = ctx.meta_by_path.get(src)
        meta_b = ctx.meta_by_path.get(dst)
        if meta_a is None or meta_b is None:
            return None

        semantic = 0.0
        if src in ctx.path_to_idx and dst in ctx.path_to_idx:
            semantic = max(
                0.0, float(ctx.sims[ctx.path_to_idx[src], ctx.path_to_idx[dst]])
            )

        link_src_to_dst = dst in ctx.resolved_links.get(src, [])
        link_dst_to_src = src in ctx.resolved_links.get(dst, [])
        link = link_component(link_src_to_dst, link_dst_to_src)

        shared_tags = sorted(
            {t.lower() for t in meta_a["tags"]} & {t.lower() for t in meta_b["tags"]}
        )
        tag = tag_jaccard(meta_a["tags"], meta_b["tags"])

        count = ctx.comention_counts.get((src, dst), 0)
        comention = comention_component(count, self.config.graph_comention_cap)

        composite = composite_score(semantic, link, tag, comention, self._weights)
        has_structural_evidence = link > 0 or count > 0
        if composite < self.config.graph_min_edge_score and not has_structural_evidence:
            return None

        return Edge(
            src=src,
            dst=dst,
            semantic=semantic,
            link=link,
            tag=tag,
            comention=comention,
            comention_count=count,
            link_src_to_dst=link_src_to_dst,
            link_dst_to_src=link_dst_to_src,
            shared_tags=shared_tags,
            composite=composite,
        )


# -- query helpers (networkx lives only here, never on the sync path) -----


def build_nx_graph(edges: list[Edge], min_score: float = 0.0):
    """Build an undirected, composite-weighted graph from persisted edges."""

    import networkx as nx

    g = nx.Graph()
    for edge in edges:
        if edge.composite >= min_score:
            g.add_edge(edge.src, edge.dst, weight=edge.composite, composite=edge.composite)
    return g


def compute_clusters(g) -> list[set[str]]:
    """Detect note neighborhoods via greedy modularity communities.

    Isolated (edgeless) nodes each form their own singleton community;
    callers typically filter those out as orphans rather than clusters.
    """

    from networkx.algorithms.community import greedy_modularity_communities

    if g.number_of_edges() == 0:
        return [{n} for n in g.nodes]
    communities = greedy_modularity_communities(g, weight="composite")
    return [set(c) for c in communities]


def shortest_evidence_path(g, a: str, b: str) -> list[str] | None:
    """Shortest path between two notes, weighting stronger edges as shorter.

    Edge weight is ``max(0.05, 1 - composite)`` so a high-composite edge
    contributes little "distance" and a weak one contributes a lot -- the
    shortest path is the strongest evidentiary chain, not just fewest hops.
    """

    import networkx as nx

    if a not in g or b not in g:
        return None

    def _distance(_u, _v, data):
        return max(0.05, 1.0 - data.get("composite", 0.0))

    try:
        return nx.dijkstra_path(g, a, b, weight=_distance)
    except nx.NetworkXNoPath:
        return None


def articulation_points(g) -> list[str]:
    """Notes whose removal would split the (thresholded) graph apart."""

    import networkx as nx

    if g.number_of_nodes() < 3:
        return []
    return list(nx.articulation_points(g))
