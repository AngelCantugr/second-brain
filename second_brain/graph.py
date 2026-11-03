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
import posixpath
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from second_brain.code_graph import code_edges
from second_brain.config import RagConfig
from second_brain.models import Edge
from second_brain.sources import is_code_source_path

_ONE_WAY_LINK_SCORE = 0.7
_MUTUAL_LINK_SCORE = 1.0
# Below this many embedded notes the mean is dominated by each note itself:
# with two notes, centering always makes them exact opposites.
_MIN_CENTERING_NOTES = 10
_VAULT_MEAN_KEY = "centroid_mean"


# Embeds of these are files, not notes, so an unmatched one is not a ghost.
_ATTACHMENT_SUFFIXES = frozenset(
    {
        "png", "jpg", "jpeg", "gif", "bmp", "svg", "webp", "avif", "heic",
        "pdf", "mp3", "wav", "m4a", "ogg", "flac", "mp4", "webm", "mov", "mkv",
        "canvas", "base",
    }
)
_LINK_SUFFIX_RE = re.compile(r"[#^]")


def normalize_link_target(link: str) -> str:
    """Reduce a raw wikilink target to a lowercase note key, Obsidian style.

    Drops ``#heading`` and ``^block`` suffixes, a leading ``./`` or ``/``,
    and a trailing ``.md``, so ``[[Folder/Note.md#Intro]]`` becomes
    ``folder/note``. A link to a section of the same note (``[[#Intro]]``)
    becomes ``""``.
    """

    target = _LINK_SUFFIX_RE.split(link, maxsplit=1)[0].strip().replace("\\", "/")
    while target.startswith(("./", "/")):
        target = target[2:] if target.startswith("./") else target[1:]
    target = target.lower()
    if target.endswith(".md"):
        target = target[:-3]
    return target.strip()


def _path_key(path: str) -> str:
    key = path.replace("\\", "/").lower()
    return key[:-3] if key.endswith(".md") else key


def _is_attachment(target: str) -> bool:
    _, dot, suffix = target.rpartition(".")
    return bool(dot) and suffix in _ATTACHMENT_SUFFIXES


class LinkResolver:
    """Resolve wikilinks to note paths the way Obsidian does.

    A bare ``[[Note]]`` matches a file name; when no file has that name, it
    falls back to a note that lists it under ``aliases``. A path-qualified
    ``[[Folder/Note]]`` matches only notes whose path ends with that path,
    and a relative ``[[../Folder/Note]]`` is first joined to the linking
    note's folder. When several notes match, the one in the linking note's folder wins,
    then the shallowest path, then the alphabetically first, so each link
    resolves to exactly one note, as it does in Obsidian.
    """

    def __init__(self, metas: list[dict]) -> None:
        self._by_stem: dict[str, list[str]] = {}
        self._by_alias: dict[str, list[str]] = {}
        for meta in metas:
            self._by_stem.setdefault(meta["title_key"], []).append(meta["path"])
            for alias in meta.get("aliases", []):
                key = alias.strip().lower()
                if key:
                    self._by_alias.setdefault(key, []).append(meta["path"])

    def resolve(self, link: str, source: str) -> str | None:
        """Return the note ``link`` points to, or None if nothing matches.

        A link that resolves to ``source`` itself also returns None.
        """

        candidates = self._candidates(*_target_from(link, source))
        if not candidates:
            return None
        best = min(candidates, key=lambda path: _closeness(path, source))
        return None if best == source else best

    def resolve_all(self, links: list[str], source: str) -> list[str]:
        """Resolve a note's outlinks to sorted, distinct target paths."""

        resolved = {self.resolve(link, source) for link in links}
        resolved.discard(None)
        return sorted(resolved)

    def unresolved(self, links: list[str], source: str) -> set[str]:
        """Return the normalized targets of links that match no note.

        Section links within the note, links to the note itself, and
        attachment embeds such as ``![[diagram.png]]`` are not counted.
        """

        missing: set[str] = set()
        for link in links:
            target, exact = _target_from(link, source)
            if target and not _is_attachment(target) and not self._candidates(target, exact):
                missing.add(target)
        return missing

    def _candidates(self, target: str, exact: bool = False) -> list[str]:
        """Notes ``target`` may mean; ``exact`` demands the full vault path."""

        if not target:
            return []
        stem = target.rsplit("/", 1)[-1]
        if exact:
            return [path for path in self._by_stem.get(stem, []) if _path_key(path) == target]
        if "/" not in target:
            return self._by_stem.get(target) or self._by_alias.get(target, [])
        return [
            path
            for path in self._by_stem.get(stem, [])
            if _path_key(path) == target or _path_key(path).endswith("/" + target)
        ]


def _target_from(link: str, source: str) -> tuple[str, bool]:
    """Normalize ``link``; return ``(target, exact)``.

    A ``../`` link is joined to ``source``'s folder and must then match that
    exact vault path (``exact`` is True): suffix matching would let a missing
    ``Area/Topic/Note`` resolve to an unrelated ``Other/Area/Topic/Note``.
    A relative link that climbs above the vault root is left as written, so
    it never matches. A leading ``/`` is vault-absolute (Obsidian's
    ``[[/Folder/Note]]``, and every canvas file card), so it too must match the
    exact path: proximity would hand a root ``/Alpha.md`` to a nearer
    ``Projects/Alpha.md``.
    """

    target = normalize_link_target(link)
    if link.strip().replace("\\", "/").startswith("/"):
        return target, True
    if not target.startswith("../"):
        return target, False
    joined = posixpath.normpath(posixpath.join(posixpath.dirname(_path_key(source)), target))
    return (target if joined.startswith("..") else joined), True


def _closeness(path: str, source: str) -> tuple[int, int, str]:
    same_folder = path.rpartition("/")[0] == source.rpartition("/")[0]
    return (0 if same_folder else 1, path.count("/"), path)


def ghost_targets(metas: list[dict], limit: int | None = None) -> list[dict]:
    """Rank link targets that match no note by how many notes link to them.

    These "ghost" notes are concepts the vault keeps referencing but has not
    written yet. Each entry has the normalized ``target``, the ``inbound``
    count of distinct linking notes, and up to five of those notes.
    """

    resolver = LinkResolver(metas)
    sources_by_target: dict[str, set[str]] = {}
    for meta in metas:
        for target in resolver.unresolved(meta["links"], meta["path"]):
            sources_by_target.setdefault(target, set()).add(meta["path"])
    ranked = sorted(sources_by_target.items(), key=lambda item: (-len(item[1]), item[0]))
    if limit is not None:
        ranked = ranked[:limit]
    return [
        {"target": target, "inbound": len(sources), "linked_from": sorted(sources)[:5]}
        for target, sources in ranked
    ]


def vault_metas(metas: list[dict]) -> list[dict]:
    """Drop code-source files from a note population.

    The vault's semantic, link, tag and co-mention graph is computed over
    vault notes only, so adding a code source never moves the vault mean,
    takes a kNN slot, or captures a ``[[README]]`` link.
    """

    return [meta for meta in metas if not is_code_source_path(meta["path"])]


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
        # unique is sorted and deduplicated, so every b drawn from after
        # index i is already strictly greater than a -- no need to re-sort.
        for b in unique[i + 1 :]:
            pairs.add((a, b))
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


def _row_to_meta(row: sqlite3.Row) -> dict:
    """Build a plain note-metadata dict from a ``note_meta`` table row."""

    return {
        "path": row["path"],
        "title": row["title"],
        "title_key": row["title_key"],
        "tags": json.loads(row["tags_json"]),
        "links": json.loads(row["links_json"]),
        "aliases": json.loads(row["aliases_json"]),
        "link_pairs": [tuple(pair) for pair in json.loads(row["link_pairs_json"])],
        "code": json.loads(row["code_json"]),
        "centroid": _decode_centroid(row["centroid"]),
        "dim": row["dim"],
    }


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
        type=row["type"],
        code_refs=json.loads(row["refs_json"]),
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
                    aliases_json TEXT NOT NULL DEFAULT '[]',
                    link_pairs_json TEXT NOT NULL DEFAULT '[]',
                    code_json TEXT NOT NULL DEFAULT '{}',
                    centroid BLOB,
                    dim INTEGER,
                    updated_at REAL DEFAULT (strftime('%s','now'))
                )
                """
            )
            # Databases created before aliases were tracked lack the column;
            # their rows read as alias-free until the next full sync.
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(note_meta)")}
            if "aliases_json" not in columns:
                conn.execute(
                    "ALTER TABLE note_meta ADD COLUMN aliases_json TEXT NOT NULL DEFAULT '[]'"
                )
            # Same for link pairs: only canvases have them, and canvases were
            # not indexed before the column existed.
            if "link_pairs_json" not in columns:
                conn.execute(
                    "ALTER TABLE note_meta ADD COLUMN link_pairs_json TEXT NOT NULL DEFAULT '[]'"
                )
            # Code mentions and symbols: rows written before they were
            # recorded read as having none until the next full sync.
            if "code_json" not in columns:
                conn.execute(
                    "ALTER TABLE note_meta ADD COLUMN code_json TEXT NOT NULL DEFAULT '{}'"
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
                    type TEXT NOT NULL DEFAULT 'links',
                    refs_json TEXT NOT NULL DEFAULT '[]',
                    updated_at REAL DEFAULT (strftime('%s','now')),
                    PRIMARY KEY (src, dst)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_state (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    updated_at REAL DEFAULT (strftime('%s','now'))
                )
                """
            )
            # Edges stored before edge types existed were all vault-note
            # links, which is what the column default says.
            edge_columns = {row["name"] for row in conn.execute("PRAGMA table_info(edges)")}
            if "type" not in edge_columns:
                conn.execute("ALTER TABLE edges ADD COLUMN type TEXT NOT NULL DEFAULT 'links'")
            if "refs_json" not in edge_columns:
                conn.execute("ALTER TABLE edges ADD COLUMN refs_json TEXT NOT NULL DEFAULT '[]'")
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
        aliases: list[str] | None = None,
        link_pairs: list[tuple[str, str]] | None = None,
        code: dict | None = None,
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
                INSERT INTO note_meta(
                    path, title, title_key, tags_json, links_json, aliases_json,
                    link_pairs_json, code_json, centroid, dim, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, strftime('%s','now'))
                ON CONFLICT(path) DO UPDATE SET
                    title=excluded.title,
                    title_key=excluded.title_key,
                    tags_json=excluded.tags_json,
                    links_json=excluded.links_json,
                    aliases_json=excluded.aliases_json,
                    link_pairs_json=excluded.link_pairs_json,
                    code_json=excluded.code_json,
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
                    json.dumps(sorted(aliases or [])),
                    json.dumps(sorted([list(pair) for pair in link_pairs or []])),
                    json.dumps(code or {}, sort_keys=True),
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
        return _row_to_meta(row)

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
        return [_row_to_meta(row) for row in rows]

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

    def vault_mean(self) -> np.ndarray | None:
        """Return the centroid mean the stored edges were scored against."""

        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM graph_state WHERE key = ?", (_VAULT_MEAN_KEY,)
            ).fetchone()
        if row is None or row["value"] is None:
            return None
        return np.frombuffer(row["value"], dtype=np.float32)

    def set_vault_mean(self, mean: np.ndarray | None) -> None:
        """Persist the centroid mean, or clear it when centering is not in use."""

        with self._connect() as conn:
            if mean is None:
                conn.execute("DELETE FROM graph_state WHERE key = ?", (_VAULT_MEAN_KEY,))
                return
            conn.execute(
                """
                INSERT INTO graph_state(key, value, updated_at)
                VALUES(?, ?, strftime('%s','now'))
                ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value, updated_at=excluded.updated_at
                """,
                (_VAULT_MEAN_KEY, np.asarray(mean, dtype=np.float32).tobytes()),
            )

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

    def replace_edges_for_paths(
        self, paths: set[str], edges: list[Edge], code_edges: list[Edge] | None = None
    ) -> None:
        """Atomically replace every edge touching any of ``paths`` with ``edges``.

        The delete and the inserts run in a single transaction (one shared
        connection), so a failure partway through -- a scoring bug, a write
        conflict from a concurrent writer, anything -- rolls back the delete
        too. Without this, deleting stale edges and inserting their
        replacements as two independently-committing operations could leave
        the affected notes with no edges at all if the second step never
        completed.

        Only ``links`` edges are replaced by path. ``code_edges``, when given,
        replaces every typed edge in the same transaction: those are
        recomputed whole on each sync.
        """

        if not paths and not edges and code_edges is None:
            return
        with self._connect() as conn:
            if paths:
                plist = list(paths)
                placeholders = ",".join("?" for _ in plist)
                conn.execute(
                    f"DELETE FROM edges WHERE type = 'links' AND "
                    f"(src IN ({placeholders}) OR dst IN ({placeholders}))",
                    plist + plist,
                )
            for edge in edges:
                self._insert_edge(conn, edge)
            if code_edges is not None:
                conn.execute("DELETE FROM edges WHERE type != 'links'")
                for edge in code_edges:
                    self._insert_edge(conn, edge)

    @staticmethod
    def _insert_edge(conn: sqlite3.Connection, edge: Edge) -> None:
        conn.execute(
            """
            INSERT INTO edges(
                src, dst, semantic, link, tag, comention, comention_count,
                link_src_to_dst, link_dst_to_src, shared_tags_json, composite,
                type, refs_json, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, strftime('%s','now'))
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
                type=excluded.type,
                refs_json=excluded.refs_json,
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
                edge.type,
                json.dumps(edge.code_refs),
            ),
        )

    def edges_for(self, path: str) -> list[Edge]:
        """Return every edge touching ``path``, unsorted."""

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM edges WHERE src = ? OR dst = ?", (path, path)
            ).fetchall()
        return [_row_to_edge(row) for row in rows]

    def edges_for_many(self, paths) -> list[Edge]:
        """Return every edge touching any of ``paths``, in one query."""

        plist = list(paths)
        if not plist:
            return []
        placeholders = ",".join("?" for _ in plist)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM edges WHERE src IN ({placeholders}) OR dst IN ({placeholders})",
                plist + plist,
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

    def semantic_edge_stats(self) -> dict[str, tuple[int, float]]:
        """Return each note's (positive-semantic edge count, weakest score).

        One query for the whole vault instead of one round trip per note.
        Powers the incremental builder's reverse-kNN heuristic: whether a
        note already has ``graph_knn_k`` semantic neighbors determines
        whether a new candidate only needs to clear ``graph_semantic_min``
        (spare capacity), or must beat the note's current weakest semantic
        edge to displace it (already at capacity). A note absent from the
        result has no positive-semantic edges yet.
        """

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT src, dst, semantic FROM edges WHERE semantic > 0"
            ).fetchall()
        stats: dict[str, tuple[int, float]] = {}
        for row in rows:
            semantic = float(row["semantic"])
            for node in (row["src"], row["dst"]):
                count, weakest = stats.get(node, (0, semantic))
                stats[node] = (count + 1, min(weakest, semantic))
        return stats

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
        return {row["path"]: _row_to_meta(row) for row in rows}

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
    resolver: LinkResolver
    resolved_links: dict[str, list[str]]
    backlinks_index: dict[str, set[str]]
    # Notes a canvas edge connects, both directions: path -> {path}. Kept apart
    # from resolved_links so a canvas edge does not make either end a source
    # of co-mentions or backlinks.
    pair_links: dict[str, set[str]]
    pair_partners: dict[str, set[str]]
    comention_counts: dict[tuple[str, str], int]
    paths: list[str]
    sims: np.ndarray
    path_to_idx: dict[str, int]
    center: np.ndarray | None


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
        """Recompute and persist edges using the same evidence engine as map."""

        metas = self.store.all_note_meta()
        edges, center = self._edges_for_population(vault_metas(metas))
        edges += self.code_edges(metas)
        self.store.replace_all_edges(edges)
        self.store.set_vault_mean(center)
        return {"nodes": len(metas), "edges": len(edges), "edges_updated": len(edges)}

    def code_edges(self, metas: list[dict]) -> list[Edge]:
        """Typed ``mentions_code`` / ``imports`` edges for this population."""

        if not self.config.graph_enabled:
            return []
        return code_edges(metas, self.config.graph_weight_link)

    def edges_from_metadata(self, metas: list[dict]) -> list[Edge]:
        """Compute a transient graph solely from the supplied note population.

        Filter before calling: excluded sources must not contribute co-mentions
        or consume semantic neighbor slots. Centroids come from these same rows,
        with no store reads or writes, and existing scoring/fanout/cap rules apply.
        Sorting paths makes equal-similarity candidate selection reproducible.
        Semantic scores are centered on this population's own mean, so
        excluded notes do not shift them either.
        """

        return self._edges_for_population(metas)[0]

    def _edges_for_population(self, metas: list[dict]) -> tuple[list[Edge], np.ndarray | None]:
        """Score every candidate pair; also return the centering mean used."""

        if not self.config.graph_enabled or len(metas) < 2:
            return [], None
        ctx = self._build_context(sorted(metas, key=lambda meta: meta["path"]))
        return self._score_all(ctx), ctx.center

    def _score_all(self, ctx: _BuildContext) -> list[Edge]:
        candidate_pairs = set(ctx.comention_counts)
        for path in ctx.meta_by_path:
            candidate_pairs |= self._candidates_for(path, ctx)
        return sorted(self._score_pairs(candidate_pairs, ctx), key=lambda edge: (edge.src, edge.dst))

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

        all_metas = self.store.all_note_meta()
        typed = self.code_edges(all_metas)
        # Code-source files are not in the vault graph (see ``vault_metas``),
        # so their changes cannot move a ``links`` edge; they still re-target
        # typed edges, which are recomputed whole below.
        changed = {path for path in changed if not is_code_source_path(path)}
        deleted = {path for path in deleted if not is_code_source_path(path)}
        if not changed and not deleted:
            self.store.replace_edges_for_paths(set(), [], code_edges=typed)
            counts = self.store.counts()
            return {"nodes": counts["nodes"], "edges": counts["edges"], "edges_updated": len(typed)}
        metas = vault_metas(all_metas)
        if len(metas) < 2:
            self.store.replace_all_edges(typed)
            return {"nodes": len(all_metas), "edges": len(typed), "edges_updated": len(typed)}

        # Score against the mean the stored edges were scored with, so new and
        # untouched edges stay comparable (the reverse-kNN check compares a
        # fresh similarity with stored ones). A full sync refreshes the mean.
        stored_mean = self.store.vault_mean()
        ctx = self._build_context(sorted(metas, key=lambda meta: meta["path"]), stored_mean)
        if ctx.center is not stored_mean:
            # The scoring frame changed: the first centered build, the vault
            # crossing the minimum size for centering, a new embedding
            # dimension, or centering toggled. Every stored score is in the
            # old frame, so rescore all edges rather than only affected ones.
            edges = self._score_all(ctx) + typed
            self.store.replace_all_edges(edges)
            self.store.set_vault_mean(ctx.center)
            return {"nodes": len(all_metas), "edges": len(edges), "edges_updated": len(edges)}
        affected = self._affected_paths(changed, deleted, old_meta, ctx)

        candidate_pairs: set[tuple[str, str]] = set()
        for path in affected:
            if path in ctx.meta_by_path:
                candidate_pairs |= self._candidates_for(path, ctx)
        for pair in ctx.comention_counts:
            if pair[0] in affected or pair[1] in affected:
                candidate_pairs.add(pair)

        edges = self._score_pairs(candidate_pairs, ctx)
        self.store.replace_edges_for_paths(affected | deleted, edges, code_edges=typed)

        counts = self.store.counts()
        return {
            "nodes": counts["nodes"],
            "edges": counts["edges"],
            "edges_updated": len(edges) + len(typed),
        }

    # -- context building ------------------------------------------------

    def _build_context(
        self, metas: list[dict], stored_mean: np.ndarray | None = None
    ) -> _BuildContext:
        """Use one population for title/link evidence and semantic candidates.

        Loading centroids separately from the store would reintroduce excluded
        notes into a transient map and let them occupy eligible kNN slots.
        ``stored_mean``, when it fits, replaces this population's own mean as
        the centering vector.
        """

        meta_by_path = {m["path"]: m for m in metas}

        resolver = LinkResolver(metas)
        resolved_links = {m["path"]: resolver.resolve_all(m["links"], m["path"]) for m in metas}

        backlinks_index: dict[str, set[str]] = {}
        for source, targets in resolved_links.items():
            for target in targets:
                backlinks_index.setdefault(target, set()).add(source)

        pair_links: dict[str, set[str]] = {}
        pair_partners: dict[str, set[str]] = {}
        for meta in metas:
            for raw_src, raw_dst in meta.get("link_pairs", []):
                src = resolver.resolve(raw_src, meta["path"])
                dst = resolver.resolve(raw_dst, meta["path"])
                if src is None or dst is None or src == dst:
                    continue
                pair_links.setdefault(src, set()).add(dst)
                pair_partners.setdefault(src, set()).add(dst)
                pair_partners.setdefault(dst, set()).add(src)

        comention_counts: dict[tuple[str, str], int] = {}
        for links in resolved_links.values():
            for pair in comention_pairs(links, self.config.graph_comention_max_fanout):
                comention_counts[pair] = comention_counts.get(pair, 0) + 1

        embedded = [meta for meta in metas if meta["centroid"] is not None]
        paths = [meta["path"] for meta in embedded]
        matrix = (
            np.stack([np.asarray(meta["centroid"], dtype=np.float32) for meta in embedded])
            if embedded
            else np.empty((0, 0), dtype=np.float32)
        )
        center = self._centering_vector(matrix, stored_mean)
        if not paths:
            sims = np.empty((0, 0))
        else:
            sims = cosine_matrix(matrix - center if center is not None else matrix)
        path_to_idx = {p: i for i, p in enumerate(paths)}

        return _BuildContext(
            meta_by_path=meta_by_path,
            resolver=resolver,
            resolved_links=resolved_links,
            backlinks_index=backlinks_index,
            pair_links=pair_links,
            pair_partners=pair_partners,
            comention_counts=comention_counts,
            paths=paths,
            sims=sims,
            path_to_idx=path_to_idx,
            center=center,
        )

    def _centering_vector(
        self, matrix: np.ndarray, stored_mean: np.ndarray | None
    ) -> np.ndarray | None:
        """Pick the vector to subtract from centroids before cosine, if any.

        Averaging a note's chunk vectors pulls every centroid toward the
        corpus mean, so raw centroid cosines bunch up (in one measured vault,
        every pair scored above 0.5 and the median was 0.71). Subtracting the
        mean leaves only what distinguishes each note, spreading the median
        pair to about 0.
        """

        if not self.config.graph_semantic_center or matrix.shape[0] < _MIN_CENTERING_NOTES:
            return None
        if stored_mean is not None and stored_mean.shape == (matrix.shape[1],):
            return stored_mean
        return matrix.mean(axis=0)

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
        for partner in ctx.pair_partners.get(path, []):
            pairs.add(canonical_pair(path, partner))
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
        touched = changed | deleted

        def add_targets(targets: list[str]) -> None:
            affected.update(targets)
            for a, b in comention_pairs(targets, max_fanout):
                affected.add(a)
                affected.add(b)

        # Resolve old links against the vault as it was, so a target that
        # only existed before this sync still counts as dropped.
        previous = [m for path, m in ctx.meta_by_path.items() if path not in touched]
        previous += [m for path in touched if (m := old_meta.get(path)) and "title_key" in m]
        old_resolver = LinkResolver(previous)

        for path in touched:
            old = old_meta.get(path)
            if old:
                add_targets(old_resolver.resolve_all(old.get("links", []), path))
            add_targets(ctx.resolved_links.get(path, []))
            affected.update(ctx.backlinks_index.get(path, []))

        # An unchanged note's link can re-resolve without its own edit: a
        # deleted duplicate hands [[Dup]] to the other Dup, and a new note in
        # the linker's folder or a changed alias takes it over. Re-check every
        # note whose links name a touched note's file name or alias.
        for source in self._possible_linkers(touched, old_meta, ctx):
            if source in touched:
                continue
            links = ctx.meta_by_path[source]["links"]
            before = old_resolver.resolve_all(links, source)
            after = ctx.resolved_links.get(source, [])
            if before != after:
                affected.add(source)
                add_targets(before)
                add_targets(after)

        # Reverse-kNN heuristic: a changed note may now belong in another
        # note's semantic neighborhood even if it wasn't there before. Fetch
        # every note's current semantic edge count + weakest score in one
        # query up front instead of one round trip per (changed note,
        # candidate) pair -- otherwise this loop is O(len(changed) * vault
        # size) individual SQLite connections, which dominates every
        # incremental sync.
        #
        # A candidate only needs to clear graph_semantic_min if it still has
        # spare neighbor capacity (fewer than graph_knn_k semantic edges);
        # only a candidate already at capacity must beat its current
        # weakest edge to be considered displaced.
        semantic_min = self.config.graph_semantic_min
        knn_k = self.config.graph_knn_k
        semantic_stats = (
            self.store.semantic_edge_stats() if any(p in ctx.path_to_idx for p in changed) else {}
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
                count, weakest = semantic_stats.get(candidate, (0, 0.0))
                threshold = semantic_min if count < knn_k else max(semantic_min, weakest)
                if float(sim) >= threshold:
                    affected.add(candidate)

        return affected

    @staticmethod
    def _possible_linkers(
        touched: set[str], old_meta: dict[str, dict | None], ctx: _BuildContext
    ) -> set[str]:
        """Notes with a link whose last path segment names a touched note."""

        names: set[str] = set()
        for path in touched:
            for meta in (ctx.meta_by_path.get(path), old_meta.get(path)):
                if meta:
                    names.add(meta.get("title_key", ""))
                    names.update(alias.strip().lower() for alias in meta.get("aliases", []))
        names.discard("")
        if not names:
            return set()
        return {
            source
            for source, meta in ctx.meta_by_path.items()
            if any(
                normalize_link_target(link).rsplit("/", 1)[-1] in names for link in meta["links"]
            )
        }

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

        link_src_to_dst = (
            dst in ctx.resolved_links.get(src, []) or dst in ctx.pair_links.get(src, set())
        )
        link_dst_to_src = (
            src in ctx.resolved_links.get(dst, []) or src in ctx.pair_links.get(dst, set())
        )
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
    """Build an undirected, composite-weighted graph from scored edges."""

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
