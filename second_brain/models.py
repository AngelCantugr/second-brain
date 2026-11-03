"""Core data contracts shared across the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from second_brain.code import CodeSymbol


@dataclass(slots=True)
class ParsedNote:
    """Canonical representation of one parsed markdown note."""

    note_id: str
    path: str
    title: str
    body: str
    frontmatter: dict[str, Any]
    tags: list[str]
    links: list[str]
    headings: list[str]
    tasks: list[str]
    mtime: float
    content_hash: str
    # Fields below default to the markdown shape, so loaders for other formats
    # fill only what their format has.
    kind: str = "markdown"
    # Pre-split ``(heading_path, text)`` sections for formats whose structure is
    # not markdown headings (canvas text nodes, PDF pages). ``None`` means the
    # chunker splits ``body`` on headings.
    sections: list[tuple[str, str]] | None = None
    # Links the file makes outside its text, such as canvas file nodes. They
    # are also in ``links``; kept apart so redaction, which re-extracts
    # ``links`` from the redacted body, does not drop them.
    file_links: list[str] = field(default_factory=list)
    # Directed ``(from, to)`` links the file asserts between two other files,
    # such as a canvas edge between two file nodes. The graph scores them as
    # link evidence between those two notes.
    link_pairs: list[tuple[str, str]] = field(default_factory=list)
    # Extra keys copied onto every chunk's metadata, such as the ``repo`` a
    # code-source file came from, so search filters can match them.
    metadata: dict[str, Any] = field(default_factory=dict)
    # Outlined functions, classes and types of a source file. When set, each
    # symbol becomes exactly one chunk and ``body``/``sections`` are unused.
    symbols: list[CodeSymbol] | None = None


@dataclass(slots=True)
class ChunkRecord:
    """Single indexable chunk derived from a note."""

    chunk_id: str
    note_id: str
    text: str
    metadata: dict[str, Any]
    bm25_text: str
    heading_path: str = ""


@dataclass(slots=True)
class RetrievalHit:
    """Search hit returned from semantic, keyword, or hybrid retrieval."""

    chunk_id: str
    score: float
    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    semantic_score: float | None = None
    keyword_score: float | None = None


@dataclass(slots=True)
class SearchResult:
    """Container for a query and its retrieval hits."""

    query: str
    hits: list[RetrievalHit]


@dataclass(slots=True)
class SyncResult:
    """Summary of one indexing sync execution."""

    processed: int
    skipped: int
    deleted: int
    errors: list[str]
    graph_edges_updated: int = 0


@dataclass(slots=True)
class Edge:
    """One association between two notes, with per-signal component scores.

    ``src``/``dst`` are canonically ordered (``src < dst``) so an undirected
    pair has exactly one row; direction-sensitive evidence (which side links
    to which) is preserved separately in ``link_src_to_dst``/``link_dst_to_src``.
    """

    src: str
    dst: str
    semantic: float
    link: float
    tag: float
    comention: float
    comention_count: int
    link_src_to_dst: bool
    link_dst_to_src: bool
    shared_tags: list[str]
    composite: float
