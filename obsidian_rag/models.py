"""Core data contracts shared across the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
