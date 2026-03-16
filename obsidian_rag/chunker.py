"""Chunking logic for parsed notes.

Chunks are heading-aware token windows with overlap so retrieval can
preserve context while still matching semantically.
"""

from __future__ import annotations

import hashlib
import re
import uuid

from obsidian_rag.models import ChunkRecord, ParsedNote
from obsidian_rag.parser import derive_metadata


def _split_sections(body: str) -> list[tuple[str, str]]:
    """Split markdown body into sections grouped by nearest heading."""

    lines = body.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_heading = "root"
    current_lines: list[str] = []

    for line in lines:
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            if current_lines:
                sections.append((current_heading, current_lines))
                current_lines = []
            current_heading = heading.group(2).strip()
            continue
        current_lines.append(line)

    if current_lines:
        sections.append((current_heading, current_lines))

    return [(h, "\n".join(ls).strip()) for h, ls in sections if "\n".join(ls).strip()]


def _token_windows(tokens: list[str], size: int, overlap: int) -> list[list[str]]:
    """Create overlapping token windows from a token list."""

    if not tokens:
        return []
    windows: list[list[str]] = []
    step = max(1, size - overlap)
    idx = 0
    while idx < len(tokens):
        window = tokens[idx : idx + size]
        if not window:
            break
        windows.append(window)
        idx += step
    return windows


def chunk_note(note: ParsedNote, chunk_size: int, chunk_overlap: int) -> list[ChunkRecord]:
    """Convert a parsed note into chunk records ready for indexing."""

    chunks: list[ChunkRecord] = []
    derived = derive_metadata(note.frontmatter)

    for heading_path, section_text in _split_sections(note.body):
        tokens = section_text.split()
        for i, window in enumerate(_token_windows(tokens, chunk_size, chunk_overlap)):
            text = " ".join(window)
            hash_input = f"{note.note_id}:{heading_path}:{i}:{text}"
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, hash_input))
            metadata = {
                "path": note.path,
                "note_title": note.title,
                "heading_path": heading_path,
                "tags": note.tags,
                "links": note.links,
                "tasks": note.tasks,
                "raw_frontmatter": note.frontmatter,
                "derived_fields": derived,
                "mtime": note.mtime,
            }
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    note_id=note.note_id,
                    text=text,
                    bm25_text=text,
                    metadata=metadata,
                    heading_path=heading_path,
                )
            )

    return chunks
