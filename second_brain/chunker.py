"""Chunking logic for parsed notes.

Chunks are heading-aware token windows with overlap so retrieval can
preserve context while still matching semantically.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import TypeVar

from second_brain.models import ChunkRecord, ParsedNote
from second_brain.parser import (
    TASK_RE,
    derive_metadata,
    findall_outside_code,
    mask_code,
)

WORD_RE = re.compile(r"\S+")
HEADING_LINE_RE = re.compile(r"^(#{1,6})\s+(.*)$")

T = TypeVar("T")


def _split_sections(body: str) -> list[tuple[str, str]]:
    """Split markdown body into sections grouped by nearest heading.

    Headings are detected on the code-masked copy so a ``# comment`` inside a
    fenced block stays section text instead of starting a new section. The
    original lines are kept, so section text retains its indentation.
    """

    lines = body.splitlines()
    masked_lines = mask_code(body).splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_heading = "root"
    current_lines: list[str] = []

    for line, masked_line in zip(lines, masked_lines):
        heading = HEADING_LINE_RE.match(masked_line)
        if heading:
            if current_lines:
                sections.append((current_heading, current_lines))
                current_lines = []
            current_heading = line[heading.start(2) : heading.end(2)].strip()
            continue
        current_lines.append(line)

    if current_lines:
        sections.append((current_heading, current_lines))

    return [(h, "\n".join(ls).strip()) for h, ls in sections if "\n".join(ls).strip()]


def _token_windows(tokens: list[T], size: int, overlap: int) -> list[list[T]]:
    """Create overlapping token windows from a token list."""

    if not tokens:
        return []
    windows: list[list[T]] = []
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
    """Chunk a note with path-scoped ownership, even for caller-built notes.

    Structured identity encoding avoids delimiter ambiguity in user paths,
    headings, and text. Equal content in two paths must remain independent.
    """

    chunks: list[ChunkRecord] = []
    derived = derive_metadata(note.frontmatter)

    for heading_path, section_text in _split_sections(note.body):
        section_tasks = findall_outside_code(
            TASK_RE, mask_code(section_text), section_text
        )
        # Windows are counted in words, but the chunk text is sliced from the
        # original section. Re-joining words would flatten newlines and indentation.
        word_spans = [m.span() for m in WORD_RE.finditer(section_text)]
        for i, window in enumerate(_token_windows(word_spans, chunk_size, chunk_overlap)):
            text = section_text[window[0][0] : window[-1][1]]
            hash_input = json.dumps([note.path, note.note_id, heading_path, i, text])
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, hash_input))
            metadata = {
                "path": note.path,
                "note_title": note.title,
                "heading_path": heading_path,
                "tags": note.tags,
                "links": note.links,
                "tasks": section_tasks,
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
