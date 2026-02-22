from __future__ import annotations

from dataclasses import dataclass
import re

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(slots=True)
class Chunk:
    heading: str
    text: str
    chunk_index: int


def _chunk_text(heading: str, body: str, max_tokens: int) -> list[str]:
    words = body.split()
    if not words:
        return []

    chunks: list[str] = []
    for i in range(0, len(words), max_tokens):
        window = " ".join(words[i : i + max_tokens]).strip()
        if window:
            chunks.append(f"{heading}\n{window}")
    return chunks


def chunk_markdown(markdown: str, max_tokens: int = 500) -> list[Chunk]:
    lines = markdown.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_heading = "Untitled"
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        if buffer:
            sections.append((current_heading, buffer))
            buffer = []

    for line in lines:
        heading_match = HEADING_PATTERN.match(line)
        if heading_match:
            flush()
            current_heading = heading_match.group(2).strip()
            continue
        buffer.append(line)
    flush()

    output: list[Chunk] = []
    idx = 0
    for heading, content_lines in sections:
        body = "\n".join(content_lines).strip()
        for text in _chunk_text(heading, body, max_tokens=max_tokens):
            output.append(Chunk(heading=heading, text=text, chunk_index=idx))
            idx += 1
    return output
