"""Source loaders: one per file format, chosen by file extension.

Every loader returns the same ``ParsedNote`` shape, so chunking, redaction,
embedding, both indexes and the graph stay format-agnostic. To add a format
(PDF, HTML), write a loader with a ``kind``, its ``extensions`` and a
``load`` that returns a ``ParsedNote`` (set ``sections`` when the format's
structure is not markdown headings, such as PDF pages), then add it to
``LOADERS``. Users opt in through ``include_extensions``.
"""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Protocol

from second_brain.canvas import parse_canvas
from second_brain.models import ParsedNote
from second_brain.parser import parse_note


class SourceLoader(Protocol):
    """Turns one file of a given format into a ``ParsedNote``."""

    # Stored on every chunk as ``kind`` and matched by ``filters={"kind": ...}``.
    kind: str
    # Lowercase, with the leading dot. Matched exactly against the file suffix.
    extensions: tuple[str, ...]

    def load(self, path: Path, vault_root: Path) -> ParsedNote: ...


class MarkdownLoader:
    kind = "markdown"
    extensions = (".md",)

    def load(self, path: Path, vault_root: Path) -> ParsedNote:
        return parse_note(path, vault_root)


class CanvasLoader:
    kind = "canvas"
    extensions = (".canvas",)

    def load(self, path: Path, vault_root: Path) -> ParsedNote:
        return parse_canvas(path, vault_root)


LOADERS: tuple[SourceLoader, ...] = (MarkdownLoader(), CanvasLoader())
_BY_EXTENSION: dict[str, SourceLoader] = {
    extension: loader for loader in LOADERS for extension in loader.extensions
}
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(_BY_EXTENSION)
DEFAULT_INCLUDE_EXTENSIONS: tuple[str, ...] = (".md",)
# Chunks indexed before ``kind`` existed have no such key; they were markdown.
DEFAULT_KIND = MarkdownLoader.kind


def loader_for(path: str | Path, include_extensions: Collection[str]) -> SourceLoader | None:
    """Return the loader for ``path`` if its extension is included, else None."""

    suffix = Path(path).suffix
    if suffix not in include_extensions:
        return None
    return _BY_EXTENSION.get(suffix)


def load_source(
    path: Path, vault_root: Path, include_extensions: Collection[str]
) -> ParsedNote:
    """Parse ``path`` with the loader for its extension."""

    loader = loader_for(path, include_extensions)
    if loader is None:
        raise ValueError(f"no loader for {path.suffix or 'files without an extension'}")
    return loader.load(path, vault_root)
