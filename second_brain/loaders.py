"""Source loaders: one per file format, chosen by file extension.

Every loader returns the same ``ParsedNote`` shape, so chunking, redaction,
embedding, both indexes and the graph stay format-agnostic. To add a format
(PDF, HTML), write a loader with a ``kind``, its ``extensions`` and a
``load`` that returns a ``ParsedNote`` (set ``sections`` when the format's
structure is not markdown headings, such as PDF pages), then add it to
``LOADERS``. Users opt in through ``include_extensions``.
"""

from __future__ import annotations

import dataclasses
import hashlib
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


# Kind of the prose files indexed from a code source (README, docs/, ADRs).
REPO_DOC_KIND = "repo_doc"
REPO_DOC_EXTENSIONS: frozenset[str] = frozenset(MarkdownLoader.extensions)


def load_repo_doc(path: Path, repo_root: Path, repo: str, stored_path: str) -> ParsedNote:
    """Parse a code source's markdown file as a ``repo_doc`` stored at ``stored_path``.

    The title names the repo ("second-brain: README") because a bare
    "README" or "CHANGELOG" says nothing in a search hit or an embedding
    header. Note identity hashes the stored path, as for vault notes, so the
    same README in two repos stays two notes.
    """

    parsed = parse_note(path, repo_root)
    identity = hashlib.sha256(f"{stored_path}\0{parsed.content_hash}".encode("utf-8")).hexdigest()
    return dataclasses.replace(
        parsed,
        note_id=identity[:16],
        path=stored_path,
        title=f"{repo}: {parsed.title}",
        kind=REPO_DOC_KIND,
        metadata={"repo": repo},
    )


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
