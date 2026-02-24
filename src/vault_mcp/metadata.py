from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import frontmatter

EXCLUDED_SEGMENTS = {"_templates", "Archive"}


@dataclass(slots=True)
class Note:
    path: Path
    relative_path: str
    content: str
    tags: list[str]
    date_modified: float


def _is_excluded(path: Path, root: Path) -> bool:
    relative_parts = path.relative_to(root).parts
    return any(part in EXCLUDED_SEGMENTS for part in relative_parts)


def iter_markdown_files(root: Path) -> list[Path]:
    root = root.resolve()
    files: list[Path] = []
    for path in sorted(root.rglob("*.md")):
        if _is_excluded(path, root):
            continue
        files.append(path)
    return files


def _normalize_tags(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def parse_note(path: Path, root: Path) -> Note:
    post = frontmatter.load(path)
    stat = path.stat()
    return Note(
        path=path,
        relative_path=path.relative_to(root).as_posix(),
        content=post.content,
        tags=_normalize_tags(post.metadata.get("tags", [])),
        date_modified=stat.st_mtime,
    )
