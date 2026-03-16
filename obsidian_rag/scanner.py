"""Filesystem scanning helpers for vault markdown files."""

from __future__ import annotations

from pathlib import Path


def iter_markdown_files(vault_path: Path, exclude_globs: list[str]) -> list[Path]:
    """Return sorted markdown files in a vault, honoring exclusions."""

    files: list[Path] = []
    for path in vault_path.rglob("*.md"):
        if _is_excluded(path, vault_path, exclude_globs):
            continue
        files.append(path)
    return sorted(files)


def _is_excluded(path: Path, root: Path, globs: list[str]) -> bool:
    """Check if a file path should be excluded from indexing."""

    rel = path.relative_to(root)
    if any(part.startswith(".") for part in rel.parts):
        return True
    rel_str = str(rel)
    return any(rel.match(pattern) or rel_str.startswith(pattern.rstrip("/**")) for pattern in globs)
