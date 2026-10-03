"""Filesystem scanning helpers for vault markdown files."""

from __future__ import annotations

from fnmatch import fnmatchcase
from functools import lru_cache
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
    return path_is_excluded(rel, globs)


def path_is_excluded(relative_path: str | Path, globs: list[str]) -> bool:
    """Match a vault-relative path using recursive, segment-aware glob rules.

    A ``**`` path segment matches zero or more complete path segments; ordinary
    wildcard segments use shell-style matching without crossing directory
    boundaries. Relative multi-segment patterns match a path suffix, preserving
    ``Path.match`` behavior; slash-free patterns match the filename at any depth.
    """

    path_parts = Path(relative_path).parts
    for pattern in globs:
        pattern_parts = tuple(part for part in pattern.split("/") if part not in ("", "."))
        if len(pattern_parts) == 1:
            if path_parts and fnmatchcase(path_parts[-1], pattern_parts[0]):
                return True
            continue

        # A leading ``**`` lets relative patterns match after any parent path,
        # as Path.match did, while the matcher keeps stars inside path segments.
        suffix_pattern = ("**", *pattern_parts)
        if _matches_glob_segments(path_parts, suffix_pattern):
            return True
    return False


def _matches_glob_segments(path_parts: tuple[str, ...], pattern_parts: tuple[str, ...]) -> bool:
    """Match path components while allowing ``**`` to span directories."""

    @lru_cache(maxsize=None)
    def matches(path_index: int, pattern_index: int) -> bool:
        if pattern_index == len(pattern_parts):
            return path_index == len(path_parts)

        pattern_part = pattern_parts[pattern_index]
        if pattern_part == "**":
            return matches(path_index, pattern_index + 1) or (
                path_index < len(path_parts) and matches(path_index + 1, pattern_index)
            )

        return (
            path_index < len(path_parts)
            and fnmatchcase(path_parts[path_index], pattern_part)
            and matches(path_index + 1, pattern_index + 1)
        )

    return matches(0, 0)
