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
    return path_is_excluded(rel, globs)


def path_is_excluded(relative_path: str | Path, globs: list[str]) -> bool:
    """Match a vault-relative path against configured globs at any depth.

    ``Path.match`` treats a leading ``**/`` as requiring at least one directory;
    trying the equivalent root pattern as well keeps root and nested semantics
    consistent for patterns such as ``**/_templates*/**``.
    """

    rel = Path(relative_path)
    rel_str = rel.as_posix()
    for pattern in globs:
        if rel.match(pattern):
            return True
        root_pattern = pattern.removeprefix("**/")
        if root_pattern != pattern and rel.match(root_pattern):
            return True
        if pattern.endswith("/**"):
            directory = pattern[:-3].rstrip("/")
            if rel_str == directory or rel_str.startswith(f"{directory}/"):
                return True
    return False
