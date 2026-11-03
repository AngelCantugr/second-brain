"""Every root the indexer reads: the vault, plus configured code sources.

A code-source file is stored under a virtual path, ``repo:<name>/<path in
repo>``, so it shares sync state, both indexes and the graph with vault notes
without colliding with them: Obsidian forbids ``:`` in file and folder names,
so no vault path starts with ``repo:``.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from second_brain.config import CodeSource, RagConfig
from second_brain.loaders import REPO_DOC_EXTENSIONS, load_repo_doc, load_source
from second_brain.models import ParsedNote
from second_brain.scanner import iter_source_files, path_is_excluded

REPO_PATH_PREFIX = "repo:"
# Dependency, build and cache folders: never the user's own prose or code,
# and often large. Skipped even when a repo's .gitignore forgets them.
VENDORED_DIRS = frozenset(
    {
        "node_modules", "bower_components", "vendor", "third_party", "dist", "build",
        "out", "target", "coverage", "htmlcov", "site-packages", "__pycache__",
        "venv", "__snapshots__",
    }
)
# Larger files are almost always generated (bundles, fixtures, dumps).
MAX_SOURCE_BYTES = 1_000_000


@dataclass(slots=True, frozen=True)
class SourceFile:
    """One file to index: where it is on disk and the path it is stored under."""

    abs_path: Path
    rel_path: str
    # The code source's name, or None for a vault note.
    repo: str | None = None


def code_source_path(repo: str, rel_path: str) -> str:
    """Return the stored path of ``rel_path`` inside code source ``repo``."""

    return f"{REPO_PATH_PREFIX}{repo}/{rel_path}"


def split_code_source_path(path: str) -> tuple[str, str] | None:
    """Return ``(repo, path in repo)`` for a code-source path, else None."""

    if not path.startswith(REPO_PATH_PREFIX):
        return None
    repo, sep, rel = path[len(REPO_PATH_PREFIX) :].partition("/")
    return (repo, rel) if sep and repo and rel else None


def is_code_source_path(path: str) -> bool:
    return split_code_source_path(path) is not None


def iter_all_source_files(config: RagConfig) -> tuple[list[SourceFile], list[str]]:
    """Return every file to index and the names of code sources that are missing.

    A missing root is reported, not treated as empty, so a repo on an
    unmounted drive keeps its index entries instead of being deleted.
    """

    files = [
        SourceFile(path, str(path.relative_to(config.vault_path)))
        for path in iter_source_files(
            config.vault_path, config.exclude_globs, config.include_extensions
        )
    ]
    missing: list[str] = []
    for source in config.code_sources:
        if not source.path.is_dir():
            missing.append(source.name)
            continue
        files.extend(iter_code_source_files(source))
    return files, missing


def load_source_file(file: SourceFile, config: RagConfig) -> ParsedNote:
    """Parse a vault note or a code-source file with the loader for its root."""

    if file.repo is None:
        return load_source(file.abs_path, config.vault_path, config.include_extensions)
    source = next(s for s in config.code_sources if s.name == file.repo)
    return load_repo_doc(file.abs_path, source.path.resolve(), source.name, file.rel_path)


def iter_code_source_files(source: CodeSource) -> list[SourceFile]:
    """Return a code source's indexable files, sorted by stored path."""

    root = source.path.resolve()
    files: list[SourceFile] = []
    for rel in _candidate_paths(root):
        if not is_eligible_code_source_path(rel, source):
            continue
        path = root / rel
        try:
            # A symlink may point outside the repo; never follow it out.
            if not path.resolve().is_relative_to(root) or not path.is_file():
                continue
            if path.stat().st_size > MAX_SOURCE_BYTES:
                continue
        except OSError:
            continue
        files.append(SourceFile(path, code_source_path(source.name, rel), source.name))
    return sorted(files, key=lambda f: f.rel_path)


def is_eligible_code_source_path(rel_path: str, source: CodeSource) -> bool:
    """Apply a code source's selection rules to a repo-relative POSIX path."""

    parts = Path(rel_path).parts
    if not parts or any(part.startswith(".") or part in VENDORED_DIRS for part in parts):
        return False
    if path_is_excluded(rel_path, list(source.exclude_globs)):
        return False
    return Path(rel_path).suffix in REPO_DOC_EXTENSIONS and path_is_excluded(
        rel_path, list(source.doc_globs)
    )


def _candidate_paths(root: Path) -> list[str]:
    """List repo-relative files, honoring .gitignore when the root is in a git repo.

    ``git ls-files`` applies every .gitignore, ``.git/info/exclude`` and the
    global excludes file exactly as git does, and still lists untracked files
    that are not ignored. Outside a git repo, walk the tree and rely on the
    vendored-folder rules alone.
    """

    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            capture_output=True,
            check=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return _walk(root)
    return sorted({p for p in result.stdout.decode("utf-8", "surrogateescape").split("\0") if p})


def _walk(root: Path) -> list[str]:
    paths: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in VENDORED_DIRS]
        base = Path(dirpath).relative_to(root)
        paths.extend((base / name).as_posix() for name in filenames)
    return sorted(paths)
