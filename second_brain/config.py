"""Configuration loading for the RAG service."""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from second_brain.code import SUPPORTED_LANGUAGES, missing_code_dependencies
from second_brain.loaders import DEFAULT_INCLUDE_EXTENSIONS, SUPPORTED_EXTENSIONS
from second_brain.redaction import compile_redact_patterns

DEFAULT_EXCLUDE_GLOBS = (
    ".obsidian/**",
    ".git/**",
    "Templates/**",
    "_types/**",
    "_templates/**",
    "**/_templates*/**",
    "CLAUDE.md",
    "AGENTS.md",
    "GEMINI.md",
)

# Repository prose indexed from a code source by default. Slash-free patterns
# match a file name at any depth; ``docs/**`` matches a docs folder at any depth.
DEFAULT_DOC_GLOBS = (
    "README*",
    "CHANGELOG*",
    "CONTRIBUTING*",
    "ARCHITECTURE*",
    "docs/**",
    "doc/**",
    "adr/**",
    "adrs/**",
    "decisions/**",
)
# A repo name becomes part of every indexed path (``repo:<name>/...``), so it
# is kept to characters that are safe in paths, filters and URLs.
_CODE_SOURCE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CODE_SOURCE_KEYS = frozenset(
    {"name", "path", "doc_globs", "exclude_globs", "languages", "code_globs"}
)


@dataclass(slots=True, frozen=True)
class CodeSource:
    """A repository indexed alongside the vault, under ``repo:<name>/``."""

    name: str
    path: Path
    doc_globs: tuple[str, ...] = DEFAULT_DOC_GLOBS
    # Added to the built-in vendored/build exclusions and ``.gitignore``.
    exclude_globs: tuple[str, ...] = ()
    # Source files of these languages are split into symbol chunks; empty
    # means the repo's docs only.
    languages: tuple[str, ...] = ()
    # Which source files of those languages to index (default: all of them).
    code_globs: tuple[str, ...] = ("*",)


@dataclass(slots=True)
class RagConfig:
    """Runtime configuration loaded from ``rag_config.toml``."""

    vault_path: Path
    qdrant_path: Path
    fts_path: Path
    sync_state_path: Path
    collection_name: str = "second_brain_chunks"
    ollama_url: str = "http://127.0.0.1:11434"
    embedding_model: str = "nomic-embed-text"
    chunk_size: int = 500
    chunk_overlap: int = 80
    watch_enabled: bool = True
    exclude_globs: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE_GLOBS))
    include_extensions: list[str] = field(
        default_factory=lambda: list(DEFAULT_INCLUDE_EXTENSIONS)
    )
    exclude_status: list[str] = field(default_factory=list)
    max_context_chunks: int = 8
    redact_patterns: list[str] = field(default_factory=list)
    graph_enabled: bool = True
    graph_knn_k: int = 8
    graph_semantic_min: float = 0.35
    graph_semantic_center: bool = True
    graph_min_edge_score: float = 0.15
    graph_weight_semantic: float = 0.5
    graph_weight_link: float = 0.25
    graph_weight_tag: float = 0.15
    graph_weight_comention: float = 0.10
    graph_comention_cap: int = 3
    graph_comention_max_fanout: int = 20
    skip_headings: list[str] = field(default_factory=list)
    embed_context_header: bool = True
    embed_batch_size: int = 32
    code_sources: list[CodeSource] = field(default_factory=list)


def load_config(path: str | Path) -> RagConfig:
    """Load and normalize config values from a TOML file."""

    p = Path(path)
    raw = tomllib.loads(p.read_text(encoding="utf-8"))
    config_dir = p.resolve().parent
    return RagConfig(
        vault_path=_resolve_path(raw["vault_path"], config_dir),
        qdrant_path=_resolve_path(raw["qdrant_path"], config_dir),
        fts_path=_resolve_path(raw["fts_path"], config_dir),
        sync_state_path=_resolve_path(
            raw.get("sync_state_path", "./data/sync_state.sqlite"), config_dir
        ),
        collection_name=raw.get("collection_name", "second_brain_chunks"),
        ollama_url=raw.get("ollama_url", "http://127.0.0.1:11434"),
        embedding_model=raw.get("embedding_model", "nomic-embed-text"),
        chunk_size=int(raw.get("chunk_size", 500)),
        chunk_overlap=int(raw.get("chunk_overlap", 80)),
        watch_enabled=bool(raw.get("watch_enabled", True)),
        exclude_globs=list(raw.get("exclude_globs", DEFAULT_EXCLUDE_GLOBS)),
        include_extensions=_validated_extensions(
            raw.get("include_extensions", list(DEFAULT_INCLUDE_EXTENSIONS))
        ),
        exclude_status=_validated_string_list(raw.get("exclude_status", []), "exclude_status"),
        max_context_chunks=int(raw.get("max_context_chunks", 8)),
        redact_patterns=_validated_redact_patterns(raw.get("redact_patterns", [])),
        graph_enabled=bool(raw.get("graph_enabled", True)),
        graph_knn_k=int(raw.get("graph_knn_k", 8)),
        graph_semantic_min=float(raw.get("graph_semantic_min", 0.35)),
        graph_semantic_center=bool(raw.get("graph_semantic_center", True)),
        graph_min_edge_score=float(raw.get("graph_min_edge_score", 0.15)),
        graph_weight_semantic=float(raw.get("graph_weight_semantic", 0.5)),
        graph_weight_link=float(raw.get("graph_weight_link", 0.25)),
        graph_weight_tag=float(raw.get("graph_weight_tag", 0.15)),
        graph_weight_comention=float(raw.get("graph_weight_comention", 0.10)),
        graph_comention_cap=int(raw.get("graph_comention_cap", 3)),
        graph_comention_max_fanout=int(raw.get("graph_comention_max_fanout", 20)),
        skip_headings=_validated_string_list(raw.get("skip_headings", []), "skip_headings"),
        embed_context_header=bool(raw.get("embed_context_header", True)),
        embed_batch_size=_validated_positive_int(
            raw.get("embed_batch_size", 32), "embed_batch_size"
        ),
        code_sources=_validated_code_sources(raw.get("code_sources", []), config_dir),
    )


def _validated_code_sources(value: object, config_dir: Path) -> list[CodeSource]:
    """Parse ``code_sources``; names must be unique because they key every path.

    The path is not required to exist here: a repo on an unmounted drive should
    not stop the MCP server from starting. Sync reports a missing root instead.
    """

    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError("code_sources must be an array of tables")
    sources: list[CodeSource] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        label = f"code_sources[{index}]"
        unknown = set(raw) - _CODE_SOURCE_KEYS
        if unknown:
            raise ValueError(f"{label}: unknown keys {sorted(unknown)}")
        name, path = raw.get("name"), raw.get("path")
        if not isinstance(name, str) or not _CODE_SOURCE_NAME_RE.match(name):
            raise ValueError(
                f"{label}.name must be letters, digits, '.', '_' or '-', got {name!r}"
            )
        if name in seen:
            raise ValueError(f"{label}.name {name!r} is used by another code source")
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"{label}.path must be a non-empty string")
        seen.add(name)
        sources.append(
            CodeSource(
                name=name,
                path=_resolve_path(path, config_dir),
                doc_globs=tuple(
                    _validated_string_list(
                        raw.get("doc_globs", list(DEFAULT_DOC_GLOBS)), f"{label}.doc_globs"
                    )
                ),
                exclude_globs=tuple(
                    _validated_string_list(raw.get("exclude_globs", []), f"{label}.exclude_globs")
                ),
                languages=_validated_languages(raw.get("languages", []), f"{label}.languages"),
                code_globs=tuple(
                    _validated_string_list(raw.get("code_globs", ["*"]), f"{label}.code_globs")
                ),
            )
        )
    return sources


def _validated_string_list(value: object, setting: str) -> list[str]:
    """Require a TOML array of strings for list-valued filtering settings."""

    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{setting} must be an array of strings")
    return value


def _validated_languages(value: object, setting: str) -> tuple[str, ...]:
    """Require supported language names, and fail early if tree-sitter is missing."""

    languages = _validated_string_list(value, setting)
    unknown = sorted(set(languages) - set(SUPPORTED_LANGUAGES))
    if unknown:
        supported = ", ".join(sorted(SUPPORTED_LANGUAGES))
        raise ValueError(f"{setting}: unsupported {unknown}; supported: {supported}")
    missing = missing_code_dependencies(languages) if languages else []
    if missing:
        raise ValueError(
            f"{setting} needs {', '.join(missing)}; "
            "install them with: pip install 'second-brain-mcp[code]'"
        )
    return tuple(dict.fromkeys(languages))


def _validated_positive_int(value: object, setting: str) -> int:
    """Require a TOML integer >= 1; ``int()`` would accept true, 2.5 or "32"."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{setting} must be an integer >= 1")
    return value


def _validated_extensions(value: object) -> list[str]:
    """Normalize to lowercase ``.ext`` and require a loader for each.

    An empty list is rejected: it would make the next sync delete every
    indexed file.
    """

    extensions = _validated_string_list(value, "include_extensions")
    normalized: list[str] = []
    for extension in extensions:
        ext = extension.strip().lower()
        ext = ext if ext.startswith(".") else f".{ext}"
        if ext not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ValueError(
                f"include_extensions: no loader for {extension!r}; supported: {supported}"
            )
        if ext not in normalized:
            normalized.append(ext)
    if not normalized:
        raise ValueError("include_extensions must name at least one extension")
    return normalized


def _validated_redact_patterns(value: object) -> list[str]:
    """Reject non-string or uncompilable patterns when the config loads."""

    patterns = _validated_string_list(value, "redact_patterns")
    compile_redact_patterns(patterns)
    return patterns


def _resolve_path(value: str, config_dir: Path) -> Path:
    """Resolve config path strings against CWD/env vars and config file directory."""

    expanded = os.path.expanduser(
        os.path.expandvars(value.replace("$CWD", str(Path.cwd())))
    )
    path = Path(expanded)
    if path.is_absolute():
        return path
    return config_dir / path
