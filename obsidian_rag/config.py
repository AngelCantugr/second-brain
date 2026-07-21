"""Configuration loading for the RAG service."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class RagConfig:
    """Runtime configuration loaded from ``rag_config.toml``."""

    vault_path: Path
    qdrant_path: Path
    fts_path: Path
    sync_state_path: Path
    collection_name: str = "obsidian_chunks"
    ollama_url: str = "http://127.0.0.1:11434"
    embedding_model: str = "nomic-embed-text"
    chunk_size: int = 500
    chunk_overlap: int = 80
    watch_enabled: bool = True
    exclude_globs: list[str] = field(default_factory=list)
    max_context_chunks: int = 8
    redact_patterns: list[str] = field(default_factory=list)
    graph_enabled: bool = True
    graph_knn_k: int = 8
    graph_semantic_min: float = 0.35
    graph_min_edge_score: float = 0.15
    graph_weight_semantic: float = 0.5
    graph_weight_link: float = 0.25
    graph_weight_tag: float = 0.15
    graph_weight_comention: float = 0.10
    graph_comention_cap: int = 3
    graph_comention_max_fanout: int = 20


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
        collection_name=raw.get("collection_name", "obsidian_chunks"),
        ollama_url=raw.get("ollama_url", "http://127.0.0.1:11434"),
        embedding_model=raw.get("embedding_model", "nomic-embed-text"),
        chunk_size=int(raw.get("chunk_size", 500)),
        chunk_overlap=int(raw.get("chunk_overlap", 80)),
        watch_enabled=bool(raw.get("watch_enabled", True)),
        exclude_globs=list(raw.get("exclude_globs", [])),
        max_context_chunks=int(raw.get("max_context_chunks", 8)),
        redact_patterns=list(raw.get("redact_patterns", [])),
        graph_enabled=bool(raw.get("graph_enabled", True)),
        graph_knn_k=int(raw.get("graph_knn_k", 8)),
        graph_semantic_min=float(raw.get("graph_semantic_min", 0.35)),
        graph_min_edge_score=float(raw.get("graph_min_edge_score", 0.15)),
        graph_weight_semantic=float(raw.get("graph_weight_semantic", 0.5)),
        graph_weight_link=float(raw.get("graph_weight_link", 0.25)),
        graph_weight_tag=float(raw.get("graph_weight_tag", 0.15)),
        graph_weight_comention=float(raw.get("graph_weight_comention", 0.10)),
        graph_comention_cap=int(raw.get("graph_comention_cap", 3)),
        graph_comention_max_fanout=int(raw.get("graph_comention_max_fanout", 20)),
    )


def _resolve_path(value: str, config_dir: Path) -> Path:
    """Resolve config path strings against CWD/env vars and config file directory."""

    expanded = os.path.expanduser(
        os.path.expandvars(value.replace("$CWD", str(Path.cwd())))
    )
    path = Path(expanded)
    if path.is_absolute():
        return path
    return config_dir / path
