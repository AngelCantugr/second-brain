"""Configuration loading for the RAG service."""

from __future__ import annotations

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


def load_config(path: str | Path) -> RagConfig:
    """Load and normalize config values from a TOML file."""

    p = Path(path)
    raw = tomllib.loads(p.read_text(encoding="utf-8"))
    return RagConfig(
        vault_path=Path(raw["vault_path"]),
        qdrant_path=Path(raw["qdrant_path"]),
        fts_path=Path(raw["fts_path"]),
        sync_state_path=Path(raw.get("sync_state_path", "./data/sync_state.sqlite")),
        collection_name=raw.get("collection_name", "obsidian_chunks"),
        ollama_url=raw.get("ollama_url", "http://127.0.0.1:11434"),
        embedding_model=raw.get("embedding_model", "nomic-embed-text"),
        chunk_size=int(raw.get("chunk_size", 500)),
        chunk_overlap=int(raw.get("chunk_overlap", 80)),
        watch_enabled=bool(raw.get("watch_enabled", True)),
        exclude_globs=list(raw.get("exclude_globs", [])),
        max_context_chunks=int(raw.get("max_context_chunks", 8)),
        redact_patterns=list(raw.get("redact_patterns", [])),
    )
