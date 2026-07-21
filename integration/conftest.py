"""Shared fixtures for real-backend integration tests.

Unlike tests/, these fixtures deliberately do NOT stub the embedder or swap
in InMemoryVectorStore: they exercise a real Ollama (embedding) server and
the embedded-but-real QdrantVectorStore, plus (in test_mcp_stdio.py) the
actual MCP stdio subprocess/transport.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from second_brain.config import RagConfig
from second_brain.service import RagService

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")

# A small, deterministic, interlinked vault so retrieval/graph assertions are
# meaningful rather than arbitrary. Notes share tags, wikilink each other, and
# co-mention "Gamma Project" so semantic + link + tag + comention signals all
# have real evidence to report on.
_VAULT_NOTES: dict[str, str] = {
    "alpha.md": """---
tags: [project, alpha]
---
# Alpha Overview

Alpha is a research initiative into local-first retrieval-augmented
generation. It stores embeddings locally and never calls a remote LLM API.

See [[beta]] for the follow-up rollout, part of the Gamma Project.
""",
    "beta.md": """---
tags: [project, beta]
---
# Beta Rollout

Beta builds on [[alpha]] by adding a note-association graph blending
semantic similarity, wikilinks, shared tags, and co-mentions. This is also
part of the Gamma Project.
""",
    "gamma.md": """---
tags: [project, gamma]
---
# Gamma Project Charter

The Gamma Project coordinates Alpha and Beta into a single vault-wide
knowledge base with hybrid search across vector and keyword indexes.
""",
    "unrelated.md": """---
tags: [misc]
---
# Grocery List

Milk, eggs, bread, and coffee. Nothing to do with software at all.
""",
}


def _write_vault(vault_dir: Path) -> None:
    vault_dir.mkdir(parents=True, exist_ok=True)
    for name, content in _VAULT_NOTES.items():
        (vault_dir / name).write_text(content, encoding="utf-8")


@pytest.fixture(scope="session")
def integration_config(tmp_path_factory: pytest.TempPathFactory) -> RagConfig:
    """RagConfig pointed at a real Ollama URL and a throwaway vault/data dir."""

    root = tmp_path_factory.mktemp("second-brain-integration")
    vault = root / "vault"
    _write_vault(vault)
    return RagConfig(
        vault_path=vault,
        qdrant_path=root / "data" / "qdrant",
        fts_path=root / "data" / "fts.sqlite",
        sync_state_path=root / "data" / "sync_state.sqlite",
        ollama_url=OLLAMA_URL,
        embedding_model=EMBEDDING_MODEL,
        chunk_size=200,
        chunk_overlap=20,
    )


@pytest.fixture(scope="session")
def config_path(integration_config: RagConfig, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write integration_config out as rag_config.toml for the MCP subprocess."""

    root = tmp_path_factory.mktemp("second-brain-config")
    path = root / "rag_config.toml"
    path.write_text(
        "\n".join(
            [
                f'vault_path = "{integration_config.vault_path}"',
                f'qdrant_path = "{integration_config.qdrant_path}"',
                f'fts_path = "{integration_config.fts_path}"',
                f'sync_state_path = "{integration_config.sync_state_path}"',
                f'ollama_url = "{integration_config.ollama_url}"',
                f'embedding_model = "{integration_config.embedding_model}"',
                f"chunk_size = {integration_config.chunk_size}",
                f"chunk_overlap = {integration_config.chunk_overlap}",
            ]
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture(scope="session")
def rag_service(integration_config: RagConfig) -> RagService:
    """A RagService against real Ollama + real (embedded) Qdrant, synced once."""

    service = RagService(integration_config)
    result = service.sync(mode="full")
    assert not result["errors"], f"initial full sync reported errors: {result}"
    assert result["processed"] == len(_VAULT_NOTES)
    return service


@pytest.fixture(scope="session")
def python_executable() -> str:
    return sys.executable
