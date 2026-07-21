"""Benchmarks for the real-backend RAG pipeline.

Requires the `pytest-benchmark` plugin (dev dependency). Run with
`--benchmark-json=<path>` to produce machine-readable stats; see
`bench_report.py` for rendering those into a human-readable summary.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from second_brain.config import RagConfig
from second_brain.embedder import OllamaEmbedder
from second_brain.service import RagService

from conftest import _write_vault


def test_benchmark_single_text_embedding(benchmark, integration_config: RagConfig) -> None:
    embedder = OllamaEmbedder(integration_config.ollama_url, integration_config.embedding_model)
    text = "Alpha is a research initiative into local-first retrieval-augmented generation."
    benchmark(lambda: embedder.embed([text]))


def test_benchmark_batch_embedding(benchmark, integration_config: RagConfig) -> None:
    embedder = OllamaEmbedder(integration_config.ollama_url, integration_config.embedding_model)
    texts = [
        "Alpha is a research initiative into local-first retrieval.",
        "Beta builds on Alpha with a note-association graph.",
        "Gamma coordinates Alpha and Beta into one knowledge base.",
        "Grocery list: milk, eggs, bread, and coffee.",
    ]
    benchmark(lambda: embedder.embed(texts))


def test_benchmark_full_sync_throughput(benchmark, integration_config: RagConfig, tmp_path: Path) -> None:
    # Isolated storage (not the shared session `rag_service`'s paths) so
    # repeated benchmark rounds don't race a live QdrantVectorStore/lock file
    # held open by the session-scoped fixture at the same path.
    vault = tmp_path / "vault"
    _write_vault(vault)
    isolated_config = RagConfig(
        vault_path=vault,
        qdrant_path=tmp_path / "data" / "qdrant",
        fts_path=tmp_path / "data" / "fts.sqlite",
        sync_state_path=tmp_path / "data" / "sync_state.sqlite",
        ollama_url=integration_config.ollama_url,
        embedding_model=integration_config.embedding_model,
        chunk_size=integration_config.chunk_size,
        chunk_overlap=integration_config.chunk_overlap,
    )

    def _reset_storage() -> None:
        # Wipe any prior round's on-disk index so each round measures a cold
        # full sync, and so a new QdrantVectorStore never opens a path a
        # not-yet-garbage-collected client from the previous round still holds.
        shutil.rmtree(tmp_path / "data", ignore_errors=True)

    def _full_sync() -> dict:
        service = RagService(isolated_config)
        return service.sync(mode="full")

    result = benchmark.pedantic(_full_sync, setup=_reset_storage, rounds=3, iterations=1)
    assert not result["errors"]


def test_benchmark_query_latency(benchmark, rag_service) -> None:
    benchmark(lambda: rag_service.query("Gamma Project charter", top_k=5))


def test_benchmark_search_latency(benchmark, rag_service) -> None:
    benchmark(lambda: rag_service.search("local-first retrieval augmented generation", top_k=10))
