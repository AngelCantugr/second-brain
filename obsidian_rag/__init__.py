"""Obsidian RAG package.

This package implements a local Retrieval-Augmented Generation (RAG) pipeline:
1. Parse Obsidian markdown notes.
2. Chunk text and attach metadata.
3. Index chunks in vector + keyword stores.
4. Retrieve relevant chunks for user queries.
5. Expose the pipeline through CLI and MCP tools.
"""

__all__ = [
    "chunker",
    "config",
    "embedder",
    "indexer",
    "keyword_store",
    "mcp_server",
    "models",
    "parser",
    "retrieval",
    "service",
    "sync_state",
    "vector_store",
    "watcher",
]
