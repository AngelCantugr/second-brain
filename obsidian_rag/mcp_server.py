"""MCP server exposing RAG tools to AI clients."""

from __future__ import annotations

import argparse

from obsidian_rag.config import load_config
from obsidian_rag.service import RagService


def build_server(config_path: str):
    """Build and return FastMCP server with all RAG tools registered."""

    from mcp.server.fastmcp import FastMCP

    config = load_config(config_path)
    service = RagService(config)
    mcp = FastMCP("obsidian-rag")

    @mcp.tool(name="rag.query")
    def rag_query(query: str, filters: dict | None = None, top_k: int = 8) -> dict:
        return service.query(query=query, filters=filters, top_k=top_k)

    @mcp.tool(name="rag.search")
    def rag_search(query: str, filters: dict | None = None, top_k: int = 10) -> dict:
        return service.search(query=query, filters=filters, top_k=top_k)

    @mcp.tool(name="rag.note_context")
    def rag_note_context(note_path: str) -> dict:
        return service.note_context(note_path=note_path)

    @mcp.tool(name="rag.sync")
    def rag_sync(mode: str = "incremental", file_path: str | None = None) -> dict:
        return service.sync(mode=mode, file_path=file_path)

    @mcp.tool(name="rag.status")
    def rag_status() -> dict:
        return service.status()

    @mcp.tool(name="rag.health")
    def rag_health() -> dict:
        return service.health()

    return mcp


def run() -> None:
    """Executable entrypoint for running the MCP server process."""

    parser = argparse.ArgumentParser(description="Obsidian RAG MCP server")
    parser.add_argument("--config", default="rag_config.toml")
    args = parser.parse_args()
    server = build_server(args.config)
    server.run()


if __name__ == "__main__":
    run()
