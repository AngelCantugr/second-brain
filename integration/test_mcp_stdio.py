"""Real MCP integration tests.

Launches the actual `second-brain-mcp` entrypoint as a subprocess and drives
it over the real stdio JSON-RPC transport with an `mcp` ClientSession. This
exercises transport -> tool registration -> serialization -> RagService ->
real Ollama + embedded Qdrant, end to end -- the one path nothing else in
this repo covers.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from mcp import StdioServerParameters
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

EXPECTED_TOOLS = {
    "rag.query",
    "rag.search",
    "rag.note_context",
    "rag.related",
    "rag.connections",
    "rag.map",
    "rag.sync",
    "rag.status",
    "rag.health",
}


async def _call(session: ClientSession, name: str, arguments: dict | None = None) -> dict:
    result = await session.call_tool(name, arguments or {})
    assert not result.isError, f"{name} returned an error: {result.content}"
    # The tools return a bare `dict`, which FastMCP does not treat as a
    # structured-output schema (see second_brain/mcp_server.py) -- so the
    # payload arrives as JSON text in `content`, not `structuredContent`.
    if result.structuredContent is not None:
        return result.structuredContent
    assert result.content, f"{name} returned neither structured content nor text content"
    first = result.content[0]
    assert isinstance(first, TextContent), f"{name} returned unexpected content type: {first!r}"
    return json.loads(first.text)


async def _run_session(config_path: Path, python_executable: str, body):
    """Start the MCP server subprocess, initialize a session, and run `body`."""

    params = StdioServerParameters(
        command=python_executable,
        args=["-m", "second_brain.mcp_server", "--config", str(config_path)],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await body(session)


@pytest.fixture(scope="module")
def synced_config_path(config_path: Path, rag_service) -> Path:
    """Ensure the shared rag_service fixture has synced before the subprocess runs.

    The subprocess launches its own RagService against the same on-disk
    stores, so it sees the data `rag_service` already synced in-process.
    """

    return config_path


def test_lists_all_nine_tools(synced_config_path: Path, python_executable: str) -> None:
    async def body(session: ClientSession):
        return await session.list_tools()

    result = asyncio.run(_run_session(synced_config_path, python_executable, body))
    tool_names = {tool.name for tool in result.tools}
    assert tool_names == EXPECTED_TOOLS


def test_rag_health_over_stdio(synced_config_path: Path, python_executable: str) -> None:
    async def body(session: ClientSession):
        return await _call(session, "rag.health")

    health = asyncio.run(_run_session(synced_config_path, python_executable, body))
    assert health == {"qdrant": True, "ollama": True, "fts": True}


def test_rag_status_over_stdio(synced_config_path: Path, python_executable: str) -> None:
    async def body(session: ClientSession):
        return await _call(session, "rag.status")

    status = asyncio.run(_run_session(synced_config_path, python_executable, body))
    assert status["model"] == "nomic-embed-text"
    assert status["model_available"] is True
    assert status["index_size"] > 0
    assert status["graph_nodes"] >= 4


def test_rag_search_over_stdio(synced_config_path: Path, python_executable: str) -> None:
    async def body(session: ClientSession):
        return await _call(session, "rag.search", {"query": "retrieval augmented generation", "top_k": 5})

    result = asyncio.run(_run_session(synced_config_path, python_executable, body))
    assert result["query"] == "retrieval augmented generation"
    assert len(result["hits"]) > 0
    paths = {hit["metadata"].get("path") for hit in result["hits"]}
    assert "alpha.md" in paths


def test_rag_query_over_stdio(synced_config_path: Path, python_executable: str) -> None:
    async def body(session: ClientSession):
        return await _call(session, "rag.query", {"query": "what is the Gamma Project", "top_k": 5})

    result = asyncio.run(_run_session(synced_config_path, python_executable, body))
    assert result["answer_draft"]
    assert result["citations"]
    assert len(result["chunks"]) == len(result["debug_scores"])


def test_rag_related_over_stdio(synced_config_path: Path, python_executable: str) -> None:
    async def body(session: ClientSession):
        return await _call(session, "rag.related", {"note_path": "alpha.md", "top_k": 5})

    result = asyncio.run(_run_session(synced_config_path, python_executable, body))
    assert result["found"] is True
    neighbor_paths = {n["path"] for n in result["neighbors"]}
    assert "beta.md" in neighbor_paths


def test_rag_connections_over_stdio(synced_config_path: Path, python_executable: str) -> None:
    async def body(session: ClientSession):
        return await _call(session, "rag.connections", {"note_a": "alpha.md", "note_b": "beta.md"})

    result = asyncio.run(_run_session(synced_config_path, python_executable, body))
    assert result["connected"] is True
    assert result["closeness"] > 0.0


def test_rag_map_over_stdio(synced_config_path: Path, python_executable: str) -> None:
    async def body(session: ClientSession):
        return await _call(session, "rag.map")

    result = asyncio.run(_run_session(synced_config_path, python_executable, body))
    assert result["note_count"] >= 4
    assert isinstance(result["clusters"], list)


def test_rag_note_context_over_stdio(synced_config_path: Path, python_executable: str) -> None:
    async def body(session: ClientSession):
        return await _call(session, "rag.note_context", {"note_path": "beta.md"})

    result = asyncio.run(_run_session(synced_config_path, python_executable, body))
    assert result["chunk_count"] > 0
    # beta.md wikilinks [[alpha]] (raw wikilink target, not a resolved path)...
    assert "alpha" in result["outlinks"]
    # ...and alpha.md wikilinks [[beta]], so alpha.md is a backlink of beta.md.
    assert "alpha.md" in result["backlinks"]


def test_rag_sync_over_stdio(synced_config_path: Path, python_executable: str) -> None:
    async def body(session: ClientSession):
        return await _call(session, "rag.sync", {"mode": "incremental"})

    result = asyncio.run(_run_session(synced_config_path, python_executable, body))
    assert result["errors"] == []
