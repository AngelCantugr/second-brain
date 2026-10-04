"""Exercise real MCP startup without an embedding service or the user's vault."""

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def test_stdio_server_initializes_and_serves_status(tmp_path: Path) -> None:
    """Catch SDK incompatibilities hidden by direct tool-call unit tests."""
    vault = tmp_path / "vault"
    vault.mkdir()
    config = tmp_path / "rag_config.toml"
    config.write_text(
        "\n".join(
            f"{key} = {json.dumps(str(value))}"
            for key, value in {
                "vault_path": vault,
                "qdrant_path": tmp_path / "qdrant",
                "fts_path": tmp_path / "fts.sqlite",
                "sync_state_path": tmp_path / "sync_state.sqlite",
            }.items()
        ),
        encoding="utf-8",
    )

    async def check_session() -> None:
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "second_brain.mcp_server", "--config", str(config)],
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert {tool.name for tool in tools.tools} == {
                    "rag.query", "rag.search", "rag.note_context", "rag.related",
                    "rag.connections", "rag.map", "rag.sync", "rag.status", "rag.health",
                }
                result = await session.call_tool("rag.status", {})
                assert not result.is_error
                assert result.content
                status = result.structured_content or json.loads(result.content[0].text)
                assert (status["stale_files"], status["untracked_files"], status["missing_files"]) == (0, 0, 0)
                invalid = await session.call_tool(
                    "rag.search", {"query": "hello", "recency_boost": True},
                )
                assert invalid.is_error

    asyncio.run(asyncio.wait_for(check_session(), timeout=30))
