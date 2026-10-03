import asyncio
from pathlib import Path

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from second_brain.config import RagConfig
from second_brain import mcp_server
from second_brain.service import RagService


class _CountingEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[1.0, 0.0, 0.0] for _ in texts]


def _build_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    config = RagConfig(
        vault_path=vault,
        qdrant_path=tmp_path / "qdrant",
        fts_path=tmp_path / "fts.sqlite",
        sync_state_path=tmp_path / "sync_state.sqlite",
    )
    service = RagService(config, use_in_memory_vector=True)
    embedder = _CountingEmbedder()
    service.embedder = embedder
    monkeypatch.setattr(mcp_server, "load_config", lambda _path: config)
    monkeypatch.setattr(mcp_server, "RagService", lambda _config: service)
    return mcp_server.build_server("unused"), embedder


@pytest.mark.parametrize("tool_name", ["rag.search", "rag.query"])
def test_registered_search_tools_reject_invalid_min_score_before_embedding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool_name: str,
) -> None:
    server, embedder = _build_server(tmp_path, monkeypatch)
    invalid_scores = [
        True,
        False,
        "0.5",
        10**400,
        -(10**400),
        -1.01,
        1.01,
        float("nan"),
        float("inf"),
        -float("inf"),
    ]

    for min_score in invalid_scores:
        before = embedder.calls
        try:
            result = asyncio.run(server.call_tool(tool_name, {"query": "hello", "min_score": min_score}))
        except Exception:
            result = None

        assert result is None
        assert embedder.calls == before


@pytest.mark.parametrize("tool_name", ["rag.search", "rag.query"])
@pytest.mark.parametrize("min_score", [-1, 0.5, 1, None])
def test_registered_search_tools_accept_integer_and_float_min_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool_name: str, min_score: int | float | None,
) -> None:
    server, embedder = _build_server(tmp_path, monkeypatch)

    result = asyncio.run(server.call_tool(tool_name, {"query": "hello", "min_score": min_score}))

    assert result
    assert embedder.calls == 1


@pytest.mark.parametrize("tool_name", ["rag.search", "rag.query"])
def test_registered_search_tool_schema_keeps_min_score_numeric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool_name: str,
) -> None:
    server, _embedder = _build_server(tmp_path, monkeypatch)

    tool = next(tool for tool in asyncio.run(server.list_tools()) if tool.name == tool_name)
    min_score_schema = tool.inputSchema["properties"]["min_score"]
    number_schema = next(option for option in min_score_schema["anyOf"] if option.get("type") == "number")

    assert number_schema["minimum"] == -1
    assert number_schema["maximum"] == 1


@pytest.mark.parametrize("tool_name", ["rag.search", "rag.query"])
def test_registered_search_tools_reject_invalid_recency_boost_before_embedding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool_name: str,
) -> None:
    server, embedder = _build_server(tmp_path, monkeypatch)

    for recency_boost in [True, False, "0.5", "1", -0.1, 1.1, float("nan"), float("inf"), None]:
        before = embedder.calls
        with pytest.raises(ToolError):
            asyncio.run(server.call_tool(tool_name, {"query": "hello", "recency_boost": recency_boost}))
        assert embedder.calls == before


@pytest.mark.parametrize("tool_name", ["rag.search", "rag.query"])
def test_registered_search_tools_accept_integer_and_float_recency_boost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool_name: str,
) -> None:
    server, embedder = _build_server(tmp_path, monkeypatch)

    assert asyncio.run(server.call_tool(tool_name, {"query": "hello"}))
    for recency_boost in (0, 0.5, 1):
        result = asyncio.run(server.call_tool(tool_name, {"query": "hello", "recency_boost": recency_boost}))
        assert result
    assert embedder.calls == 4


@pytest.mark.parametrize("tool_name", ["rag.search", "rag.query"])
def test_registered_search_tool_schema_bounds_recency_boost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool_name: str,
) -> None:
    server, _embedder = _build_server(tmp_path, monkeypatch)

    tool = next(tool for tool in asyncio.run(server.list_tools()) if tool.name == tool_name)
    schema = tool.inputSchema["properties"]["recency_boost"]

    assert schema["type"] == "number"
    assert schema["minimum"] == 0
    assert schema["maximum"] == 1


@pytest.mark.parametrize("tool_name", ["rag.search", "rag.query"])
@pytest.mark.parametrize("modified_since", [
    "2026-10-01T00:00:00+00:99",
    "2026-10-01T00:00:00+00:00:99",
    "2026-10-01T00:00:00+24:00",
])
def test_registered_search_tools_reject_invalid_offset_before_embedding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool_name: str, modified_since: str,
) -> None:
    server, embedder = _build_server(tmp_path, monkeypatch)
    before = embedder.calls

    with pytest.raises(ToolError):
        asyncio.run(server.call_tool(tool_name, {
            "query": "hello", "filters": {"modified_since": modified_since},
        }))

    assert embedder.calls == before
