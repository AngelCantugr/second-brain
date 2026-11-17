"""Exercise the Streamable HTTP CLI against a disposable local index."""

import asyncio
import http.client
import json
import socket
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import Mock

import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from second_brain import mcp_server


def test_http_cli_uses_loopback_and_default_port(monkeypatch):
    server = Mock()
    build_server = Mock(return_value=server)
    monkeypatch.setattr(mcp_server, "build_server", build_server)
    monkeypatch.setattr(sys, "argv", ["second-brain-mcp", "--transport", "streamable-http"])

    mcp_server.run()

    build_server.assert_called_once_with("rag_config.toml", start_watcher=True)
    server.run.assert_called_once_with(
        transport="streamable-http", host="127.0.0.1", port=8765
    )


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--transport", "streamable-http", "--port", "0"], "between 1 and 65535"),
        (["--transport", "streamable-http", "--port", "-1"], "between 1 and 65535"),
        (["--transport", "streamable-http", "--port", "65536"], "between 1 and 65535"),
        (["--transport", "streamable-http", "--port", "invalid"], "invalid int value"),
        (["--port", "8765"], "--port requires --transport streamable-http"),
    ],
)
def test_invalid_http_options_fail_before_build(monkeypatch, capsys, args, message):
    build_server = Mock()
    monkeypatch.setattr(mcp_server, "build_server", build_server)
    monkeypatch.setattr(sys, "argv", ["second-brain-mcp", *args])

    with pytest.raises(SystemExit) as exc:
        mcp_server.run()

    assert exc.value.code == 2
    assert message in capsys.readouterr().err
    build_server.assert_not_called()


@asynccontextmanager
async def _session(url: str):
    async with streamable_http_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def _call(session: ClientSession, name: str, arguments: dict) -> dict:
    result = await session.call_tool(name, arguments)
    assert not result.is_error, result.content
    return result.structured_content or json.loads(result.content[0].text)


def _wait_for_listener(process: subprocess.Popen, port: int, log: Path) -> None:
    deadline = time.monotonic() + 15
    while process.poll() is None and time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except ConnectionRefusedError:
            time.sleep(0.05)
    pytest.fail(f"HTTP server did not start:\n{log.read_text(encoding='utf-8')}")


def test_two_http_clients_share_one_server(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    # An empty note adds graph metadata without contacting an embedder.
    (vault / "shared.md").touch()
    config = tmp_path / "rag_config.toml"
    config.write_text(
        'qdrant_path = "qdrant"\n'
        'fts_path = "fts.sqlite"\n'
        'sync_state_path = "sync_state.sqlite"\n'
        '[[banks]]\nid = "vault"\ntype = "knowledge-base"\npath = "vault"\nwatch_enabled = false\n',
        encoding="utf-8",
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    async def check_clients() -> None:
        url = f"http://127.0.0.1:{port}/mcp"
        async with _session(url) as first:
            async with _session(url) as second:
                first_tools, second_tools = await asyncio.gather(
                    first.list_tools(), second.list_tools()
                )
                first_names = {tool.name for tool in first_tools.tools}
                assert first_names == {tool.name for tool in second_tools.tools}
                assert {"rag.related", "rag.sync"} <= first_names
                note = {"file": {"bank": "vault", "path": "shared.md"}}
                assert not (await _call(first, "rag.related", note))["found"]
                result = await _call(second, "rag.sync", {"mode": "file", **note})
                assert result["errors"] == []
                assert result["processed"] == 1
                assert (await _call(first, "rag.related", note))["found"]
            assert (await _call(first, "rag.related", note))["found"]

    log = tmp_path / "server.log"
    with log.open("wb") as output, subprocess.Popen(
        [
            sys.executable, "-m", "second_brain.mcp_server", "--config", str(config),
            "--transport", "streamable-http", "--port", str(port),
        ],
        stdout=output,
        stderr=subprocess.STDOUT,
    ) as process:
        try:
            _wait_for_listener(process, port, log)
            for headers, status in [
                ({"Origin": "https://untrusted.example"}, 403),
                ({"Host": f"untrusted.example:{port}"}, 421),
            ]:
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                try:
                    connection.request("GET", "/mcp", headers=headers)
                    assert connection.getresponse().status == status
                finally:
                    connection.close()
            asyncio.run(asyncio.wait_for(check_clients(), timeout=15))
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
