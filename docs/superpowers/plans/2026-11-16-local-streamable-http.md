# Local Streamable HTTP Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans or superpowers:subagent-driven-development. Steps use checkboxes for tracking.

**Goal.** Let local AI clients share one Second Brain MCP server.

**Architecture.** Add a transport choice to the existing entry point. Both modes build the same server and HTTP delegates to the MCP SDK.

**Tech stack.** Python, MCP SDK 2.3, pytest.

**Spec.** [Local Streamable HTTP design](../specs/2026-11-16-local-streamable-http-design.md).

## Global constraints

- Keep stdio as the default. HTTP requires explicit selection.
- Bind the HTTP server to `127.0.0.1`, with port `8765` by default and endpoint `/mcp`.
- Keep the SDK's localhost Host and Origin protections and reuse the existing watcher and index service.

## Review focus

- Invalid ports fail before a vault is opened. An HTTP-only port option in stdio mode produces a clear usage error.
- An untrusted Host or Origin is rejected by the transport.
- Closing a client leaves the remaining client able to call tools.

## Implement and verify the transport

- [x] Add `tests/test_mcp_http.py`. Drive the real entry point with a disposable vault and two initialized SDK HTTP clients. Assert that both list the tools, a sync from one adds a note the other can find in the graph, and the first can call a tool after the second disconnects. Verify Host and Origin rejection.
- [x] Add CLI tests in the same file for the default HTTP bind and port and pre-start argument errors. Run `uv run --locked pytest -q tests/test_mcp_http.py` and confirm the new checks fail because the HTTP flags are missing.
- [x] In `second_brain/mcp_server.py`, keep `run() -> None`. Add `--transport {stdio,streamable-http}` with default `stdio` and HTTP-only `--port` with valid range `1` to `65535`. Build one server and call `server.run()` for stdio or `server.run(transport="streamable-http", host="127.0.0.1", port=port)` for HTTP.
- [x] Update `README.md` with the launch command, an HTTP client example, and how to migrate all clients sharing the index and run `rag.sync` through the shared process.
- [x] Run `uv run --locked pytest -q tests/test_mcp_http.py tests/test_mcp_stdio.py`, then `uv run --locked pytest -q` and `git diff --check`.
