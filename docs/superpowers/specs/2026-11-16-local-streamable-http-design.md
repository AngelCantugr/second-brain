# Local Streamable HTTP for Second Brain

## Goal

Start one Second Brain MCP server for a vault and share it across local AI sessions. The process owns one `RagService` and starts one vault watcher when `watch_enabled` is set. Existing stdio clients can continue to launch `second-brain-mcp` with their current arguments.

## Approaches

- Built-in Streamable HTTP, selected in conversation. Use the MCP Python SDK's HTTP transport on the existing server. This reuses the tool registrations, config loading, and watcher and adds one launch mode.
- Local HTTP bridge. Forward HTTP connections to one stdio subprocess. This preserves the existing server entry point and adds a separate process and configuration to run and troubleshoot.

## Interface and scope

```bash
second-brain-mcp --config /absolute/path/to/rag_config.toml --transport streamable-http --port 8765
```

Connect each client to `http://127.0.0.1:8765/mcp`. The proposed defaults are TCP port `8765` and the SDK's `/mcp` path. The port can be overridden in HTTP mode. The server binds only to IPv4 loopback and uses the SDK's localhost Host and Origin protections. Local processes that can reach the port can read vault data and call `rag.sync`. Network access, client authentication, and service auto-start are outside this change.

Build the same server once per process for either transport. Reuse the service's existing index synchronization. On migration, close stdio processes using the same index and switch their client registrations from command and args to the HTTP URL. Document the launch command, a client config example, and using `rag.sync` through the shared process while it owns the index.

## Acceptance

- The existing stdio protocol test passes with the original invocation.
- Two MCP clients can connect to the same HTTP server at the same time, list the tools, and make tool calls against the same index. One client remains usable after the other disconnects.
- The listener binds to loopback. Invalid HTTP port inputs fail before the service starts, and the server rejects an untrusted Origin.
