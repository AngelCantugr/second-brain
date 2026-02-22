# How I made Claude Code aware of my Obsidian vault via MCP

## Architecture

Obsidian markdown files flow through an ingestion pipeline into a local ChromaDB index. An MCP server on stdio exposes 4 tools that Claude Code can call.

- `search_vault`: semantic retrieval with score + excerpt.
- `get_note`: full markdown fetch by vault-relative path.
- `list_topics`: top-level folder counts.
- `recent_activity`: notes modified in the last N days.

## Ingestion internals

Each note is parsed with frontmatter support, then chunked by heading context. Chunks are embedded with Ollama (`nomic-embed-text`) and upserted into Chroma with metadata:

- `file_path`
- `heading`
- `tags`
- `date_modified`
- `chunk_index`

Incremental indexing is checkpoint-driven using per-file modification times.

## MCP implementation details

The server uses a service layer (`VaultService`) for business logic and a thin MCP registration layer for tool definitions.

This separation made it easy to write deterministic tests against tool behavior without requiring a running MCP process.

## Local configuration

Claude Code can connect through `.mcp.json` using:

- command: `uv`
- args: `run python -m vault_mcp.server`
- cwd: `agents/codex/vault-mcp`

## Practical prompts

- "What did I learn about rate limiting?"
- "Show my recent notes about MCP tools"
- "Open the note on embedding chunking strategy"

## What I’d improve next

- Better chunk ranking and reranking.
- Richer metadata filters (tags/date/topic) in `search_vault`.
- Optional hybrid retrieval (keyword + vector).
