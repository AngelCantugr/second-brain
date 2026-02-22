# Vault MCP Semantic Search Design

## Goal
Build a local semantic search MCP server for the Obsidian vault so Claude Code can retrieve notes, excerpts, and activity with low-latency local inference.

## Scope
- Ingest markdown notes from `Obsidian/`.
- Parse frontmatter and capture note metadata.
- Chunk note bodies with heading context.
- Embed chunks with local Ollama model `nomic-embed-text`.
- Persist vectors in local ChromaDB (`~/.vault-index/`).
- Expose MCP stdio tools:
  - `search_vault(query, top_k=5)`
  - `get_note(path)`
  - `list_topics()`
  - `recent_activity(days=7)`

## Non-goals
- Cloud vector databases.
- Multi-user auth.
- Realtime filesystem watchers (batch reindex only for v1).

## Data Model
Each chunk document stored in Chroma contains:
- `id`: deterministic id `sha1(file_path + chunk_index + heading + chunk_text)`
- `document`: chunk text (with heading prefix)
- `metadata`:
  - `file_path` (vault-relative path)
  - `heading` (closest markdown heading)
  - `tags` (comma-delimited list)
  - `date_modified` (epoch seconds)
  - `chunk_index` (int)
  - `title` (basename without extension)

## Ingestion Strategy
1. Walk `Obsidian/` recursively.
2. Skip `_templates/` and `Archive/` path segments.
3. For changed files only (mtime > checkpoint), parse and chunk.
4. Embed new chunks with Ollama.
5. Replace chunk rows for changed files in Chroma.
6. Persist checkpoint map `{file_path: last_mtime}` at `~/.vault-index/checkpoint.json`.

## Incremental Re-indexing
- First run: index all files.
- Subsequent runs:
  - If path missing from checkpoint => index.
  - If mtime newer than checkpoint => re-index.
  - Else skip.

## MCP Contracts
- `search_vault` returns `[{note, excerpt, score, path}]`.
- `get_note` returns full markdown body for a vault-relative path.
- `list_topics` returns folder tree counts keyed by top-level topic folder.
- `recent_activity` returns notes modified in last N days sorted newest first.

## Project Layout
- `src/vault_mcp/ingest.py` ingestion entrypoint.
- `src/vault_mcp/server.py` MCP entrypoint.
- `src/vault_mcp/chunker.py` chunking utilities.
- `src/vault_mcp/metadata.py` frontmatter parsing and vault walking.
- `src/vault_mcp/index_store.py` Chroma read/write wrappers.

## Verification
- `uv run pytest -v`
- `uv run python -m vault_mcp.ingest`
- `uv run python -m vault_mcp.server`
- manual Claude Code prompt test through `.mcp.json`.
