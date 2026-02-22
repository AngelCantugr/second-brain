# vault-mcp

Local semantic search MCP server for an Obsidian vault.

## Requirements

- Python 3.13 (`chromadb` compatibility)
- `uv`
- Ollama running locally
- Ollama model: `nomic-embed-text`

## Setup

```bash
uv sync --python 3.13 --group dev
ollama pull nomic-embed-text
```

## Ingest Vault Notes

By default, ingestion scans `../../../../Obsidian` relative to this directory and stores the index in `~/.vault-index`.

```bash
uv run --python 3.13 python -m vault_mcp.ingest
```

Output example:

```text
scanned=120 indexed_files=8 indexed_chunks=44
```

Incremental behavior:
- First run indexes all notes.
- Next runs only re-index files with newer modification times.

## Run MCP Server (stdio)

```bash
uv run --python 3.13 python -m vault_mcp.server
```

Tools exposed:
- `search_vault(query, top_k=5)`
- `get_note(path)`
- `list_topics()`
- `recent_activity(days=7)`

## Claude Code MCP Config

Copy the shape from `.mcp.json.example`:

```json
{
  "mcpServers": {
    "vault": {
      "command": "uv",
      "args": ["run", "python", "-m", "vault_mcp.server"],
      "cwd": "agents/codex/vault-mcp"
    }
  }
}
```

## Troubleshooting

- `ModuleNotFoundError`: run `uv sync --python 3.13 --group dev` again.
- `ollama` connection errors: ensure `ollama serve` is running and model is pulled.
- No search results: run ingestion first.
