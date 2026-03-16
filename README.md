# Obsidian RAG MCP

Local-first RAG for Obsidian vaults with:
- Incremental indexing
- Hybrid retrieval (vector + keyword)
- MCP tool interface

## Documentation

- Beginner guide: [`docs/rag-beginners-guide.md`](docs/rag-beginners-guide.md)
- End-to-end query trace: [`docs/query-trace-end-to-end.md`](docs/query-trace-end-to-end.md)
- In-code docs: each module in `obsidian_rag/` now includes module/class/function docstrings.

## Quick Start

1. Sync environment with `uv`:

```bash
uv sync --dev
```

2. Create config:

```bash
cp rag_config.example.toml rag_config.toml
```

3. Sync vault:

```bash
uv run obsidian-rag sync --mode full
```

4. Run MCP server:

```bash
uv run obsidian-rag-mcp --config rag_config.toml
```

## Development

```bash
uv run pytest -q
```
