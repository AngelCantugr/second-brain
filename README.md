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

1. Initialize config in your current folder:

```bash
uv run obsidian-rag init
```

This creates `rag_config.toml` using `$CWD` for `vault_path` and local data stores.

1. Sync vault:

```bash
uv run obsidian-rag sync --mode full
```

1. Run MCP server:

```bash
uv run obsidian-rag-mcp --config rag_config.toml
```

## Development

```bash
uv run pytest -q
```

## Run With pipx

Install directly from this repository:

```bash
pipx install .
```

Then in any directory you want to use as your workspace:

```bash
obsidian-rag init
obsidian-rag sync --mode full
obsidian-rag-mcp --config rag_config.toml
```
