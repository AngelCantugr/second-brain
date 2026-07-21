# Obsidian RAG MCP

A local-first Retrieval-Augmented Generation (RAG) pipeline for [Obsidian](https://obsidian.md) vaults. Index your notes once, then query them semantically from the command line or through any MCP-compatible AI client (Claude Desktop, Cursor, VS Code Copilot, etc.).

## Features

- **Hybrid retrieval** — combines semantic vector search (Qdrant) with full-text keyword search (SQLite FTS5) via Reciprocal Rank Fusion
- **Note-association graph** — a precomputed graph blending semantic similarity, wikilinks, shared tags, and co-mentions, so you can ask "what's related to this note, and why?" instead of only "what matches this text?"
- **Incremental indexing** — only re-indexes files that changed since the last sync, and only recomputes the graph edges affected by that change
- **MCP tool interface** — expose your vault as tools that any MCP client can call
- **Local-first** — all embeddings and storage run on your machine; nothing leaves it
- **Configurable chunking** — tune chunk size, overlap, and glob exclusions per vault
- **Privacy controls** — redact sensitive patterns before they reach the index

## Prerequisites

| Dependency | Purpose | Install |
|---|---|---|
| Python >= 3.11 | Runtime | [python.org](https://www.python.org/downloads/) |
| [Ollama](https://ollama.com) | Local embeddings | `brew install ollama` |
| `nomic-embed-text` model | Default embedding model | `ollama pull nomic-embed-text` |

> Qdrant runs embedded via `qdrant-client` — no separate server needed.

## Installation

### Option A — pipx (recommended for end-users)

Install directly from GitHub and make the CLI globally available:

```bash
pipx install git+https://github.com/AngelCantugr/second-brain.git
```

Or from a local clone:

```bash
git clone https://github.com/AngelCantugr/second-brain.git
cd second-brain
pipx install .
```

### Option B — uv (recommended for development)

```bash
git clone https://github.com/AngelCantugr/second-brain.git
cd second-brain
uv sync --dev
```

Then prefix all commands with `uv run`:

```bash
uv run obsidian-rag --help
```

### Option C — pip in a virtual environment

```bash
git clone https://github.com/AngelCantugr/second-brain.git
cd second-brain
python -m venv .venv && source .venv/bin/activate
pip install .
```

## CLI Usage

Point the CLI at any directory that contains (or will contain) an Obsidian vault.

### 1. Initialize

Creates `rag_config.toml` and a `data/` directory in the current working directory:

```bash
cd ~/my-obsidian-vault
obsidian-rag init
```

Use `--force` to regenerate an existing config:

```bash
obsidian-rag init --force
```

### 2. Sync the vault

| Mode | When to use |
|---|---|
| `full` | First run, or after bulk changes |
| `incremental` | Routine updates (only changed files) |
| `file` | Re-index a single file |

```bash
# First-time full index
obsidian-rag sync --mode full

# Pick up recent changes
obsidian-rag sync --mode incremental

# Re-index one file
obsidian-rag sync --mode file --file-path "Projects/my-note.md"
```

### 3. Search and query

```bash
# Hybrid search — returns scored chunks
obsidian-rag search "async rust patterns"

# Query — returns an answer draft with citations
obsidian-rag query "What are my notes on system design?"

# Adjust result count
obsidian-rag query "project ideas" --top-k 5
```

### 4. Operational commands

```bash
# Runtime status (model, index size, last sync)
obsidian-rag status

# Health check (Qdrant, Ollama, FTS)
obsidian-rag health
```

### Using a custom config path

All commands accept `--config` to point at a non-default config file:

```bash
obsidian-rag --config ~/vaults/work/rag_config.toml sync --mode full
```

## MCP Server

The MCP server exposes your vault as callable tools for any compatible AI client.

### Start the server

```bash
obsidian-rag-mcp --config /absolute/path/to/rag_config.toml
```

### Available MCP tools

| Tool | Description |
|---|---|
| `rag.query` | Hybrid search + answer draft with citations |
| `rag.search` | Raw hybrid search returning scored chunks |
| `rag.note_context` | Chunk summary and outlinks for a specific note |
| `rag.related` | Notes associated with one note, ranked with a per-signal score breakdown |
| `rag.connections` | How closely two notes are associated, direct or via the strongest path between them |
| `rag.map` | Vault-wide summary of note clusters, orphans, and bridge notes |
| `rag.sync` | Trigger vault re-index (`full` or `incremental`), including the graph |
| `rag.status` | Runtime status (now including graph node/edge counts) |
| `rag.health` | Health check |

### Claude Desktop

Add the following to your `claude_desktop_config.json` (typically `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
	"mcpServers": {
		"obsidian-rag": {
			"command": "obsidian-rag-mcp",
			"args": ["--config", "/absolute/path/to/your/vault/rag_config.toml"]
		}
	}
}
```

> If you installed with `uv`, use the full path to the venv binary:
> `"/path/to/second-brain/.venv/bin/obsidian-rag-mcp"`

### VS Code (GitHub Copilot)

Add to your `.vscode/mcp.json` or user-level MCP settings:

```json
{
	"servers": {
		"obsidian-rag": {
			"type": "stdio",
			"command": "obsidian-rag-mcp",
			"args": ["--config", "/absolute/path/to/your/vault/rag_config.toml"]
		}
	}
}
```


## Configuration Reference

`obsidian-rag init` generates a `rag_config.toml` with these defaults:

```toml
# Path to your Obsidian vault (or any markdown directory)
vault_path = "$CWD"

# Local storage — relative paths resolve from the config file's directory
qdrant_path     = "$CWD/data/qdrant"
fts_path        = "$CWD/data/fts.sqlite"
sync_state_path = "$CWD/data/sync_state.sqlite"

# Qdrant collection name
collection_name = "obsidian_chunks"

# Ollama settings
ollama_url      = "http://127.0.0.1:11434"
embedding_model = "nomic-embed-text"

# Chunking
chunk_size    = 500
chunk_overlap = 80

# Auto-watch vault for file changes (used by long-running processes)
watch_enabled = true

# Glob patterns excluded from indexing
exclude_globs = [".obsidian/**", ".git/**", "Templates/**"]

# Max chunks returned per query
max_context_chunks = 8

# Regex patterns to redact from chunk text before indexing
redact_patterns = []

# Note-association graph (used by rag.related / rag.connections / rag.map)
graph_enabled = true             # set false to skip graph computation entirely
graph_knn_k = 8                  # max semantic neighbors considered per note
graph_semantic_min = 0.35        # minimum cosine similarity to become a semantic candidate
graph_min_edge_score = 0.15      # minimum composite score to persist an edge
                                  # (edges with a wikilink or co-mention persist regardless)
graph_weight_semantic = 0.5      # composite score weights (should sum to 1.0)
graph_weight_link = 0.25
graph_weight_tag = 0.15
graph_weight_comention = 0.10
graph_comention_cap = 3          # co-mentions beyond this count don't add further score
graph_comention_max_fanout = 20  # notes linking to more targets than this don't contribute
                                  # co-mention pairs (keeps hub/MOC notes from exploding edge count)
```

`$CWD` resolves to the working directory at the time `init` is run. Standard `~` and environment variable expansions are supported in all path fields.

## Project Structure

```
obsidian_rag/
├── cli.py           # CLI entrypoint (obsidian-rag)
├── mcp_server.py    # MCP server entrypoint (obsidian-rag-mcp)
├── service.py       # High-level facade shared by CLI and MCP
├── indexer.py       # Orchestrates parse, chunk, embed, store
├── parser.py        # Markdown + frontmatter parser
├── chunker.py       # Text chunking with overlap
├── embedder.py      # Ollama embedding client
├── vector_store.py  # Qdrant vector store wrapper
├── keyword_store.py # SQLite FTS5 keyword store
├── retrieval.py     # Reciprocal Rank Fusion merge
├── graph.py         # Note-association graph: scoring, storage, builder, queries
├── sync_state.py    # Incremental sync state tracking
├── watcher.py       # File-system watcher (watchfiles)
└── config.py        # TOML config loader
```

## Development

```bash
uv sync --dev
uv run pytest -q
```

## Further Reading

- [Beginner's guide to RAG](docs/rag-beginners-guide.md) — how the pipeline works from first principles
- [End-to-end query trace](docs/query-trace-end-to-end.md) — follow a query through every layer
