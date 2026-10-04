# Second Brain MCP

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
uv run second-brain --help
```

### Option C — pip in a virtual environment

```bash
git clone https://github.com/AngelCantugr/second-brain.git
cd second-brain
python -m venv .venv && source .venv/bin/activate
pip install .
```

## Upgrading from `obsidian-rag`

This project was previously named `obsidian-rag`. If you have an existing install:

1. **Reinstall under the new name** — the console scripts changed from `obsidian-rag`/`obsidian-rag-mcp` to `second-brain`/`second-brain-mcp`.
   - pipx: `pipx uninstall obsidian-rag-mcp` then reinstall per [Installation](#installation) above
   - uv/pip: pull the latest code and re-run `uv sync --dev` or `pip install .`
2. **Re-index your vault** — the default Qdrant collection name changed from `obsidian_chunks` to `second_brain_chunks`. Your existing index is not migrated automatically; run a full sync to rebuild it:
   ```bash
   second-brain sync --mode full
   ```
   If you had set `collection_name` explicitly in your `rag_config.toml`, either update it to `second_brain_chunks` or leave your custom value as-is — it isn't affected by this rename.
3. **Update MCP client configs** — if you registered the server with Claude Desktop, VS Code, or another MCP client, update the server key and `command` from `obsidian-rag`/`obsidian-rag-mcp` to `second-brain`/`second-brain-mcp` (see [MCP Server](#mcp-server) below).

Chunk identities now include each note's vault-relative path, so identical notes
keep independent entries. On the first sync after this upgrade, eligible tracked
notes from the old identity scheme are reindexed once even when their content is
unchanged. This incurs one embedding pass per legacy note; file sync upgrades only
the selected note. Normal sync removes index entries and checkpoints for tracked
excluded notes; re-including those notes makes them untracked and reindexed.
Legacy IDs are removed from both stores as each note is replaced. Existing hashes,
timestamps, and watcher metadata are preserved when the sync schema is upgraded.

A note's migration checkpoint advances only after both index stores and its graph
metadata succeed. Embedding failures preserve existing chunks. Later failures can
leave the stores temporarily different because SQLite and Qdrant do not share a
transaction; rerunning sync discovers leftover IDs in both stores and converges.
Deleted-note cleanup follows the same retry rule. This upgrade does not reset data
and does not automatically roll back earlier successful store operations. If you
need to downgrade the application, restore a backup of all index stores together
or rebuild with a full sync; older code cannot understand the new identity scheme.

## CLI Usage

Point the CLI at any directory that contains (or will contain) an Obsidian vault.

### 1. Initialize

Creates `rag_config.toml` and a `data/` directory in the current working directory:

```bash
cd ~/my-obsidian-vault
second-brain init
```

Use `--force` to regenerate an existing config:

```bash
second-brain init --force
```

### 2. Sync the vault

| Mode | When to use |
|---|---|
| `full` | First run, or after bulk changes |
| `incremental` | Routine updates (only changed files) |
| `file` | Re-index a single file |

```bash
# First-time full index
second-brain sync --mode full

# Pick up recent changes
second-brain sync --mode incremental

# Re-index one file
second-brain sync --mode file --file-path "Projects/my-note.md"
```

### 3. Search and query

```bash
# Hybrid search — returns scored chunks
second-brain search "async rust patterns"

# Query — returns retrieved chunks with citations
second-brain query "What are my notes on system design?"

# Adjust result count
second-brain query "project ideas" --top-k 5
```

### 4. Operational commands

```bash
# Runtime status (model, index size, last sync)
second-brain status

# Health check (Qdrant, Ollama, FTS)
second-brain health
```

### Using a custom config path

All commands accept `--config` to point at a non-default config file:

```bash
second-brain --config ~/vaults/work/rag_config.toml sync --mode full
```

## MCP Server

The MCP server exposes your vault as callable tools for any compatible AI client.

The server uses MCP Python SDK v2 (`mcp>=2.3.0,<3`). After updating a local
clone, refresh an existing pipx installation with `pipx install --force .`.

### Start the server

```bash
second-brain-mcp --config /absolute/path/to/rag_config.toml
```

### Available MCP tools

| Tool | Description |
|---|---|
| `rag.query` | Hybrid search + citations |
| `rag.search` | Raw hybrid search returning scored chunks |
| `rag.note_context` | Chunk summary and outlinks for a specific note |
| `rag.related` | Notes associated with one note, ranked with a per-signal score breakdown |
| `rag.connections` | How closely two notes are associated, direct or via the strongest path between them |
| `rag.map` | Vault-wide summary of note clusters, orphans, and bridge notes |
| `rag.sync` | Trigger vault re-index (`full` or `incremental`), including the graph |
| `rag.status` | Index, watcher, model, and graph status, including vault staleness counts |
| `rag.health` | Health check |

`rag.map` constructs a transient graph from stored metadata and centroids for
notes eligible under the current `exclude_globs`. Excluded notes cannot contribute
co-mention evidence, consume semantic neighbor slots, or affect cluster labels,
orphans, and bridges. The view uses the configured scoring weights, co-mention
cap/fanout, and semantic neighbor limits; `graph_enabled = false` leaves eligible
notes as orphans. Changing exclusions takes effect on the next map call without
a sync, and map never writes index or graph state. New or edited note content still
requires sync to refresh its stored metadata.

Map recomputes candidates and scores on every call. Its pairwise cosine matrix
uses O(n²) memory and pair comparisons for n eligible stored centroids, so large
vaults may incur more latency than reading persisted edges. No large-vault
performance guarantee is made.

### `rag.status` fields

`rag.status` reports runtime configuration and index state without syncing files
or changing the vault or sync-state records. It scans eligible markdown paths
and reads and hashes tracked files, so the check is read-only but its cost grows
with the number and size of tracked notes. It does not expose note contents.

| Field | Meaning |
|---|---|
| `watch_enabled` | Configured watcher setting (`true` or `false`). |
| `watcher_state` | `enabled` or `disabled`, derived from that setting; it is not a liveness check. |
| `watcher_last_event` | Unix epoch timestamp in seconds for the last observed eligible markdown watcher event; `null` until one is observed. It records event observation independently of indexing success. |
| `max_context_chunks` | Configured maximum number of chunks used for context. |
| `model` | Configured embedding model name. |
| `model_available` | Whether the embedding model health check succeeds. |
| `index_size` | Number of chunks in the keyword index. |
| `last_sync_timestamp` | Latest note-state update timestamp in Unix epoch seconds, or `null` when no note is tracked. Watcher events do not update it. |
| `last_tracked_files` | Number of note paths recorded in sync state. This may include paths excluded by current scanner settings. |
| `stale_files` | Eligible checkpointed files still on disk whose content hash or modification time differs from recorded state, or whose checkpoint has a pending replacement or requires identity migration. An unreadable checkpointed file is counted as stale. |
| `untracked_files` | Eligible markdown files on disk without a successful checkpoint, including paths with only a pending replacement. |
| `missing_files` | Eligible checkpointed or pending-replacement paths that no longer exist on disk. |
| `graph_nodes` / `graph_edges` | Current graph node and edge counts. |
| `graph_last_built` | Timestamp recorded for the last graph build, or `null` if no build is recorded. |

Eligibility follows the scanner: files must end in `.md`, paths containing a
hidden segment are skipped, and configured `exclude_globs` are applied to disk
and tracked paths. A quiet vault or an old `watcher_last_event` cannot establish
that a watcher is dead; an idle watcher has no event to report, and a disabled
watcher can retain a timestamp from before it was disabled.

An mtime-only change is reported as stale even if the content hash is unchanged.
Incremental sync compares content hashes and may leave that mtime warning until
a full sync refreshes the recorded mtime.

`rag.search` and `rag.query` accept an optional `min_score` cosine similarity
threshold from `-1.0` to `1.0`. When set, hits below the threshold are removed
before `top_k` is applied, and keyword-only hits are omitted because they have no
semantic similarity score. The default is no threshold. Each hit includes
`semantic_score` (cosine similarity) and `keyword_score` (SQLite FTS5 BM25
relevance); `score` remains the reciprocal-rank-fusion score. In `rag.query`,
`debug_scores` is the same fused ranking score for each returned chunk.

Both tools accept `filters.modified_since` as an ISO date or datetime. It
matches the last indexed note modification time (`mtime`) inclusively;
timezone-naive datetimes are interpreted as UTC. Incremental and file sync skip
timestamp-only edits when the content hash is unchanged, so these filters and
`recency_boost` use the previous indexed mtime until a full sync refreshes it.
The filter composes with `date_range`, which continues to match frontmatter
due/deadline/start/created/date fields. An optional `recency_boost`
from `0.0` to `1.0` adds an age-decay contribution to RRF scores. A full boost
is capped at one rank-1 RRF contribution and decays with a 30-day half-life;
it reranks the candidates returned by semantic and keyword retrieval. The
default `0.0` preserves ranking.

For example, a `rag.search` call can include:

```json
{"query": "async rust patterns", "top_k": 5, "min_score": 0.65}
```

`rag.search`, `rag.query`, and `rag.related` also accept `verbose` (default
`true`) to preserve the existing full chunk responses. Set `verbose` to
`false` to reduce retrieval payloads: search hits and query chunks become flat objects
with `chunk_id`, fused `score`, `semantic_score`, `keyword_score`, `text`,
`path`, `note_title`, and `heading_path`. Query citations and `debug_scores`
remain available; `rag.query` does not generate answer text. Compact related
results omit the `signals` breakdown and retain each neighbor's path, title,
composite score, and evidence.

```json
{"query": "async rust patterns", "top_k": 5, "verbose": false}
```

### Claude Desktop

Add the following to your `claude_desktop_config.json` (typically `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
	"mcpServers": {
		"second-brain": {
			"command": "second-brain-mcp",
			"args": ["--config", "/absolute/path/to/your/vault/rag_config.toml"]
		}
	}
}
```

> If you installed with `uv`, use the full path to the venv binary:
> `"/path/to/second-brain/.venv/bin/second-brain-mcp"`

### VS Code (GitHub Copilot)

Add to your `.vscode/mcp.json` or user-level MCP settings:

```json
{
	"servers": {
		"second-brain": {
			"type": "stdio",
			"command": "second-brain-mcp",
			"args": ["--config", "/absolute/path/to/your/vault/rag_config.toml"]
		}
	}
}
```


## Configuration Reference

`second-brain init` generates a `rag_config.toml` with these defaults:

```toml
# Path to your Obsidian vault (or any markdown directory)
vault_path = "$CWD"

# Local storage — relative paths resolve from the config file's directory
qdrant_path     = "$CWD/data/qdrant"
fts_path        = "$CWD/data/fts.sqlite"
sync_state_path = "$CWD/data/sync_state.sqlite"

# Qdrant collection name
collection_name = "second_brain_chunks"

# Ollama settings
ollama_url      = "http://127.0.0.1:11434"
embedding_model = "nomic-embed-text"

# Chunking
chunk_size    = 500
chunk_overlap = 80

# Auto-watch vault for file changes (used by long-running processes)
watch_enabled = true

# Glob patterns excluded from indexing by default
exclude_globs = [".obsidian/**", ".git/**", "Templates/**", "_types/**", "_templates/**", "**/_templates*/**", "CLAUDE.md", "AGENTS.md", "GEMINI.md"]

# Optional frontmatter statuses omitted from search results (off by default)
exclude_status = []

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

The default exclusions cover Obsidian/Git metadata, common template and type
folders, and agent instruction files at the vault root or in nested folders.
To include any of those files, edit `exclude_globs` and remove the matching
pattern; an explicit `exclude_globs = []` includes all paths allowed by the
scanner. Existing config files without an `exclude_globs` key receive the
current defaults, while configured lists are preserved as written.

Set `exclude_status = ["superseded", "archived"]` to omit notes whose
frontmatter `status` matches those values from search results. This is disabled
by default. A per-query `filters.exclude_status` list replaces the configured
list; use `filters.exclude_status = []` to include all statuses for that query.

`$CWD` resolves to the working directory at the time `init` is run. Standard `~` and environment variable expansions are supported in all path fields.

## Project Structure

```
second_brain/
├── cli.py           # CLI entrypoint (second-brain)
├── mcp_server.py    # MCP server entrypoint (second-brain-mcp)
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

## Integration tests & benchmarks

`tests/` above is fast and network-free (stubbed embedder, in-memory vector
store). For real-backend tests against a live Ollama plus the actual MCP
stdio protocol, and benchmarks (embedding latency, sync throughput, query
latency), see [`integration/README.md`](integration/README.md). One-command
run via Docker Compose:

```bash
docker compose -f integration/docker-compose.yml up --build \
  --abort-on-container-exit --exit-code-from test-runner
```

## Further Reading

- [Beginner's guide to RAG](docs/rag-beginners-guide.md) — how the pipeline works from first principles
- [End-to-end query trace](docs/query-trace-end-to-end.md) — follow a query through every layer
