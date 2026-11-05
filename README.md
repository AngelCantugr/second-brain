# Second Brain MCP

A local-first Retrieval-Augmented Generation (RAG) pipeline for [Obsidian](https://obsidian.md) vaults. Index your notes once, then query them semantically from the command line or through any MCP-compatible AI client (Claude Desktop, Cursor, VS Code Copilot, etc.).

## Features

- **Hybrid retrieval** — combines semantic vector search (Qdrant) with full-text keyword search (SQLite FTS5, OR-matched with stopwords dropped and note titles and headings weighted above body text) via Reciprocal Rank Fusion
- **Note-association graph** — a precomputed graph blending semantic similarity, wikilinks, shared tags, and co-mentions, so you can ask "what's related to this note, and why?" instead of only "what matches this text?"
- **Markdown and canvas sources** — Obsidian `.canvas` boards can be indexed alongside notes (opt in with `include_extensions`): text cards become searchable chunks, and file cards and arrows between them become graph links. Every chunk records its source `kind`, so search can be limited to one kind
- **Code repositories (opt in)** — list repos in `code_sources` to index their README, `docs/`, ADRs and CHANGELOG next to your notes, and optionally their Python, Go and TypeScript functions, classes and types as symbol outlines (signature, docstring, leading comment), so a search for a design decision finds the repo doc and the code it explains as well as your notes about it
- **Incremental indexing** — only re-indexes files that changed since the last sync, and only recomputes the graph edges affected by that change
- **MCP tool interface** — expose your vault as tools that any MCP client can call
- **Local-first** — all embeddings and storage run on your machine; nothing leaves it
- **Configurable chunking** — tune chunk size, overlap, and glob exclusions per vault
- **Privacy controls** — `redact_patterns` replaces regex matches with `[REDACTED]` before note text reaches the embedder or any index

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

## Upgrading

This release changes how notes are chunked and embedded:

- **Chunking** — code fences and inline code no longer produce headings, tags, links or tasks, and chunk text keeps its original newlines and indentation.
- **Embeddings** — chunks are embedded with a note-title and heading header (`embed_context_header`), and sections named in `skip_headings` are left out of the index.
- **FTS schema** — the keyword table gains title and heading columns. This is migrated automatically on the next start; no action needed.

Incremental sync skips files whose content hash is unchanged, so existing notes keep their old chunks and vectors. Run one full sync after upgrading to re-chunk and re-embed everything:

```bash
second-brain sync --mode full
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

Code fences and inline code no longer produce headings, tags, links, or tasks, and chunk
text now keeps its original newlines and indentation. Incremental sync skips files whose
content hash is unchanged, so it won't re-chunk them. After upgrading, run
`second-brain sync --mode full` once.

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

### Measuring retrieval quality

`second-brain eval` scores search against a golden query set: each case lists a query and the note paths a good answer must surface. It reports note-level recall@5, recall@10 and MRR, so you can tell whether a parsing or ranking change helped.

```bash
second-brain --config ~/vaults/work/rag_config.toml eval --golden ~/vaults/work/golden.toml --depth 10
```

Eval searches with `top_k = 50` internally, a larger candidate pool than default search, and collapses those chunks to distinct notes. `--depth` beyond the number of distinct notes found in 50 chunks is effectively capped. Expected paths that are not indexed notes (a typo or an excluded file) are listed per case under `unknown_expected` and counted in the summary, instead of silently scoring as misses.

See `eval/golden.example.toml` for the format. Its paths are illustrative. Keep your real golden set beside your vault config, since it names your personal notes.

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
| `rag.related` | Notes associated with one note, ranked with a per-signal score breakdown and an edge `type`; `edge_types` filters by type |
| `rag.connections` | How closely two notes are associated, direct or via the strongest path between them |
| `rag.drift` | Note/code pairs that may have drifted apart (code changed after the note, or a mention stopped resolving), with both texts for the caller to judge; designs and plans are summarized per document by how far their code has moved |
| `rag.map` | Vault-wide summary of note clusters, orphans, bridge notes, and ghost (unwritten) notes |
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

Wikilinks resolve the way Obsidian resolves them. `[[Note]]` matches a file
name, then falls back to a note listing it under `aliases` (or legacy `alias`)
frontmatter. `[[Folder/Note]]` matches only notes whose path ends with that path,
and a relative `[[../Folder/Note]]` is first joined to the linking note's folder.
`#Heading`, `#^block`, and `^block` suffixes and a trailing `.md` are ignored.
When several notes share a name, the link resolves to one of them: the note in
the linking note's folder, else the shallowest path, else the first
alphabetically. Links that match no note appear in `rag.map` as `ghosts`, the
top 25 ranked by how many notes link to them, with up to five linking notes
each. They are concepts you keep referencing but have not written, or notes
that are excluded. Section links, self-links, and attachment embeds such as
`![[diagram.png]]` are never ghosts.

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
| `untracked_files` | Eligible files on disk without a successful checkpoint, including paths with only a pending replacement. |
| `missing_files` | Eligible checkpointed or pending-replacement paths that no longer exist on disk. |
| `graph_nodes` / `graph_edges` | Current graph node and edge counts. |
| `graph_last_built` | Timestamp recorded for the last graph build, or `null` if no build is recorded. |

Eligibility follows the scanner: a file's extension must be in
`include_extensions` (default `.md` only), paths containing a hidden segment are skipped, and configured `exclude_globs` are applied to disk
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

An optional `graph_boost` from `0.0` to `1.0` lets the note graph inform
ranking. After fusion, it takes the five best-ranked notes as seeds. Every
retrieved chunk from a graph neighbor of a seed gains up to `graph_boost` times
one rank-1 RRF contribution, scaled by the strongest edge composite to any
seed. Seeds do not lift each other. Like `recency_boost`, it only reorders
retrieved candidates, and a chunk found by only one retrieval leg cannot pass a
chunk that both legs ranked first. The default `0.0` leaves ranking unchanged, and `graph_enabled =
false` disables it. The CLI `search`, `query`, and `eval` commands accept
`--graph-boost`, so you can measure it against your golden set before using it.

`filters.kind` limits results to one source kind (`"markdown"`, `"canvas"`,
`"repo_doc"` or `"code"`) or a list of kinds. Chunks indexed before kinds existed count as
`"markdown"`, so the filter works without a re-sync. `filters.repo` (a name or a
list of names) keeps only chunks from those [code sources](#indexing-code-repositories).
`filters.doc_role` (`"reference"`, `"record"`, `"history"`, or a list) keeps
notes and docs of those [roles](#doc-roles-reference-record-history), such as
designs and plans for a "why" question.

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

# Sections whose heading matches (case- and whitespace-insensitive) are not indexed
skip_headings = []

# Prefix each chunk's embedding input with "Note Title > Heading"
embed_context_header = true

# Chunks sent per Ollama /api/embed request (integer >= 1)
embed_batch_size = 32

# Auto-watch vault for file changes (used by long-running processes)
watch_enabled = true

# Glob patterns excluded from indexing by default
exclude_globs = [".obsidian/**", ".git/**", "Templates/**", "_types/**", "_templates/**", "**/_templates*/**", "CLAUDE.md", "AGENTS.md", "GEMINI.md"]

# File types to index; each needs a loader (supported: .md, .canvas)
include_extensions = [".md"]

# Optional frontmatter statuses omitted from search results (off by default)
exclude_status = []

# Max chunks returned per query
max_context_chunks = 8

# Regex patterns replaced with [REDACTED] before embedding and indexing
redact_patterns = []

# Note-association graph (used by rag.related / rag.connections / rag.map)
graph_enabled = true             # set false to skip graph computation entirely
graph_knn_k = 8                  # max semantic neighbors considered per note
graph_semantic_min = 0.35        # minimum (mean-centered) cosine to become a semantic candidate
graph_semantic_center = true     # subtract the vault-mean centroid before cosine (see below)
graph_min_edge_score = 0.15      # minimum composite score to persist an edge
                                  # (edges with a wikilink or co-mention persist regardless)
graph_weight_semantic = 0.5      # composite score weights (should sum to 1.0)
graph_weight_link = 0.25
graph_weight_tag = 0.15
graph_weight_comention = 0.10
graph_comention_cap = 3          # co-mentions beyond this count don't add further score
graph_comention_max_fanout = 20  # notes linking to more targets than this don't contribute
                                  # co-mention pairs (keeps hub/MOC notes from exploding edge count)

# Search-score multiplier for code-source hits on queries that name nothing
# distinctive in a code source; 0 < value <= 1, and 1.0 leaves ranking
# unchanged (see "Ranking code sources against notes")
code_source_weight = 1.0

# Doc roles for vault notes, replacing the defaults of each role named
# (reference, record, history; see "Doc roles")
# doc_roles = { record = ["docs/specs/**"] }

# Code repositories indexed alongside the vault (none by default; see
# "Indexing code repositories"). A [[table]] must come after every top-level
# setting: TOML puts the lines below it inside the table.
# [[code_sources]]
# name = "second-brain"
# path = "~/src/second-brain"
```

A note's centroid is the mean of its chunk embeddings, and averaging pulls every
centroid toward the vault-wide mean. Raw centroid cosines therefore bunch up: in
one 648-note vault every pair scored above 0.5, with a median of 0.71, so a
fixed `graph_semantic_min` filtered nothing. With `graph_semantic_center = true`
the graph subtracts the vault-mean centroid before taking cosine. That spreads
the median pair to about 0 and puts `0.35` near the 92nd percentile of pairs,
so semantic edges mark notes that are actually alike. The mean is stored with
the graph. Incremental syncs score against it, and a full sync recomputes it.
Vaults with fewer than 10 embedded notes always use raw cosine, because their
mean is dominated by the notes themselves. If you set
`graph_semantic_center = false`, raise `graph_semantic_min` to about 0.9 to
keep the same selectivity.

`include_extensions` picks which file types are indexed. It defaults to `[".md"]`,
so existing vaults index exactly what they did before. Add `".canvas"` to index
Obsidian canvases ([JSON Canvas 1.0](https://jsoncanvas.org/spec/1.0/)):

- Each text card becomes a chunk. Its heading is the label of the group that
  contains it (`Outer > Inner` for nested groups), or `root` outside any group.
- Each file card is a link from the canvas to that exact file, so the canvas joins the
  note graph like a note with wikilinks. Card paths are vault-relative, so they
  resolve exactly, not to a same-named note nearer the canvas.
- An arrow between two file cards is a link between those two notes, in the
  arrow's direction (both ways for a two-headed arrow, from→to for a plain line).
- A canvas is titled with its file name, `Board.canvas`. That matches Obsidian, where
  `[[Board]]` means `Board.md` and only `[[Board.canvas]]` means the canvas.
- Link cards (URLs) and arrow labels are not indexed yet. A canvas that is not
  valid JSON is reported in the sync `errors` and retried on the next sync.

Removing an extension from the list removes its files from the index on the
next sync. An extension with no loader fails at config load.

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

Daily-journal templates often end with navigation blocks such as `## 🔗 Related Notes`
that are only link lists and drown out real content in semantic search. List their
headings in `skip_headings` to keep those sections out of the index:

```toml
skip_headings = ["🔗 Related Notes", "Navigation"]
```

Matching is case-insensitive and ignores surrounding whitespace; include any emoji
exactly as written in the heading. The match is against the heading text, and only the
section directly under a matching heading is skipped: subsections beneath it (for
example an `###` under a skipped `##`) are still indexed. Links and tags inside a
skipped section still count toward the note's graph, because they come from the whole
body.

With `embed_context_header = true` (the default), the text sent to the embedder is
prefixed with the note title and heading, such as `Project > Plan`. This anchors short
sections to their note's topic. Stored text, keyword search, and returned snippets do
not include the prefix. Changing either key requires `second-brain sync --mode full`
so existing chunks are re-embedded.

Embeddings go to Ollama's batch `/api/embed` endpoint (the successor to
`/api/embeddings`), `embed_batch_size` chunks per request. Vectors come back in input order. Full and
incremental syncs pack whole notes into one request up to `embed_batch_size` chunks, so
short notes share a request; a note longer than that gets requests of its own. Lower it
if a big batch times out (each request has a 120-second timeout). Chunks are sized in
words but the model's limit is in tokens, so code-dense text can overflow it (500 words
of fenced Python can pass nomic-embed-text's 2,048 tokens). A request Ollama rejects
for exceeding the context length is sent once more with truncation and logged as a
warning: the over-length chunk's vector covers its start, the other chunks' vectors are
unchanged, and keyword search still indexes the full text. When a request holding several notes fails, each of those
notes is embedded once more on its own, so one note Ollama rejects cannot fail the
others. A note whose embedding fails is reported in `errors` with the batch number and
Ollama's message, keeps its previously indexed chunks, and is retried on the next sync.

`redact_patterns` lists Python regular expressions. Every match is replaced with
`[REDACTED]` before the note is chunked, embedded or stored, so the original text never
reaches Ollama, Qdrant, the SQLite FTS store or the note graph:

```toml
redact_patterns = ['sk-[A-Za-z0-9]{20,}', 'password:\s*\S+']
```

Use TOML literal strings (single quotes) so backslashes reach the regex unchanged. Each
pattern is compiled when the config loads, and an invalid one stops startup with an
error naming it. All patterns are matched against the original text and overlapping
matches are merged into one token, so adding a pattern never weakens another. Redaction
covers the note body (and therefore chunk text, embedding input and snippets), the note
title, and frontmatter values (keys are kept; a number that matches becomes the token).
Headings, tasks, tags and wikilinks are then re-extracted from the redacted text, so a
pattern that includes markdown context such as `#private-\S+` also removes the tag. It does not cover the vault-relative file path, which is
the note's identity key and is stored with every chunk: put notes with sensitive file
names in `exclude_globs` instead. Patterns match within the whole note before it is split
into chunks, so a match that spans a chunk boundary is still redacted. `skip_headings`
is compared against the redacted heading text. Your vault files are never modified.

Incremental sync only re-indexes notes whose content changed, so adding, changing or
removing a pattern does not touch notes that are already indexed. Run
`second-brain sync --mode full` after editing `redact_patterns` to re-index every note
with the new patterns.

`$CWD` resolves to the working directory at the time `init` is run. Standard `~` and environment variable expansions are supported in all path fields.

## Indexing code repositories

The vault is the only source by default. To index a repository's prose as well,
add a `[[code_sources]]` table per repo:

```toml
[[code_sources]]
name = "second-brain"              # letters, digits, ".", "_", "-"; unique
path = "~/src/second-brain"        # ~, $VARS and paths relative to this file work
# Optional; these are the defaults:
# doc_globs = ["README*", "CHANGELOG*", "CONTRIBUTING*", "ARCHITECTURE*",
#              "docs/**", "doc/**", "adr/**", "adrs/**", "decisions/**"]
# exclude_globs = []               # added to the built-in exclusions below
# languages = []                   # "python", "go", "typescript": also index symbols
# code_globs = ["*"]               # which source files of those languages
# doc_roles = { record = [...] }   # replaces that role's default globs (see "Doc roles")
```

Each repo's markdown files that match `doc_globs` are indexed with the same
markdown loader as vault notes. A slash-free glob matches a file name at any
depth, and `docs/**` matches a `docs` folder at any depth. These files:

- are stored under `repo:<name>/<path in repo>` (for example
  `repo:second-brain/docs/capability-review-2026-11.md`), so they never collide
  with vault paths: Obsidian forbids `:` in file and folder names;
- have `kind = "repo_doc"` and `repo = "<name>"` in chunk metadata, so
  `filters={"kind": "repo_doc"}` or `filters={"repo": "second-brain"}` narrows a
  search, and their titles name the repo (`second-brain: README`);
- go through `redact_patterns` like every note.

#### Symbol chunks

With `languages` set, source files of those languages are split with
[tree-sitter](https://tree-sitter.github.io/) into one chunk per function,
class, method, type or interface (Python `.py`/`.pyi`, Go `.go`, TypeScript
`.ts`/`.tsx`/`.mts`/`.cts`, including exported arrow functions), plus a chunk
for a Python module docstring. tree-sitter is an optional extra:

```bash
pipx install 'second-brain-mcp[code] @ git+https://github.com/AngelCantugr/second-brain.git'
# or, in a clone: uv sync --extra code   (the dev group already includes it)
```

A source with `languages` set fails at config load, naming the missing
package, when the extra is not installed.

A symbol chunk holds the symbol's *why*, not its body: the leading comment,
the signature and the docstring, copied from the file as written. Grep, LSP
tools and the file itself answer *what* the code does; a RAG copy of bodies
is weaker and goes stale. A symbol with no comment or docstring gets its first
12 body lines instead, and type and interface declarations keep up to 40 lines.
Each chunk is stored with `kind = "code"`, `heading_path` set to the dotted
symbol path (`GraphBuilder.rebuild_full`), and these metadata keys: `repo`,
`language`, `file` (path in the repo), `symbol`, `symbol_name`,
`parent_symbol`, `symbol_kind` (`function`, `method`, `class`, `type`,
`interface`, `module`), `start_line`, `end_line` and `doc`. The note title is
`<repo>: <file>`, so the embedded text starts with
`second-brain: second_brain/graph.py > GraphBuilder.rebuild_full`.

`eval/golden.code.toml` is a synthetic golden set against this repository:
index it as a code source with `languages = ["python"]` and run
`second-brain eval --golden eval/golden.code.toml`.

File selection honors the repo's `.gitignore` (through `git ls-files`, when the
path is inside a git work tree), and always skips hidden files and folders,
dependency and build folders (`node_modules`, `vendor`, `dist`, `build`,
`target`, `venv`, `__pycache__`, and similar), symlinks that leave the repo,
and files over 1 MB.

Code sources join the graph's node list but not the vault's semantic, link,
tag or co-mention scoring: adding a repo never moves the vault mean, takes a
semantic neighbor slot, or captures a `[[README]]` link from a note.
`rag.map` reports vault notes only. They connect to notes through typed edges
instead.

#### Ranking code sources against notes

A repo can hold as many chunks as the vault, and its docs and symbols share
the vault's vocabulary, so for a question about your notes, repo chunks take
top ranks that a note would otherwise hold. `code_source_weight` (default
`1.0`, off) multiplies the fused score of every code-source hit, but only for
a query that names nothing distinctive in a code source. A query that names
one keeps today's ranking, because it is a question about that repo.

A name is distinctive when it is the repo's name, a directory or file name
(without its extension) or a symbol name, is at least 4 characters, and
appears in at most 2 vault notes. That last rule keeps everyday words that
also name folders (`plugins`, `scripts`, `docs`) from counting. Names with
hyphens or underscores also match when typed with spaces, so `career desk`
names `career-desk`, and vault notes are counted the same way. Private and
dunder symbols (`_extract_python`, `__init__`) count too. The names are
rebuilt after every sync.

On one vault (650 notes) with one repo indexed (about half the index's
chunks), `0.5` raised a 24-case vault golden set's recall@10 from 0.833 to
0.896 and left a 26-case set unchanged. On 10 questions about the repo,
recall@10 stayed at 1.0; MRR fell from 0.864 to 0.789, all of it from the 4
questions that named nothing in the repo. Measure on your own golden sets with
`second-brain eval --golden <file> --code-source-weight 0.5`; `search` and
`query` take the same flag. The weight only reorders retrieved candidates: a
note that repo chunks pushed out of both retrieval legs' candidate lists stays
out.

#### Note↔code edges

Every graph edge has a `type`:

| `type` | Between | When |
|---|---|---|
| `links` | two vault notes | the blended semantic/wikilink/tag/co-mention score (every edge before code sources existed) |
| `mentions_code` | a note and a code-source file | the note names the file, one of its symbols, or its repo |
| `imports` | two source files of one repo | one imports the other (Python modules; relative TypeScript specifiers, including emitted `.js`/`.mjs`/`.cjs` names; Go module packages, including the root package; standard-library imports are skipped) |

A note mentions code through inline code (`` `GraphBuilder` ``,
`` `Indexer.sync` ``, `` `second_brain/graph.py:42` ``, `` `second-brain` ``),
a GitHub URL (`https://github.com/<owner>/<repo>/blob/<ref>/<path>`), or a
slash-separated path with a source or markdown extension written in prose.
Fenced code blocks are ignored: a pasted snippet names everything it uses.
Matching favors precision:

- A path matches a file by exact path in the repo, else by a unique path
  suffix (`graph.py` alone works only if one file has that name). A bare
  markdown file name (`README.md`) or a name every project has (`main.go`,
  `__init__.py`, `index.ts`) needs a folder or the repo name in front of it
  (`second-brain/README.md`).
- A symbol matches only when exactly one file defines it. `Class.method` and
  `module.Class` forms are tried from the most qualified end.
- A mention that matched a file keeps matching it while that file still
  matches, even when another file starts defining the same name or another
  file of that name appears: the note was written about the first one.
- In a vault note, a one-word bare name (`` `Task` ``, `` `base.py` ``,
  `` `util.go:12` ``) links only if the note also names that repo some other
  way: its name, a GitHub URL, a path with a folder, or a name of two or more
  words (`GraphBuilder`, `compute_backoff`, `stack-ledger.py`). One-word
  names are what other projects, and other languages, use too, so a Go note's
  `` `Task` `` struct does not link to a Python class. Repo docs are already
  scoped to their own repo and resolve them as before.
- A bare lowercase word (`sync`, `path`) never matches a symbol; in prose it is
  usually the word. Names with an underscore or a capital letter
  (`chunk_note`, `Redactor`) do.
- A repo name (or a GitHub URL with no file path) links to the repo's root
  README.
- Vault notes can mention any code source; a repo doc only its own repo.

A typed edge's `link` signal is 0.7 one-way (1.0 for files that import each
other) and its `composite` is `graph_weight_link × link`, so it ranks like a
wikilink with no other evidence. `rag.related` neighbors carry `type`, and typed
ones add `evidence.code_refs` (the mentions or import specs that matched, up to
10). Pass `edge_types=["mentions_code"]` to ask which code a note discusses, or
call `rag.related` on a code path (`repo:second-brain/second_brain/graph.py`) to
ask which notes discuss that file. `rag.connections` reports the `type` of the
direct edge and of each path hop. Typed edges are recomputed whole on every
sync, so renaming a function re-targets mentions in notes that did not change.
Notes indexed before this release record their mentions on their next re-index;
run `second-brain sync --mode full` once after adding your first code source.

#### Drift between notes and code

Every mention that resolves to a code-source file or symbol is tracked across
syncs, so `rag.drift` can report when the two sides may no longer agree:

- `stale`: the mentioned code changed and the note has not changed since. A
  mention of a symbol (`` `compute_backoff` ``, `` `GraphBuilder.rebuild_full` ``)
  compares that symbol's whole declaration, body included, so edits elsewhere
  in the file do not count; a path or repo mention compares the whole file.
- `dangling`: a mention that used to resolve no longer does, because the file
  or symbol was removed or renamed. A name that also starts matching other
  files does not dangle while its old file still matches it. The
  edge disappears; the drift row keeps the old target, the date it was lost, and
  `candidates` for where it went (`repo:second-brain/x.py::new_name`).
- `ok`: tracked, no change.

Editing a note marks every mention in it as up to date with the code as it is
at the next sync; a full sync, which reprocesses notes without changing them,
does not. The first sync with this release records a baseline, so nothing is
stale until code changes after it.

The server does not decide whether a stale note is wrong. Each pair carries the
note's chunks that contain the mention, the symbol's indexed outline with its
line span and local file path, the code's change time (its file's mtime when
the change was first synced) and the evidence, and the calling agent judges
them. Filter with `status` (`["stale", "dangling"]` by default, or `["ok"]`),
`note_path`, `repo` and `levels`: whether the mention names a `symbol`, a
`file` or a whole `repo`. Repo mentions resolve to the README, which changes in
nearly every pull request, so they are left out unless `levels` includes
`"repo"`; symbol pairs are listed before file pairs. Drift needs `graph_enabled`; run
`second-brain sync --mode full` once after upgrading so existing notes and code
files record their fingerprints. Design notes and the measurements behind it:
`docs/drift-detection.md`.

#### Doc roles: reference, record, history

Docs relate to code in different ways, so the same change means different
things. Each note and repo doc has a role, decided from its path by globs:

| Role | Default globs | Code changed under it means | In `rag.drift` |
|---|---|---|---|
| `reference` | `README*`, `CONTRIBUTING*`, `ARCHITECTURE*`, `SKILL.md`, and anything no other glob matches | the doc may be out of date | listed pair by pair (the default `roles`) |
| `record` | `design.md`, `*-design.md`, `plan.md`, `plan-*.md`, `*-plan.md`, `specs/**`, `spec/**`, `plans/**`, `adr/**`, `adrs/**`, `decisions/**`, `rfcs/**`, and file or folder names holding a date (`2026-11-04`) | the code moved on from what was decided: expected, but a record whose code mostly changed or disappeared was superseded or built differently | one row per record in `records`, most moved first; pairs with `roles=["record"]` |
| `history` | `CHANGELOG*`, `HISTORY*` | nothing: a changelog is history | not tracked |

Globs are checked in the order reference, history, record, so a README inside
a specs folder stays a reference. Vault notes and every code source start from
these defaults; a `doc_roles` table replaces the defaults of each role it
names and keeps the others. The top-level table matches vault paths, the one
in a `[[code_sources]]` table paths in that repo:

```toml
doc_roles = { record = ["docs/specs/**", "Journal/**"] }   # vault notes

[[code_sources]]
name = "second-brain"
path = "~/src/second-brain"
doc_roles = { reference = ["README*", "docs/guides/**"] }  # this repo's docs
```

Roles are computed from the current config when asked, so changing a glob
needs no re-sync. Each `records` row has `tracked` mentions, how many
`changed` and are `gone`, `moved_share`, `gone_mentions` and the latest
`changed_at`. Records rank by moved share, damped for records that name only a
few things. A record's starting point is the first sync that saw it, so a
spec that was already out of date then shows no movement until the code
changes again. The ranking cannot tell a plan that was carried out (its code
changed because it was built) from one that was abandoned; that takes reading
it next to the code, which `rag.drift` leaves to the calling agent.

Search can use roles too: `filters={"doc_role": "record"}` narrows a "why did
we decide X" question to designs, plans and dated notes, and
`"reference"` a "how does X work" question to living docs. Code chunks match
no role.

`sync` (full and incremental) rescans every code source; the watcher and
`sync --mode file` cover the vault only. Removing a source from the config
removes its files from the index on the next sync. If a source's path is
missing (an unmounted drive), or `git ls-files` fails inside its git repo (a
corrupt index, no `git` binary), or one of its folders or files cannot be read, sync reports it in `errors` and keeps its index
entries; it never falls back to a listing that ignores `.gitignore`.

## Project Structure

```
second_brain/
├── cli.py           # CLI entrypoint (second-brain)
├── mcp_server.py    # MCP server entrypoint (second-brain-mcp)
├── service.py       # High-level facade shared by CLI and MCP
├── indexer.py       # Orchestrates load, chunk, embed, store
├── loaders.py       # Loader registry: picks a parser by file extension
├── code.py          # tree-sitter symbol outlines for Python, Go, TypeScript
├── code_graph.py    # Typed mentions_code / imports edges between notes and code
├── doc_roles.py     # Doc roles (reference, record, history) from path globs
├── drift.py         # Per-mention drift state: stale and dangling note/code pairs
├── sources.py       # Vault + code-source roots, repo:<name>/ paths, .gitignore-aware listing
├── parser.py        # Markdown + frontmatter parser
├── canvas.py        # Obsidian .canvas (JSON Canvas) parser
├── chunker.py       # Text chunking with overlap
├── redaction.py     # redact_patterns applied before indexing
├── embedder.py      # Ollama embedding client
├── vector_store.py  # Qdrant vector store wrapper
├── keyword_store.py # SQLite FTS5 keyword store
├── retrieval.py     # Reciprocal Rank Fusion merge
├── routing.py       # Repo-intent terms that decide when code_source_weight applies
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
