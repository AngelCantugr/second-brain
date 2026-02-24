# Obsidian Vault MCP Server (Semantic Search) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a local semantic-search MCP server over the Obsidian vault so Claude Code can query notes through MCP tools.

**Architecture:** A Python ingestion pipeline reads markdown notes from `Obsidian/`, extracts metadata/chunks, embeds chunks with local Ollama (`nomic-embed-text`), and persists vectors in local ChromaDB. A Python MCP server over stdio exposes search and retrieval tools backed by this index. The design optimizes local-first use, incremental indexing, and reproducibility.

**Tech Stack:** Python 3.11+, `uv`, `chromadb`, `mcp`, `ollama`, `python-frontmatter`, `pytest`

---

## Assumptions and Constraints

- Current workspace does not yet contain `agents/` or prior helper files; this plan creates needed structure from scratch and adds optional integration points for future reuse.
- Obsidian vault root is expected at `<repo-root>/Obsidian/` (override via env var later if needed).
- Index persistence path: `~/.vault-index/`.
- Embedding model: `nomic-embed-text` served by local Ollama at default host.
- Required process skills during execution:
  - `@test-driven-development` before each implementation task.
  - `@systematic-debugging` when any test/command fails unexpectedly.
  - `@verification-before-completion` before claiming done.

## Proposed File Layout

- Create: `agents/codex/vault-mcp/pyproject.toml`
- Create: `agents/codex/vault-mcp/README.md`
- Create: `agents/codex/vault-mcp/design.md`
- Create: `agents/codex/vault-mcp/src/vault_mcp/__init__.py`
- Create: `agents/codex/vault-mcp/src/vault_mcp/config.py`
- Create: `agents/codex/vault-mcp/src/vault_mcp/chunker.py`
- Create: `agents/codex/vault-mcp/src/vault_mcp/metadata.py`
- Create: `agents/codex/vault-mcp/src/vault_mcp/index_store.py`
- Create: `agents/codex/vault-mcp/src/vault_mcp/ingest.py`
- Create: `agents/codex/vault-mcp/src/vault_mcp/server.py`
- Create: `agents/codex/vault-mcp/src/vault_mcp/topic_tree.py`
- Create: `agents/codex/vault-mcp/.mcp.json.example`
- Create: `agents/codex/vault-mcp/tests/test_chunker.py`
- Create: `agents/codex/vault-mcp/tests/test_metadata.py`
- Create: `agents/codex/vault-mcp/tests/test_ingest_incremental.py`
- Create: `agents/codex/vault-mcp/tests/test_search_tool.py`
- Create: `Personal/Blog/2026-02-22-building-semantic-search-engine-codex-cli.md`
- Create: `Personal/Blog/2026-02-22-claude-code-obsidian-vault-mcp.md`

## Task 1: Create and Commit Design Doc First

**Files:**
- Create: `agents/codex/vault-mcp/design.md`

**Step 1: Write the design doc (no code yet)**

Include:
- Scope, non-goals, dependencies, and local setup prerequisites.
- Data model for chunks/metadata (`id`, `file_path`, `heading`, `tags`, `date_modified`, `chunk_index`).
- Incremental indexing strategy (mtime checkpoint file + per-file hash fallback).
- MCP tool contracts for `search_vault`, `get_note`, `list_topics`, `recent_activity`.

**Step 2: Commit design before implementation**

Run:
```bash
cd agents/codex/vault-mcp
git add design.md
git commit -m "docs: add design for vault MCP semantic search"
```
Expected: commit succeeds with only design artifact.

## Task 2: Bootstrap Python Project with uv

**Files:**
- Create: `agents/codex/vault-mcp/pyproject.toml`
- Create: `agents/codex/vault-mcp/README.md`
- Create: `agents/codex/vault-mcp/src/vault_mcp/__init__.py`
- Create: `agents/codex/vault-mcp/src/vault_mcp/config.py`

**Step 1: Write failing smoke test for imports/config**

Add `tests/test_metadata.py` basic import/config assertions, e.g. `from vault_mcp.config import Settings`.

**Step 2: Run test to verify it fails**

Run:
```bash
cd agents/codex/vault-mcp
uv run pytest tests/test_metadata.py -v
```
Expected: `ModuleNotFoundError` / missing package failure.

**Step 3: Implement minimal project scaffolding**

- `pyproject.toml` with dependencies:
  - runtime: `chromadb`, `mcp`, `ollama`, `python-frontmatter`
  - dev: `pytest`, `pytest-mock`
- `config.py` dataclass/pydantic-style settings:
  - `vault_root` default `../../../../Obsidian` (resolved absolute)
  - `index_dir` default `~/.vault-index`
  - `collection_name` default `vault_chunks`
  - `embedding_model` default `nomic-embed-text`

**Step 4: Run test to verify it passes**

Run:
```bash
uv run pytest tests/test_metadata.py -v
```
Expected: PASS.

**Step 5: Commit**

```bash
git add pyproject.toml README.md src/vault_mcp/__init__.py src/vault_mcp/config.py tests/test_metadata.py
git commit -m "build: bootstrap vault-mcp package with uv"
```

## Task 3: Build Markdown Metadata + Chunking Core

**Files:**
- Create: `agents/codex/vault-mcp/src/vault_mcp/metadata.py`
- Create: `agents/codex/vault-mcp/src/vault_mcp/chunker.py`
- Create: `agents/codex/vault-mcp/tests/test_chunker.py`
- Modify: `agents/codex/vault-mcp/tests/test_metadata.py`

**Step 1: Write failing tests for parsing and chunking**

Tests should cover:
- YAML frontmatter tags extraction (`tags`, aliases ignored unless needed).
- Folder exclusions: `_templates/`, `Archive/`.
- Chunking around ~500-token target (approx by word count proxy), preserving heading in each chunk metadata.

**Step 2: Run tests and confirm failures**

Run:
```bash
uv run pytest tests/test_chunker.py tests/test_metadata.py -v
```
Expected: FAIL for unimplemented parsing/chunking functions.

**Step 3: Implement minimal parser/chunker**

- `metadata.py`:
  - recursive walker for `*.md`
  - exclusion filter by relative path segments
  - `load_note(path) -> {content, tags, date_modified, headings}`
- `chunker.py`:
  - split markdown by heading blocks
  - emit chunk objects with `chunk_text`, `heading`, `chunk_index`

**Step 4: Re-run tests**

Run:
```bash
uv run pytest tests/test_chunker.py tests/test_metadata.py -v
```
Expected: PASS.

**Step 5: Commit**

```bash
git add src/vault_mcp/metadata.py src/vault_mcp/chunker.py tests/test_chunker.py tests/test_metadata.py
git commit -m "feat: add vault metadata parsing and markdown chunking"
```

## Task 4: Implement Vector Store + Incremental Ingestion

**Files:**
- Create: `agents/codex/vault-mcp/src/vault_mcp/index_store.py`
- Create: `agents/codex/vault-mcp/src/vault_mcp/ingest.py`
- Create: `agents/codex/vault-mcp/tests/test_ingest_incremental.py`

**Step 1: Write failing tests for incremental behavior**

Test cases:
- First run indexes all markdown files.
- Second run with unchanged files performs zero new embeddings.
- Touch one file and rerun indexes only that file's chunks.

Use mocks for Ollama embeddings and Chroma client.

**Step 2: Run tests to verify failures**

Run:
```bash
uv run pytest tests/test_ingest_incremental.py -v
```
Expected: FAIL due to missing ingestion/index APIs.

**Step 3: Implement minimal ingestion pipeline**

- `index_store.py`:
  - Chroma collection init (persistent client at `~/.vault-index/`)
  - upsert/query/delete-by-file helpers
- `ingest.py`:
  - walk vault notes
  - detect changed files via checkpoint file (`~/.vault-index/checkpoint.json`)
  - chunk changed files
  - call Ollama embeddings (`nomic-embed-text`)
  - upsert chunks + metadata (`file_path`, `heading`, `tags`, `date_modified`)

**Step 4: Re-run tests**

Run:
```bash
uv run pytest tests/test_ingest_incremental.py -v
```
Expected: PASS.

**Step 5: Commit**

```bash
git add src/vault_mcp/index_store.py src/vault_mcp/ingest.py tests/test_ingest_incremental.py
git commit -m "feat: add chroma-backed incremental vault ingestion"
```

## Task 5: Implement MCP Server Tools (stdio)

**Files:**
- Create: `agents/codex/vault-mcp/src/vault_mcp/server.py`
- Create: `agents/codex/vault-mcp/src/vault_mcp/topic_tree.py`
- Create: `agents/codex/vault-mcp/tests/test_search_tool.py`
- Create: `agents/codex/vault-mcp/.mcp.json.example`

**Step 1: Write failing tests for tool contracts**

Cover:
- `search_vault(query, top_k)` returns list items with keys: `note`, `excerpt`, `score`, `path`.
- `get_note(path)` returns full markdown string.
- `list_topics()` returns folder/topic counts.
- `recent_activity(days)` returns notes modified within window.

**Step 2: Run tests to verify failures**

Run:
```bash
uv run pytest tests/test_search_tool.py -v
```
Expected: FAIL for missing server/tool handlers.

**Step 3: Implement MCP tool layer**

- Initialize MCP app using Python SDK over stdio transport.
- Wire search tool to Chroma semantic query.
- Wire note reading and folder aggregation helpers.
- Add validation/error surfaces (missing path, empty query, invalid top_k).

**Step 4: Re-run tests**

Run:
```bash
uv run pytest tests/test_search_tool.py -v
```
Expected: PASS.

**Step 5: Add local Claude config example**

`.mcp.json.example` entry should point to:
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

**Step 6: Commit**

```bash
git add src/vault_mcp/server.py src/vault_mcp/topic_tree.py tests/test_search_tool.py .mcp.json.example
git commit -m "feat: expose vault semantic tools via MCP stdio server"
```

## Task 6: End-to-End Verification

**Files:**
- Modify: `agents/codex/vault-mcp/README.md`

**Step 1: Environment verification**

Run:
```bash
ollama pull nomic-embed-text
ollama list | rg nomic-embed-text
```
Expected: model present locally.

**Step 2: Run ingestion end-to-end**

Run:
```bash
cd agents/codex/vault-mcp
uv run python -m vault_mcp.ingest
```
Expected: creates index at `~/.vault-index/` and logs indexed/unchanged counts.

**Step 3: Run server smoke test**

Run:
```bash
uv run python -m vault_mcp.server
```
Expected: process starts cleanly on stdio without import/runtime errors.

**Step 4: Incremental verification**

Run:
```bash
touch ../../Obsidian/<some-note>.md
uv run python -m vault_mcp.ingest
```
Expected: only touched file is re-embedded.

**Step 5: Document usage**

Add README sections: setup, ingest, run server, Claude `.mcp.json`, troubleshooting.

**Step 6: Final verification gate (`@verification-before-completion`)**

Run:
```bash
uv run pytest -v
uv run python -m vault_mcp.ingest
```
Expected: full tests PASS and ingest executes successfully.

**Step 7: Commit**

```bash
git add README.md
git commit -m "docs: add vault-mcp setup and verification guide"
```

## Task 7: Blog Drafts in Parallel with Build

**Files:**
- Create: `Personal/Blog/2026-02-22-building-semantic-search-engine-codex-cli.md`
- Create: `Personal/Blog/2026-02-22-claude-code-obsidian-vault-mcp.md`

**Step 1: Draft Post 1 (process narrative)**

Title: `Building a semantic search engine for my second brain with Codex CLI`

Outline:
- Why this project (ultralearning roadmap + daily utility).
- Design decisions (Python, Ollama local embeddings, Chroma, stdio MCP).
- Build workflow with Codex CLI and TDD.
- Problems encountered and fixes.

**Step 2: Draft Post 2 (technical deep dive)**

Title: `How I made Claude Code aware of my Obsidian vault via MCP`

Outline:
- MCP architecture and tool contracts.
- Ingestion/indexing internals.
- Incremental indexing details.
- Example Claude prompts and real outputs.

**Step 3: Commit drafts**

```bash
git add Personal/Blog/2026-02-22-building-semantic-search-engine-codex-cli.md Personal/Blog/2026-02-22-claude-code-obsidian-vault-mcp.md
git commit -m "docs: add blog drafts for vault MCP build"
```

## Risks and Mitigations

- Ollama latency for large vaults: mitigate with batching, progress logs, and checkpointing.
- Chunk quality drift: mitigate with conservative heading-aware chunking and deterministic IDs.
- Chroma schema drift: centralize metadata keys/constants in `index_store.py`.
- Vault path portability: environment variable override in `config.py`.

## Definition of Done

- `uv run pytest -v` passes in `agents/codex/vault-mcp/`.
- Ingest command creates/updates `~/.vault-index/`.
- MCP server starts on stdio and all 4 tools return contract-compliant output.
- Claude Code can query vault notes via `.mcp.json` integration.
- Two blog drafts exist under `Personal/Blog/`.

