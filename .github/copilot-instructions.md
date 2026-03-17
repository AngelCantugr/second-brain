# Copilot Instructions for second-brain

## Project intent

This repository implements a local-first RAG pipeline for Obsidian-style markdown vaults.
Primary goals:

- keep indexing and retrieval fully local by default
- preserve deterministic, testable behavior for parsing, chunking, indexing, and retrieval
- expose the same core behavior through both the CLI and the MCP server

When making changes, optimize for correctness, predictable JSON responses, and minimal coupling between layers.

## Source of truth

Work from the source package in `obsidian_rag/`.

- Do not edit generated artifacts under `build/`
- Do not edit packaging metadata under `obsidian_rag_mcp.egg-info/`
- Keep README examples aligned with the actual CLI and MCP behavior when commands, config, or outputs change

## Architecture

Preserve the current layering unless a task explicitly requires a refactor:

1. `config.py`
   Loads `rag_config.toml` into `RagConfig` and resolves paths.

2. `cli.py`
   Thin argparse entrypoint. It should parse arguments, call the service layer, and print JSON.

3. `mcp_server.py`
   Thin MCP registration layer. It should expose tools backed by the service layer, not reimplement business logic.

4. `service.py`
   Shared facade for CLI and MCP. Put cross-interface orchestration here.

5. `indexer.py`
   Owns sync orchestration: scan, parse, chunk, embed, upsert, delete missing paths.

6. `parser.py`, `chunker.py`, `scanner.py`, `retrieval.py`
   Pure or mostly pure domain logic. Prefer focused, easily testable functions.

7. `embedder.py`, `vector_store.py`, `keyword_store.py`, `sync_state.py`
   Infrastructure adapters. Keep storage and external-service details here.

If a change affects both CLI and MCP behavior, implement the logic in `service.py` or lower, then wire it into the entrypoints.

## Coding conventions

- Target Python 3.11+
- Use type hints throughout new code
- Follow the existing preference for `pathlib.Path` over string path manipulation
- Prefer small functions with explicit inputs and outputs
- Reuse existing dataclasses and add new shared contracts in `models.py` when a new cross-module data shape is needed
- When adding new dataclasses, follow the existing pattern of `@dataclass(slots=True)` unless there is a clear reason not to
- Keep public return payloads stable and JSON-serializable
- Raise specific exceptions for invalid inputs rather than silently accepting bad state
- Preserve the repository's local-first stance; do not introduce cloud-only defaults or remote dependencies for core flows

## Behavior guardrails

Be careful with these existing semantics:

- `load_config()` supports `$CWD`, environment-variable expansion, `~`, and config-relative paths
- file-scoped sync must stay constrained to files inside `vault_path`
- CLI commands print JSON objects; do not switch to human-only output unless explicitly requested
- MCP tool names are part of the public interface: `rag.query`, `rag.search`, `rag.note_context`, `rag.sync`, `rag.status`, `rag.health`
- retrieval is hybrid and should continue to combine semantic and keyword results
- local privacy matters: redaction, exclusions, and local storage behavior should not be weakened casually

If you change any of the above, update tests and docs in the same task.

## Testing expectations

Every non-trivial change should include or update focused tests in `tests/`.

Preferred workflow:

```bash
uv run pytest -q
```

If the environment is using a virtualenv instead of `uv`, this is also acceptable:

```bash
pytest -q
```

Testing guidance:

- prefer `tmp_path` fixtures for filesystem behavior
- avoid real network calls in tests
- use stubs or in-memory implementations for embeddings and vector storage when possible
- follow the existing test style: direct assertions, minimal fixtures, and explicit edge-case coverage
- add regression tests when fixing bugs in config resolution, indexing modes, retrieval, or persistence behavior

## Change placement guide

Use this routing when deciding where code belongs:

- add a CLI flag or command: `cli.py`, then delegate to `service.py`
- add an MCP tool or parameter: `mcp_server.py`, then delegate to `service.py`
- change sync behavior or indexing lifecycle: `indexer.py`
- change parsing of markdown/frontmatter/links/tasks/headings: `parser.py`
- change chunk generation or chunk metadata: `chunker.py`
- change ranking or fusion logic: `retrieval.py`
- change embedding provider behavior: `embedder.py`
- change vector persistence/search behavior: `vector_store.py`
- change FTS behavior or metadata filtering: `keyword_store.py`
- change config parsing or defaults: `config.py` and likely `cli.py` and README examples

## Documentation expectations

Update `README.md` when you change:

- CLI commands or flags
- MCP startup or tool behavior
- generated config defaults
- installation or developer workflow

Update docs under `docs/` when a change alters how the pipeline works conceptually.

## What to avoid

- do not put business logic directly into the CLI or MCP registration layer
- do not edit `build/` outputs to make tests pass
- do not introduce broad refactors when a targeted fix is enough
- do not add unnecessary dependencies when the standard library or existing modules already cover the need
- do not bypass existing path-safety or privacy-related checks for convenience

## Definition of done

A change is in good shape when:

- code is placed in the correct layer
- tests cover the changed behavior
- CLI and MCP interfaces remain consistent with the shared service behavior
- README or docs are updated when public behavior changes
- no generated artifacts were manually edited
