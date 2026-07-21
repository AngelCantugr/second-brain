# Integration tests & benchmarks

Real-backend tests for `second-brain` — no stubbed embedder, no in-memory
vector store. They exercise a live [Ollama](https://ollama.com) server
running the open-source `nomic-embed-text` embedding model, the embedded
(but real) Qdrant vector store, and — in `test_mcp_stdio.py` — the actual
`second-brain-mcp` process talking real stdio JSON-RPC.

These are kept separate from `tests/` (which stays fast, network-free, and
is what `uv run pytest` runs by default): `integration/` is not on
`testpaths` in `pyproject.toml`, so it only runs when explicitly targeted.

## Run everything in one command

```bash
docker compose -f integration/docker-compose.yml up --build \
  --abort-on-container-exit --exit-code-from test-runner
```

This starts a vanilla `ollama/ollama` container, waits for it to become
healthy, brings up the **test-runner** container (the "Test container" —
its only job is running these tests and producing benchmarks), which:

1. waits for Ollama, pulls `nomic-embed-text` (~274MB, cached in a named
   volume across runs), and confirms it can serve an embedding request,
2. runs `pytest integration/ -v --benchmark-json=/artifacts/benchmarks.json`,
3. renders `/artifacts/benchmarks.md` from that JSON.

Artifacts land in `integration/artifacts/` (mounted from the container):

- `benchmarks.json` — raw pytest-benchmark stats
- `benchmarks.md` — human-readable summary table

Tear down (and drop the cached model volume) with:

```bash
docker compose -f integration/docker-compose.yml down -v
```

## Run locally without Docker

If you already have Ollama running locally with `nomic-embed-text` pulled
(`ollama pull nomic-embed-text`):

```bash
uv sync --dev
uv run pytest integration/ -v --benchmark-json=integration/artifacts/benchmarks.json
uv run python integration/bench_report.py integration/artifacts/benchmarks.json integration/artifacts/benchmarks.md
```

Set `OLLAMA_URL` / `EMBEDDING_MODEL` env vars to point at a non-default
Ollama instance or model.

## What's covered

| File | Covers |
|---|---|
| `test_mcp_stdio.py` | All 9 MCP tools over the real stdio transport (the MCP protocol layer has no other test coverage in this repo) |
| `test_rag_integration.py` | `RagService` against real Ollama + real Qdrant, independent of the MCP transport |
| `test_benchmarks.py` | Embedding latency, full-sync throughput, and query/search latency |
