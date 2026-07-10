"""MCP server exposing RAG tools to AI clients."""

from __future__ import annotations

import argparse

from obsidian_rag.config import load_config
from obsidian_rag.service import RagService


def build_server(config_path: str):
    """Build and return FastMCP server with all RAG tools registered."""

    from mcp.server.fastmcp import FastMCP

    config = load_config(config_path)
    service = RagService(config)
    mcp = FastMCP("obsidian-rag")

    @mcp.tool(name="rag.query")
    def rag_query(query: str, filters: dict | None = None, top_k: int = 8) -> dict:
        """Retrieve context for a query.

        `answer_draft` is a naive extractive snippet built from the top
        retrieved chunks' text — it is NOT a synthesized answer to the
        query. Callers must not relay it to a user as a complete answer;
        use `chunks` and `citations` to ground any actual synthesis.

        `filters` supports:
        - `tags`: a tag string or list of tags a chunk must have (case-insensitive,
          matches both inline `#tags` and frontmatter `tags:`).
        - `path_prefix`: a vault-relative path prefix string.
        - `date_range`: `{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}` matched
          against the note's due/deadline/start/created date, whichever is set.
        - `frontmatter_contains`: a dict of exact frontmatter key/value pairs.
        - Any other key is matched by exact equality against top-level chunk
          metadata (e.g. `status`, `project`, `context`, `note_title`) or the
          note's derived fields.
        """
        return service.query(query=query, filters=filters, top_k=top_k)

    @mcp.tool(name="rag.search")
    def rag_search(query: str, filters: dict | None = None, top_k: int = 10) -> dict:
        """Return raw hybrid retrieval hits for a query, with no answer draft.

        Use this when you need the ranked chunks/citations themselves (e.g.
        to synthesize your own answer or inspect retrieval quality). Unlike
        `rag.query`, the response has no `answer_draft` field. Prefer
        `rag.query` when you just want a quick extractive snippet plus
        citations in one call.

        `filters` supports the same keys as `rag.query` — see that tool's
        description for the supported filter shapes.
        """
        return service.search(query=query, filters=filters, top_k=top_k)

    @mcp.tool(name="rag.note_context")
    def rag_note_context(note_path: str) -> dict:
        """Return chunk metadata and the link graph for one note, not its text.

        Given a vault-relative note path, returns the note's chunk IDs and
        backlink/outlink relationships to other notes. It does NOT return
        the note's rendered content — to read the note's actual text, fetch
        the vault file directly or use `rag.search`/`rag.query` with a
        query that targets this note.
        """
        return service.note_context(note_path=note_path)

    @mcp.tool(name="rag.sync")
    def rag_sync(mode: str = "incremental", file_path: str | None = None) -> dict:
        """Re-index vault files into the vector and keyword stores.

        `mode="incremental"` (default) only re-indexes files changed since
        the last sync; `mode="full"` rebuilds all indexes from scratch.
        Pass `file_path` to sync a single file. Call this after vault
        content changes and before relying on `rag.search`/`rag.query` to
        reflect those changes.
        """
        return service.sync(mode=mode, file_path=file_path)

    @mcp.tool(name="rag.status")
    def rag_status() -> dict:
        """Return index and model runtime status for operational visibility.

        Use this to check what indexes exist, how many chunks/notes are
        indexed, and which embedding model is configured — useful before
        deciding whether a `rag.sync` is needed.
        """
        return service.status()

    @mcp.tool(name="rag.health")
    def rag_health() -> dict:
        """Return liveness checks for the vector store, embedder, and keyword DB.

        Use this to diagnose whether the underlying dependencies (Qdrant,
        the embedding model, the SQLite keyword store) are reachable and
        working, as opposed to `rag.status` which reports index contents.
        """
        return service.health()

    return mcp


def run() -> None:
    """Executable entrypoint for running the MCP server process."""

    parser = argparse.ArgumentParser(description="Obsidian RAG MCP server")
    parser.add_argument("--config", default="rag_config.toml")
    args = parser.parse_args()
    server = build_server(args.config)
    server.run()


if __name__ == "__main__":
    run()
