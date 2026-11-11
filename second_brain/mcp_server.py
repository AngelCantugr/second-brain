"""MCP server exposing RAG tools to AI clients."""

from __future__ import annotations

import argparse
from typing import Annotated

from pydantic import Field

from second_brain.config import load_config
from second_brain.service import RagService
from second_brain.watcher import VaultWatcher


def build_server(config_path: str, start_watcher: bool = False):
    """Build and return an MCPServer with all RAG tools registered.

    With ``start_watcher`` and ``watch_enabled`` in the config, vault changes
    are indexed in the background for as long as the process lives. It is
    off by default so building a server (in tests, say) starts no thread.
    """

    from mcp.server.mcpserver import MCPServer

    config = load_config(config_path)
    service = RagService(config)
    if start_watcher and config.watch_enabled:
        VaultWatcher(service, debounce_seconds=config.watch_debounce_seconds).start()
    mcp = MCPServer("second-brain")

    @mcp.tool(name="rag.query")
    def rag_query(
        query: str,
        filters: dict | None = None,
        top_k: int = 8,
        min_score: Annotated[float, Field(strict=True, ge=-1, le=1, allow_inf_nan=False)] | None = None,
        verbose: bool = True,
        recency_boost: Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)] = 0.0,
        graph_boost: Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)] = 0.0,
    ) -> dict:
        """Retrieve context for a query.

        Returns retrieved `chunks` and their `citations` to ground any
        synthesis performed by the caller. No answer text is generated.

        `min_score` optionally filters by cosine semantic similarity in [-1, 1]
        before the final `top_k` cutoff. `debug_scores` contains fused ranking
        scores, including an optional recency contribution, while each chunk
        exposes its semantic and keyword scores.
        `recency_boost` in [0, 1] adds a bounded 30-day-half-life age-decay
        contribution when reranking retrieved candidates; zero preserves
        default ranking. Recency uses the last indexed mtime snapshot.
        `graph_boost` in [0, 1] lifts retrieved chunks from notes that the
        note graph links to the top five hit notes (but not those five), by at most one rank-1
        fusion term scaled by edge strength; zero (default) disables it.

        `verbose` defaults to true and preserves full chunk metadata. Set it to
        false for flat compact chunks with `chunk_id`, score signals, `text`,
        `path`, `note_title`, and `heading_path`.

        `filters` supports:
        - `tags`: a tag string or list of tags a chunk must have (case-insensitive,
          matches both inline `#tags` and frontmatter `tags:`).
        - `path_prefix`: a vault-relative path prefix string.
        - `date_range`: `{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}` matched
          against the note's due/deadline/start/created/date field, whichever is set
          (in that priority order).
        - `modified_since`: an ISO date or datetime matched inclusively against
          the last indexed note modification time (`mtime`); timezone-naive
          datetimes use UTC. Timestamp-only edits with unchanged content are
          skipped by incremental and file sync, so use full sync to refresh this
          snapshot. It composes with `date_range`, which continues to inspect
          frontmatter dates.
        - `frontmatter_contains`: a dict of exact frontmatter key/value pairs.
        - `kind`: a source kind string or list of kinds (`"markdown"`,
          `"canvas"`, `"repo_doc"`, `"code"`); a chunk matches if its kind is any of
          them. Chunks indexed before kinds existed count as `"markdown"`.
        - `repo`: a code source name or list of names; only chunks from those
          configured `code_sources` match (their paths start with
          `repo:<name>/`).
        - `doc_role`: a role or list of roles, by the configured `doc_roles`
          globs: `"reference"` (living docs: READMEs, skill docs, and vault
          notes by default), `"record"` (designs, plans, specs, ADRs: intent
          at a point in time; use for "why was X decided"), `"history"`
          (changelogs). Code chunks match no role.
        - `exclude_status`: a list of frontmatter status strings to omit from
          search results. It replaces the configured `exclude_status`; `[]`
          disables status exclusions for this query.
        - Any other key is matched by exact equality against top-level chunk
          metadata (e.g. `status`, `project`, `context`, `note_title`) or the
          note's derived fields.
        """
        return service.query(
            query=query,
            filters=filters,
            top_k=top_k,
            min_score=min_score,
            verbose=verbose,
            recency_boost=recency_boost,
            graph_boost=graph_boost,
        )

    @mcp.tool(name="rag.search")
    def rag_search(
        query: str,
        filters: dict | None = None,
        top_k: int = 10,
        min_score: Annotated[float, Field(strict=True, ge=-1, le=1, allow_inf_nan=False)] | None = None,
        verbose: bool = True,
        recency_boost: Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)] = 0.0,
        graph_boost: Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)] = 0.0,
    ) -> dict:
        """Return raw hybrid retrieval hits for a query.

        Use this when you need the ranked chunks/citations themselves (e.g.
        to synthesize your own answer or inspect retrieval quality). Use
        `rag.query` when you also want a normalized citations list.

        `filters` supports the same keys as `rag.query` — see that tool's
        description for the supported filter shapes, including `exclude_status`
        to override configured status exclusions for one query.

        `min_score` optionally filters by cosine semantic similarity in [-1, 1]
        before final top-k truncation; keyword-only hits are excluded when set.
        `recency_boost` in [0, 1] adds a bounded 30-day-half-life age-decay
        contribution when reranking retrieved candidates; zero preserves
        default ranking. Recency uses the last indexed mtime snapshot.
        `graph_boost` in [0, 1] lifts retrieved chunks from notes that the
        note graph links to the top five hit notes (but not those five), by at most one rank-1
        fusion term scaled by edge strength; zero (default) disables it.
        `verbose` defaults to true; false returns flat compact hits with score
        signals and only the path, title, and heading metadata.
        """
        return service.search(
            query=query,
            filters=filters,
            top_k=top_k,
            min_score=min_score,
            verbose=verbose,
            recency_boost=recency_boost,
            graph_boost=graph_boost,
        )

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

    @mcp.tool(name="rag.related")
    def rag_related(
        note_path: str,
        top_k: int = 10,
        verbose: bool = True,
        edge_types: list[str] | None = None,
    ) -> dict:
        """Return notes associated with one note, ranked by how closely related.

        Association blends four signals — semantic similarity, wikilinks,
        shared tags, and co-mentions by a third note — into one composite
        score per neighbor. Each result's `signals` breaks down the
        individual components when `verbose` is true (the default), while
        `evidence` explains *why* the notes are related (e.g.
        `links_to`/`linked_from`, `shared_tags`,
        `comention_count`), not just that they are. Returns
        `{"found": false, "neighbors": []}` if `note_path` isn't indexed.
        Set `verbose` to false to omit the `signals` breakdown. Reflects the
        graph as of the last `rag.sync`.

        Each neighbor has a `type`: `links` for the blended note-to-note
        association above; `mentions_code` when one side names a code-source
        file, symbol or repo (in inline code, a path, or a GitHub URL);
        `imports` when one source file imports the other. Typed edges add
        `evidence.code_refs`, the mentions or imports that matched. Pass
        `edge_types` (e.g. `["mentions_code"]`) to keep only those types,
        such as "which code does this note discuss?" or, from a code path
        like `repo:<name>/src/x.py`, "which notes discuss this file?".
        """
        return service.related(
            note_path=note_path, top_k=top_k, verbose=verbose, edge_types=edge_types
        )

    @mcp.tool(name="rag.connections")
    def rag_connections(note_a: str, note_b: str) -> dict:
        """Return how closely two notes are associated, direct or via a path.

        If a direct edge exists between the notes, `direct_edge` and
        `closeness` describe it. Otherwise, if they're connected through
        other notes, `path` is the strongest evidentiary chain between them
        (not necessarily the fewest hops) and `closeness` is the product of
        the composite scores along that chain. `connected` is false and
        `closeness` is 0.0 if the notes aren't linked in the graph at all.
        The direct edge and each path hop carry the edge `type` (`links`,
        `mentions_code` or `imports`).
        """
        return service.connections(note_a=note_a, note_b=note_b)

    @mcp.tool(name="rag.drift")
    def rag_drift(
        status: list[str] | None = None,
        note_path: str | None = None,
        repo: str | None = None,
        levels: list[str] | None = None,
        limit: int = 20,
        roles: list[str] | None = None,
    ) -> dict:
        """Return note/code pairs that may disagree, for you to judge.

        Every inline-code name, path or GitHub URL in a note that matches a
        code-source file or symbol (a `mentions_code` edge) is tracked across
        syncs. A pair is `stale` when that code changed after the note last
        changed, and `dangling` when a mention that used to match no longer
        does (`evidence.reason`: `file_removed`, `symbol_removed`, or
        `unresolved` when the code is still there but the name no longer
        resolves to it); `evidence.candidates` lists where a dangling name
        may have gone, as `path` or `path::Qualname`. `status` defaults to
        `["stale", "dangling"]`; pass `["ok"]` to list pairs with no change. `evidence.level` says whether
        the mention names a `symbol` (its whole declaration is compared), a
        `file` (any edit counts) or a `repo` (its README; left out unless
        `levels` includes it, since READMEs change constantly). Symbol pairs
        come first after dangling ones.

        Docs play different parts (`evidence.role`, from the `doc_roles`
        config). `roles` defaults to `["reference"]`: living docs (READMEs,
        skill docs, vault notes by default), where a stale pair means the doc
        may be out of date. `record` docs (designs, plans, specs, ADRs) were
        written against the code of their day, so their pairs go stale as a
        matter of course; instead `records` summarizes each one that moved,
        most moved first: `tracked` mentions, `changed`, `gone`,
        `moved_share` and `gone_mentions`. A record whose code mostly moved
        was superseded or built differently: read it against the current code
        and say which. Pass `roles=["record"]` with `note_path` for its
        pairs. Changelogs (`history`) are not tracked.

        The server does not decide whether the texts disagree. Each pair has
        `note.excerpts` (the note's chunks containing the mention) and
        `code.text` (the symbol's indexed outline: comment, signature and
        docstring, not the body), plus `code.start_line`/`end_line` and
        `code.local_path` to read the body. Judge each pair: the note is
        still right, the note is outdated (say what changed), or the code is
        wrong. Dates are Unix epoch seconds: `code.changed_at` is the code
        file's mtime when the change was first seen; `note.mtime` the note's
        mtime when last indexed. Reflects the last `rag.sync`.
        """
        return service.drift(
            status=status, note_path=note_path, repo=repo, levels=levels, limit=limit, roles=roles
        )

    @mcp.tool(name="rag.map")
    def rag_map(min_score: float | None = None) -> dict:
        """Summarize the vault's note graph as clusters, orphans, and bridges.

        `clusters` are note neighborhoods detected via community detection
        on the association graph, each labeled by its most common tag (or
        its hub note's title) — use this to see what topical areas the
        vault contains. `orphans` are notes with no association above
        `min_score` (defaults to the configured minimum edge score).
        `bridges` are notes whose removal would split their neighborhood
        apart, i.e. notes connecting otherwise-separate clusters. Cluster
        `notes` lists are capped at 25 entries (highest-degree first) — use
        `rag.related` on a cluster's `hub` for the full neighborhood.
        `ghosts` (top 25) are wikilink targets that match no note, ranked by
        how many notes link to them: concepts referenced but not yet written.
        Candidates and evidence are recomputed from eligible stored metadata
        and centroids, including exclusion of co-mention sources and semantic
        neighbors. This read-only view needs no sync after exclusion changes;
        its pairwise cosine matrix has quadratic cost in eligible centroids.
        """
        return service.graph_map(min_score=min_score)

    @mcp.tool(name="rag.sync")
    def rag_sync(mode: str = "incremental", file_path: str | None = None) -> dict:
        """Re-index vault files into the vector, keyword, and graph stores.

        `mode="incremental"` (default) only re-indexes files changed since
        the last sync, and updates graph edges only for the notes affected
        by that change (an approximation of exact recomputation). Pass
        `mode="file"` with `file_path` to sync a single vault file the same
        way; a file that was indexed and is now gone from disk is removed,
        and a path that was never indexed is an error. If the vault folder
        itself is missing, nothing is deleted and `errors` says so. `mode="full"`
        rebuilds every index — including an exact graph rebuild — from
        scratch; use it periodically or if `rag.related`/`rag.connections`/
        `rag.map` results look stale or inconsistent. Call this after vault
        content changes and before relying on `rag.search`/`rag.query`/
        `rag.related`/`rag.connections`/`rag.map` to reflect those changes.
        Incremental and file sync skip timestamp-only changes when a file's
        content hash is unchanged; full sync refreshes the indexed mtime snapshot
        used by `modified_since` filters and `recency_boost`.
        """
        return service.sync(mode=mode, file_path=file_path)

    @mcp.tool(name="rag.status")
    def rag_status() -> dict:
        """Return index and model runtime status for operational visibility.

        In addition to index/model/graph counts, status reports `stale_files`
        (eligible checkpointed files with changed hashes or mtimes, pending
        replacements, or identity migrations, including files that cannot be
        read), `untracked_files` (eligible files on disk without a
        successful checkpoint, including pending-only paths), and `missing_files`
        (eligible checkpointed or pending-replacement paths no longer on disk).
        Eligibility uses scanner rules: files whose extension is in
        `include_extensions` (default `.md` only), excluding hidden path
        segments and configured `exclude_globs`.

        `watcher_running` says whether this server is watching the vault and
        indexing changes in the background (`watch_enabled` in the config);
        when it is true, vault edits are searchable a few seconds after the
        vault goes quiet, without calling `rag.sync`. Code sources are not
        watched.

        `watcher_last_event` is a Unix epoch timestamp in seconds for the last
        observed eligible watcher event, or `null` before one occurs.
        It records event observation separately from sync completion; it is not
        a heartbeat. Quiet vaults, and watchers disabled after an earlier event,
        can retain an old timestamp, so this field alone cannot establish
        watcher liveness. Status does not sync or mutate vault/sync-state data,
        but it scans eligible paths and hashes tracked files to calculate
        staleness. An mtime-only change can remain reported until full sync,
        because incremental sync skips files whose content hash is unchanged.
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

    parser = argparse.ArgumentParser(description="Second Brain MCP server")
    parser.add_argument("--config", default="rag_config.toml")
    args = parser.parse_args()
    server = build_server(args.config, start_watcher=True)
    server.run()


if __name__ == "__main__":
    run()
