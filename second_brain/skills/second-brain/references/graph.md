# Graph tools

These read the note graph as of the last `rag.sync`. If results look stale
or inconsistent, see [maintenance.md](maintenance.md).

## rag.related

`rag.related(note_path, top_k=10, verbose=true, edge_types=null)` returns
notes associated with one note, ranked by a composite score that blends
semantic similarity, wikilinks, shared tags, and co-mentions by a third note.

- `signals`: the per-component breakdown. Present when `verbose` is true; set
  `verbose: false` to omit it.
- `evidence`: why the notes relate, for example `links_to` / `linked_from`,
  `shared_tags`, `comention_count`.
- Each neighbor has a `type`: `links` (note-to-note association),
  `mentions_code` (one side names a code-source file, symbol, or repo in
  inline code, a path, or a GitHub URL), or `imports` (one source file
  imports the other). Typed edges add `evidence.code_refs`.
- `edge_types`: keep only those types. `["mentions_code"]` answers "which
  code does this note discuss?". From a code path such as
  `repo:<name>/src/x.py` it answers "which notes discuss this file?".
- If `note_path` is not indexed you get `{"found": false, "neighbors": []}`.

## rag.connections

`rag.connections(note_a, note_b)` says how two notes are associated.

- Direct edge: `direct_edge` and `closeness` describe it.
- Otherwise `path` is the strongest evidentiary chain (not necessarily the
  fewest hops) and `closeness` is the product of composite scores along it.
- `connected: false` with `closeness` 0.0 means no link in the graph.
- The direct edge and each hop carry the edge `type`.

## rag.note_context

`rag.note_context(note_path)` returns a note's chunk IDs and its backlink and
outlink relationships. It does not return the note's text. To read the text,
open the vault file or use `rag.search` / `rag.query` with a query aimed at
that note.

## rag.map

`rag.map(min_score=null)` summarizes the whole graph:

- `clusters`: neighborhoods from community detection, labeled by the most
  common tag or the hub note's title. `notes` lists cap at 25 (highest
  degree first); call `rag.related` on a cluster's `hub` for the rest.
- `orphans`: notes with no association above `min_score` (defaults to the
  configured minimum edge score).
- `bridges`: notes whose removal splits their neighborhood, connecting
  otherwise-separate clusters.
- `ghosts` (top 25): wikilink targets that match no note, ranked by how many
  notes link to them. Concepts referenced but not yet written.

It is read-only and needs no sync after exclusion changes. Its pairwise
cosine matrix costs quadratic time in the number of eligible centroids, so
call it for an overview, not in a loop.
