# Graph tools

These read the note graph as of the last `rag.sync`. If results look stale
or inconsistent, see [maintenance.md](maintenance.md). Pass a file as
`{"bank": "work", "path": "Projects/Roadmap.md"}`. Ordinary wikilinks stay
within their bank. Semantic relationships can span banks, as can explicit
wikilinks such as `[[bank:work/Projects/Roadmap.md]]`.

## rag.related

`rag.related(file, top_k=10, verbose=true, edge_types=null, banks=null)` returns
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
  `{"bank": "repo", "path": "src/x.py"}` it answers "which notes discuss this file?".
- If the file is not indexed you get `{"found": false, "neighbors": []}`.
- `top_k` counts distinct files. The strongest matching edge supplies the
  score, and `alternate_edges` holds the other matching edge details.

`banks` restricts neighbors while allowing any indexed anchor.

## rag.connections

`rag.connections(file_a, file_b, banks=null)` says how two notes are associated.

- Direct edge: `direct_edge` and `closeness` describe it.
- Otherwise `path` is the strongest evidentiary chain (not necessarily the
  fewest hops) and `closeness` is the product of composite scores along it.
- `connected: false` with `closeness` 0.0 means no link in the graph.
- The direct edge and each hop carry the edge `type`.
- `banks` restricts intermediate files. Endpoints can come from any bank.

## rag.note_context

`rag.note_context(file)` returns chunk IDs and resolved backlinks and
outlinks. Call `rag.fetch(file)` for the indexed text.

## rag.map

`rag.map(min_score=null, banks=null)` summarizes the whole graph:

- `clusters`: neighborhoods from community detection, labeled by the most
  common tag or the hub note's title. `notes` lists cap at 25 (highest
  degree first); call `rag.related` on a cluster's `hub` for the rest.
- `orphans`: notes with no association above `min_score` (defaults to the
  configured minimum edge score).
- `bridges`: notes whose removal splits their neighborhood, connecting
  otherwise-separate clusters.
- `ghosts` (top 25): wikilink targets that match no note, ranked by how many
  notes link to them. Concepts referenced but not yet written.

Use `banks` to restrict the map while retaining shared-graph edge scores.

It is read-only and needs no sync after exclusion changes. Its pairwise
cosine matrix costs quadratic time in the number of eligible centroids, so
call it for an overview, not in a loop.
