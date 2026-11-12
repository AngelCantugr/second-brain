# Retrieval

## rag.query versus rag.search

Both run the same hybrid (semantic plus keyword) retrieval and take the same
parameters. They differ in output and default `top_k`.

- `rag.query` (default `top_k` 8): returns `chunks` and a normalized
  `citations` list to ground your own synthesis. It generates no answer text.
  Use it when you want evidence to cite.
- `rag.search` (default `top_k` 10): returns the raw ranked hits. Use it to
  inspect ranking quality or to synthesize from the hits yourself.

## Filters

Pass as `filters`, a dict. Keys combine; a chunk must satisfy all of them.

| Key | Shape | Example |
| --- | --- | --- |
| `tags` | tag string or list; case-insensitive; matches inline `#tags` and frontmatter `tags:` | `{"tags": ["adr", "auth"]}` |
| `path_prefix` | vault-relative path prefix string | `{"path_prefix": "projects/ingest/"}` |
| `date_range` | `{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}`; matched against the note's due, deadline, start, created, or date field, first one set in that order | `{"date_range": {"start": "2026-01-01", "end": "2026-03-31"}}` |
| `modified_since` | ISO date or datetime, inclusive, against the last indexed `mtime`; naive datetimes are UTC | `{"modified_since": "2026-10-01"}` |
| `frontmatter_contains` | dict of exact frontmatter key/value pairs | `{"frontmatter_contains": {"status": "accepted"}}` |
| `kind` | source kind or list: `markdown`, `canvas`, `repo_doc`, `code`; older chunks count as `markdown` | `{"kind": ["repo_doc", "code"]}` |
| `repo` | configured code source name or list; those chunks have paths starting `repo:<name>/` | `{"repo": "second-brain"}` |
| `doc_role` | role or list: `reference` (living docs: READMEs, skill docs, vault notes), `record` (designs, plans, specs, ADRs: intent at a point in time), `history` (changelogs); code chunks match no role | `{"doc_role": "record"}` |
| `exclude_status` | list of frontmatter status strings to omit; replaces the configured list; `[]` disables exclusions for this query | `{"exclude_status": []}` |

Any other key is matched by exact equality against top-level chunk metadata
(for example `status`, `project`, `context`, `note_title`) or the note's
derived fields.

Notes:

- `modified_since` reads the indexed `mtime` snapshot. Timestamp-only edits
  with unchanged content are skipped by incremental and file sync, so run a
  full `rag.sync` to refresh it.
- For "why was X decided", use `doc_role: "record"`. For what is true now,
  use `doc_role: "reference"`.

## Tuning

- `top_k`: result count. Start at 5 to 8.
- `verbose`: defaults to true (full chunk metadata). Set false for flat
  compact chunks: `chunk_id`, score signals, `text`, `path`, `note_title`,
  `heading_path`. Prefer false.
- `min_score`: cosine semantic similarity floor in [-1, 1], applied before the
  `top_k` cutoff. Keyword-only hits are excluded when it is set.
- `recency_boost`: in [0, 1]. Adds a bounded age-decay (30-day half-life)
  contribution when reranking. 0 (default) keeps the default ranking. Uses
  the last indexed mtime.
- `graph_boost`: in [0, 1]. Lifts chunks from notes that the note graph links
  to the top five hit notes (but not those five), scaled by edge strength.
  0 (default) disables it.

`debug_scores` holds the fused ranking scores, including any recency
contribution. Each chunk also exposes its semantic and keyword scores.
