# Maintenance

## rag.status: is the index stale?

Call `rag.status()` when results look out of date. Fields that signal
staleness:

- `stale_files`: indexed files whose hash or mtime changed, with pending
  replacements or identity migrations, including files that cannot be read.
- `untracked_files`: eligible files on disk with no successful checkpoint.
- `missing_files`: indexed or pending files no longer on disk.
- `watcher_running`: true when the server is watching the vault. Edits are
  then searchable a few seconds after the vault goes quiet, with no sync
  needed. Code sources are not watched.
- `watcher_last_event`: epoch seconds of the last eligible event, or `null`.
  Not a heartbeat; do not use it alone to decide the watcher is alive.

Eligible files have an extension in `include_extensions` (default `.md` only),
no hidden path segments, and no match in `exclude_globs`. An mtime-only
change can stay reported until a full sync.

## rag.sync: refresh the index

`rag.sync(mode, file_path)`:

- `incremental` (default): re-index files changed since the last sync and
  update graph edges only for affected notes. Semantic scores stay on the
  scale of the last full sync.
- `file` with `file_path`: sync one vault file. A previously indexed file now
  gone from disk is removed; a path never indexed is an error.
- `full`: rebuild every index and rescore the whole graph. Needed when
  `rag.related`, `rag.connections`, or `rag.map` look stale or inconsistent,
  and to refresh the `mtime` snapshot behind `modified_since` and
  `recency_boost`.

If the vault folder is missing, nothing is deleted and `errors` says so.
Incremental and file sync skip timestamp-only changes when the content hash
is unchanged.

## rag.health: is a dependency down?

`rag.health()` checks the vector store, embedder, and keyword DB are
reachable. Call it when a tool call fails. `rag.status` reports index
contents; `rag.health` reports liveness.

## rag.drift: notes that may disagree with code

`rag.drift` returns note and code pairs for you to judge. The server does
not decide whether the texts disagree.

Parameters: `status` (default `["stale", "dangling"]`; `["ok"]` lists
unchanged pairs), `note_path`, `repo`, `levels`, `limit` (default 20),
`roles`.

- `stale`: the code changed after the note last changed.
- `dangling`: a mention that used to match no longer does.
  `evidence.reason` is `file_removed`, `symbol_removed`, or `unresolved`
  (code still there, name no longer resolves). `evidence.candidates` lists
  where it may have gone.
- `evidence.level`: `symbol`, `file`, or `repo`. `repo` pairs (READMEs) are
  left out unless `levels` includes them.
- `roles` defaults to `["reference"]`: living docs, where a stale pair means
  the doc may be outdated.
- `record` docs (designs, plans, specs, ADRs) go stale as a matter of course.
  Their pairs are replaced by `records`, a summary per moved record, most
  moved first: `tracked`, `changed`, `gone`, `moved_share`,
  `gone_mentions`. A record whose code mostly moved was superseded or built
  differently: read it against current code and say which. Pass
  `roles: ["record"]` with `note_path` for its pairs.
- Changelogs (`history`) are not tracked.

Each pair carries `note.excerpts` (chunks containing the mention) and
`code.text` (the symbol's indexed outline: comment, signature, docstring;
not the body), plus `code.start_line`, `code.end_line`, and `code.local_path`
to read the body. Dates are Unix epoch seconds.

Judge each pair as one of:

1. The note is still right.
2. The note is outdated (say what changed).
3. The code is wrong.

Report your judgments. Do not edit notes unprompted. Results reflect the last
`rag.sync`.
