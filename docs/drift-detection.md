# Drift detection between notes and code — 2026-11-04

Scope: find places where a vault note (or a repo doc) and the code it describes stopped
saying the same thing, because one side changed without the other, or because they
disagree. Builds on the typed graph edges (G6): `mentions_code` already links a note to a
code-source file it names.

The server runs no LLM, so it can find *candidates* (something changed, a name stopped
resolving) but cannot decide whether two texts disagree. That decision goes to the calling
agent through a new tool, `rag.drift`.

## TL;DR

| Question | Verdict |
|---|---|
| Explicit pairs: can we tell that mentioned code changed after the note, or that a mention went dangling? | **Not before this change:** typed edges are recomputed whole each sync and nothing about the previous resolution survives. **Prototyped:** a per-symbol body hash plus one row per `(note, mention)` holding the fingerprint the note was last written against. |
| Implicit pairs: is chunk cosine a usable signal? | **As a ranking, yes; as a detector, no.** It separates explicit pairs from other note/code pairs (AUC 0.89), but the scores bunch between 0.77 and 0.83 and the top pairs are about half "describes this code". Not built; proposed as an opt-in candidate list. |
| Disagreement | `rag.drift` returns each candidate pair with the note excerpt, the code outline, change dates and evidence. The agent judges. |

---

## 1. Explicit pairs (built)

### What existed

| Data | Where | Changes when | Useful for drift? |
|---|---|---|---|
| File content hash, mtime | `sync_state.note_state` | the file changes (full syncs rewrite it unchanged) | file level only; a one-line edit anywhere in a 600-line module flags every note that mentions the module |
| Chunk ids | Qdrant / FTS | the chunk's text changes | a symbol chunk is the outline (comment, signature, docstring), so a body-only change keeps its id |
| `note_meta.code_json` | `fts.sqlite` | the file is re-indexed | holds mentions, symbols and imports, but no fingerprints |
| `mentions_code` edges | `edges` (`refs_json` = mention text) | every sync, recomputed whole | target is a file, not a symbol; a mention that stops resolving just drops its edge |

So the two missing pieces were a fingerprint per symbol that covers the body, and memory of
what each mention pointed at last time.

### Stored state

- `CodeSymbol.body_hash`: sha256 (16 hex chars) of the whole declaration, from its first
  line (decorator or `export`) to its end, dedented and with trailing spaces stripped. Adding
  code above a function, or nesting a class, keeps the hash; editing the body changes it. It is
  read through redaction like every other symbol field, so the hash never covers a matched secret.
- `note_meta.code_json` gains `hash` and `mtime` on every code-source file and on every note that
  has mentions, and `symbol_hashes` (`{qualname: body_hash}`) on source files. A name defined twice
  in one file (overloads, a conditional `def`) gets one hash over both definitions.
- `code_mentions` table in `fts.sqlite`, one row per `(src, mention)`:

| Column | Meaning |
|---|---|
| `target`, `symbol` | the file the mention resolves to, and the qualname if it names a symbol (`GraphBuilder.rebuild_full`, `graph.GraphBuilder`); `NULL` for a path, URL or module |
| `src_hash` | the note's content hash at the last sync |
| `acked_hash` | the target's fingerprint when the note last changed: what it was written against |
| `current_hash` | the target's fingerprint now; `NULL` once dangling |
| `changed_at` | the target file's mtime at the sync that first saw `current_hash`; `NULL` until a change is seen |
| `lost_at`, `reason` | when a mention stopped resolving, and why: `file_removed`, `symbol_removed`, `unresolved` (ambiguous now, or out of scope) |
| `status` | `ok`, `stale` or `dangling` |

The fingerprint is `symbol_hashes[symbol]` for a symbol mention and the file's `hash` otherwise.
`code_graph.typed_graph` returns, alongside the edges, what each mention resolved to; the
symbol comes from `_CodeIndex.symbol_in`, which only names a symbol the mention spells
(full or trailing qualname, or a terminal name unique in that file).

### State machine

Computed in `drift.mention_states` every time typed edges are recomputed (every sync that
changes anything) and written in the same transaction as the edges.

| Previous row | This sync | Result |
|---|---|---|
| none | mention resolves | `ok`, `acked = current` (baseline: nothing is stale on the first sync) |
| any | note's hash changed | `acked = current`: the note was rewritten, so it is taken as written against today's code |
| any | note unchanged, fingerprint differs from `acked` | `stale`; `changed_at` = target mtime when the fingerprint first changed |
| resolved | mention no longer resolves | `dangling`, keeps the old target and symbol, `lost_at` = now (kept on later syncs) |
| any | note no longer contains the mention, or note deleted | row dropped |
| none | mention never resolved | no row (most inline code never named code) |

Choices and their costs:

- **Note hash, not note mtime, decides "written against".** Comparing a note's mtime with its
  code's would flag most notes at once, because `git clone`, `checkout` and `pull` set code mtimes to
  the day they ran. Hashes do not care, and a full sync, which reprocesses every note without
  changing it, does not clear staleness.
- **Any edit to the note re-acknowledges all of its mentions**, including a typo fix elsewhere in a
  long note. The alternative, a per-mention "reviewed" marker the user sets, was judged too much
  ceremony for a personal vault. A follow-up could ack only the mentions whose chunk changed.
- **`changed_at` is when the change reached this machine**, not when it was committed. True dates
  need `git log -L` per symbol: possible later, too slow for every sync.
- **Retargeting counts as drift.** If a name used to resolve to `a.py` and now to `b.py` (the symbol
  moved), the fingerprints differ and the pair is `stale` with the new target.
- **Indexes from before this change** have no hashes. A note without `hash` is skipped; a target
  without one reads as unknown and becomes the baseline once its file is indexed again. One
  `second-brain sync --mode full` makes everything tracked.

### On a real change: nine merged pull requests

Same live-index copy, with this repository as a code source: a full sync with the code at
PR #51 as the baseline, then an incremental sync with the code at PR #60 (nine merged PRs: batched
embeddings, link resolution, loader registry, code indexing, typed edges). Nothing in the vault
changed between the two syncs.

| | Pairs |
|---|---|
| Tracked mentions | 76 (19 from vault notes, 57 from this repo's own docs) |
| `dangling` | 2: both renames (`is_eligible_markdown_path`, `iter_markdown_files`), and in each case `candidates` lists the new name first (`is_eligible_source_path`, `iter_source_files`) |
| `stale`, symbol mentions | 17 |
| `stale`, file mentions | 25 |
| `stale`, repo mentions | 8 (every note that names the repo: its README changed) |
| `ok` | 24 |

What this showed:

- **Repo mentions are noise.** The README changes in almost every PR, so `rag.drift` leaves
  `repo`-level pairs out by default (`levels`), which brings the list from 52 to 44.
- **Symbol pairs are the useful ones, and they come first.** File mentions flag a note for any
  edit to the file. A class mention covers the whole class body, so `RagService` behaves like a
  file mention; method mentions (`OllamaEmbedder.embed`, `Indexer._upsert_parsed`) are precise.
- **Dated docs drift by design.** 32 of the stale pairs come from this repo's own docs, nearly all of them
  plan and review documents, which describe the code as it was on the day they were written. They are correctly
  flagged, but the agent should not try to "fix" them. A follow-up could let a code source exclude
  such docs from drift (`docs/plans/**`).
- The note that still names the replaced embedding endpoint appears as a stale file-level
  pair (it mentions the embedder module by file name).

## 2. Implicit pairs (measured, not built)

Notes often describe code without naming it in inline code. Could high cosine between a vault
chunk and a `kind = "code"` chunk find those?

Measured on a copy of the live index (651 vault notes, 10,272 vault chunks) with this repository
added as a code source (60 files, 940 code chunks, 18 explicit vault→code-file pairs from 10
notes), using nomic-embed-text vectors already in the index. A pair's score is the highest
chunk-to-chunk cosine between the note and the file.

| Measure | Value |
|---|---|
| Explicit pairs, cosine p10 / p50 / p90 | 0.713 / 0.754 / 0.814 |
| All other note/file pairs, p10 / p50 / p90 | 0.614 / 0.667 / 0.734 |
| AUC, explicit vs other | 0.893 (0.881 with mean-centered vectors) |
| Rank of the mentioned file among 60, per note | median 2; hit@1 0.39, hit@5 0.56, hit@10 0.72 |
| Pairs with cosine ≥ 0.80 / ≥ 0.75 / ≥ 0.70 (of 39,060) | 12 (4 explicit) / 106 (10) / 753 (17) |

Hand-judging the top 25 distinct (note, file) chunk pairs for "this note passage describes this
code": 9 yes, 4 weak, 12 no. Seven of the 12 misses were test files; without tests, 9 of 18 were
yes. The misses are general RAG concept notes matching module docstrings: same topic, not a
description. Mean-centering spreads the scores but does not separate them better.

One of the top pairs was a real disagreement: a learning note that still describes the
embedding endpoint replaced in the batched-embedding change. It also names the module in inline
code, so the explicit path catches it too.

**Verdict:** cosine ranks candidates well enough for an agent to review a short list, and no
threshold makes it a fact. If built, it should be an opt-in mode of `rag.drift`
(`include_implicit=true`) returning the top few code chunks per changed note, tests excluded,
with `evidence.cosine` and no status of its own. It would need its own golden set of
"note describes symbol" pairs before any default changes.

## 3. Disagreement: `rag.drift` (built)

Input:

| Parameter | Default | Meaning |
|---|---|---|
| `status` | `["stale", "dangling"]` | any of `ok`, `stale`, `dangling` |
| `note_path` | all | only mentions in this note |
| `repo` | all | only mentions of this code source |
| `levels` | `["symbol", "file"]` | what the mention names: a `symbol`, a `file`, or a whole `repo` (its README) |
| `limit` | 20 (max 50) | pairs returned; dangling first, then symbol before file before repo, then most recent change |

Output:

```json
{
  "counts": {"ok": 41, "stale": 3, "dangling": 1},
  "total": 4,
  "pairs": [
    {
      "status": "stale",
      "note": {"path": "Notes/Retries.md", "title": "Retries", "mtime": 1793700000.0,
               "excerpts": ["I capped `compute_backoff` at 30s."]},
      "code": {"path": "repo:svc/pkg/retry.py", "symbol": "compute_backoff", "exists": true,
               "changed_at": 1793770000.0, "mtime": 1793770000.0,
               "local_path": "/src/svc/pkg/retry.py", "start_line": 12, "end_line": 20,
               "text": "def compute_backoff(n):\n    \"\"\"Delay before retry n.\"\"\""},
      "evidence": {"mention": "compute_backoff", "level": "symbol"}
    },
    {
      "status": "dangling",
      "note": {"...": "..."},
      "code": {"path": "repo:svc/pkg/retry.py", "symbol": "backoff", "exists": true, "...": "..."},
      "evidence": {"mention": "backoff", "level": "symbol", "reason": "symbol_removed", "lost_at": 1793770000.0,
                   "candidates": ["repo:svc/pkg/retry.py::backoff_delay"]}
    }
  ]
}
```

- `note.excerpts`: up to two of the note's chunks that contain the mention (1,500 characters each).
- `code.text`: the symbol's indexed outline (comment, signature, docstring), or a file's first chunk.
  Bodies are not indexed by design, so `local_path` and the line span tell the agent where to read it.
- `evidence.candidates` (dangling only): close names in the same file, the same name elsewhere in the
  repo, files of the same name, or the files that made the name ambiguous.

The tool description tells the agent to classify each pair as "note still right", "note outdated
(say what changed)" or "code wrong". The server records nothing about the verdict; editing the
note re-acknowledges it on the next sync.

## 4. Testing

`tests/test_drift.py`:

- `symbol_in` resolves only spelled symbols; `typed_graph` reports each mention's target.
- State machine on plain metadata: baseline is `ok`; a changed body under an unchanged note is
  `stale` and stays so through a full sync; editing the note clears it; a change to another symbol
  in the file leaves a symbol mention alone but flags a file mention; a renamed symbol dangles from
  the sync that lost it, keeps `lost_at`, and suggests the new name; removed files and ambiguous
  names get their own reasons; rows from before hashes existed are skipped, then baselined.
- `body_hash` ignores position and trailing spaces, follows body edits.
- `mention_level` tells symbol, file and repo mentions apart (bare repo name, GitHub URL with and
  without a path).
- End to end through `RagService` with tree-sitter: stale pair with excerpt, outline, line span,
  `changed_at` and `local_path`; rename → dangling with a candidate; ordering, `levels`, filters
  and validation.

## 5. Follow-ups

- Let a code source exclude dated docs (plans, reviews) from drift, or report them separately.
- For a class mention, compare only the class's own lines (signature, docstring, fields) rather
  than its whole body, so it stops behaving like a file mention.
- Implicit candidates (section 2), behind a flag, with a golden set first.
- Ack only the mentions whose chunk changed, instead of every mention in an edited note.
- File-level stale pairs could list which symbols changed, if the previous `symbol_hashes` were kept.
- Commit dates from git for `changed_at`.
- The watcher ignores code sources, so code changes are seen at the next manual or scheduled sync.
