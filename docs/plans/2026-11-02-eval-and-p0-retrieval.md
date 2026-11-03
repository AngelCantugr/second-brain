# Plan: golden-query eval + P0 retrieval fixes

Spec: `docs/capability-review-2026-11.md` (sections A1–A4, A6). That review is the binding authority.

## Global Constraints

- Python ≥ 3.11, run everything via `uv run --frozen --offline …`. Test command: `uv run --frozen --offline pytest -q` (baseline: 315 passed). Pristine output is required.
- Follow the existing style: module and function docstrings that explain *why*, `from __future__ import annotations`, dataclasses with `slots=True`, plain pytest functions (no classes), and no new third-party dependencies.
- Never touch the user's live index or vault. Tests use `tmp_path`, and `RagService(config, use_in_memory_vector=True)` where a service is needed (see `tests/test_service.py` for fixtures and a fake embedder).
- Every new config key gets: a `RagConfig` field with a default, a `load_config` line, a line in `DEFAULT_INIT_CONFIG` (`second_brain/cli.py`), a line in `rag_config.example.toml`, and a README mention.
- Add one bullet per task under `## Unreleased` in `CHANGELOG.md`.
- One commit per task (fix rounds may add commits). Conventional-commit subject, ending with the trailer:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
- Do not run `git config`, do not push, and do not dispatch subagents.

---

## Task 1: Golden-query retrieval eval harness

**Goal:** measure retrieval quality as note-level recall@5, recall@10, and MRR over a TOML golden set.

**Files:** create `second_brain/evaluation.py`, `tests/test_evaluation.py`, and `eval/golden.example.toml`. Modify `second_brain/cli.py`, `README.md`, and `CHANGELOG.md`.

**Golden file format (TOML):**

```toml
# Each case: a query and the vault-relative note paths a good answer must surface.
[[cases]]
query = "how do I handle retries with exponential backoff"
expected = ["Engineering/Resilience.md"]

[[cases]]
query = "kafka consumer groups"
expected = ["Learning/Kafka.md", "Learning/Streaming.md"]
```

**`second_brain/evaluation.py` API (exact names):**

```python
@dataclass(slots=True)
class GoldenCase:
    query: str
    expected: list[str]

@dataclass(slots=True)
class CaseResult:
    query: str
    expected: list[str]
    ranked_paths: list[str]          # distinct note paths, best-first, at most `depth`
    first_relevant_rank: int | None  # 1-based rank of the first expected path, None if absent
    recall: dict[int, float]         # k -> |expected ∩ ranked_paths[:k]| / |expected|

@dataclass(slots=True)
class EvalReport:
    cases: list[CaseResult]
    recall: dict[int, float]         # k -> mean of per-case recall@k
    mrr: float                       # mean of 1/first_relevant_rank (0 when None)

def load_golden(path: str | Path) -> list[GoldenCase]: ...
def rank_note_paths(hits: list[dict], depth: int) -> list[str]: ...
def evaluate(service, cases: list[GoldenCase], ks: tuple[int, ...] = (5, 10), depth: int = 10) -> EvalReport: ...
def report_to_dict(report: EvalReport) -> dict: ...
```

- `load_golden` validates the input. It raises `ValueError` with a clear message if `cases` is missing or empty, if `query` is not a non-empty string, if `expected` is not a non-empty list of strings, or if a case has unknown keys.
- `rank_note_paths` reads the path from `hit["metadata"]["path"]`, or from `hit["path"]` for compact hits. It dedupes while preserving first-occurrence order and stops at `depth` distinct paths.
- `evaluate` calls `service.search(query=case.query, top_k=MAX_TOP_K, verbose=False)`, imports `MAX_TOP_K` from `second_brain.service`, and computes the metrics. Many chunks share a note, so it needs the large `top_k` to reach `depth` distinct notes. `depth` must be ≥ `max(ks)`; raise `ValueError` otherwise.
- `report_to_dict` returns JSON-safe output with these keys: `{"summary": {"cases": n, "recall@5": …, "recall@10": …, "mrr": …}, "cases": [{"query", "expected", "first_relevant_rank", "recall@5", "recall@10", "top_paths": ranked_paths[:5]}]}`. Round floats to 4 decimals.

**CLI:** add `second-brain eval --golden PATH [--depth 10]`. It prints `json.dumps(report_to_dict(...), indent=2)` like the other commands. Wire it in `main()` next to `search`.

**Tests (`tests/test_evaluation.py`), at minimum:**

1. `load_golden` round-trips a valid file and rejects each invalid shape listed above (parametrize).
2. `rank_note_paths` dedupes and handles both verbose and compact hit shapes.
3. `evaluate` with a stub service whose `search` returns canned hits. Assert exact recall@5, recall@10, and MRR for three cases:
   - first expected path at rank 1
   - only expected path at rank 3 of distinct notes, with duplicate chunks of the same note at earlier ranks
   - expected path absent
4. `depth < max(ks)` raises `ValueError`.
5. A CLI test in the style of the existing `tests/test_config_cli.py`, if that file exercises `main()`. Otherwise test only `report_to_dict`'s shape.

`eval/golden.example.toml`: 3 illustrative cases using generic paths, with a header comment explaining the format. Real golden sets contain personal note paths and belong next to the user's vault config, not in this repo. Say so in the comment and in the README section ("Measuring retrieval quality").

---

## Task 2: Code-fence-aware parsing and chunking (spec A1 + A2)

**Goal:** stop code fences and inline code from producing fake headings, tags, links, and tasks, and preserve the original whitespace (newlines and indentation) in chunk text.

**Files:** modify `second_brain/parser.py`, `second_brain/chunker.py`, `tests/test_parser.py`, `tests/test_chunker.py`, `README.md` (upgrade note), and `CHANGELOG.md`.

**Parser:**

- Add a public function `mask_code(text: str) -> str`. It returns `text` with the *contents* of fenced code blocks and inline code spans replaced by spaces, and with every `\n` kept so line numbers and offsets are unchanged (same length as the input).
  - **Fences:** a line whose first non-space characters (≤3 leading spaces) are ≥3 backticks or ≥3 tildes opens a fence. The fence closes at the next line that starts (≤3 leading spaces) with the same character repeated at least as many times, followed only by whitespace. An unclosed fence extends to end of text, per CommonMark. Mask the opening and closing fence lines as well (the whole block becomes spaces, except newlines).
  - **Inline code:** outside fences, mask backtick spans (`` `x` ``, ``` ``x`` ```) where an opening run of N backticks is closed by the next run of exactly N backticks on the same line. Leave unmatched runs alone.
- In `parse_note`, compute `masked = mask_code(body)` and use `masked` for `WIKILINK_RE`, `_inline_tags`, `HEADING_RE`, and `TASK_RE`. Frontmatter tags are unaffected. `body` itself stays unmasked in `ParsedNote.body`.

**Chunker:**

- `_split_sections` must only treat a line as a heading when the same line in `mask_code(body)` is a heading. Masked lines are blank, so code comments like `# compute delay` can't match. Keep the heading regex as is.
- Section tasks: use `TASK_RE.findall(mask_code(section_text))`.
- **Preserve whitespace in chunk text.** Keep the window semantics (`chunk_size` / `chunk_overlap` count whitespace-separated words, with the same step and overlap as `_token_windows`). Each chunk's `text` must be the *original slice* of `section_text` from the first word's start offset to the last word's end offset, so newlines and indentation inside a window survive. Implement it with `re.finditer(r"\S+", section_text)` spans. Set `bm25_text` to the same text.
- Keep the chunk-id derivation formula unchanged: `json.dumps([note.path, note.note_id, heading_path, i, text])`. Ids will change wherever text gained whitespace, which is expected.

**Tests (add, don't weaken existing ones):**

1. `mask_code`: length preserved and newlines preserved. Backtick fence, tilde fence, a 4-backtick fence containing a 3-backtick line, an unclosed fence, inline code, and an unmatched backtick left intact.
2. `parse_note` on a note with a fenced python block containing `# compute delay`, `#include <x>`, `[[NotALink]]`, and `- [ ] not a task`, plus mermaid `style A fill:#e1f5ff` inside a ```` ```mermaid ```` block, and inline `` `#fff` ``. Assert none of these become headings, tags, links, or tasks, and that real ones outside the code still do.
3. `chunk_note`: a section containing a fenced block with a `# comment` line stays one section under the real heading, and the chunk text contains `"\n"` and the original indentation (`"    return"`).
4. The existing overlap test still passes unchanged (word-level overlap semantics retained).

**README:** add an upgrade note: after upgrading, run `second-brain sync --mode full` once. Incremental sync skips files whose content hash is unchanged, so it won't re-chunk them.

---

## Task 3: FTS keyword leg — OR semantics, stopwords, title/heading columns (spec A3)

**Goal:** natural-language questions must reach the keyword leg, and note titles and headings must be keyword-searchable.

**Files:** modify `second_brain/keyword_store.py`, `tests/test_keyword_store.py`, `CHANGELOG.md`, and `README.md` (one line in the retrieval description).

**Query building — `build_fts_match_query(query: str) -> str`:**

- Tokenize with the existing `_TOKEN_RE`. Drop tokens whose lowercase form is in a module-level `frozenset` named `_STOPWORDS`. Use exactly this list: `a an and are as at be but by can could did do does for from had has have how i if in into is it its me my of on or our should so that the their then there these this those to was we were what when where which who why will with would you your`.
- If every token is a stopword, fall back to all tokens. If there are no tokens at all, return `""` (existing behavior).
- Quote each remaining token as today, and join with `" OR "`. BM25 still ranks chunks that match more terms higher.

**Schema — FTS gains `title` and `heading` columns:**

- New DDL: `CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id UNINDEXED, text, title, heading)`.
- `upsert_chunks` inserts `title = chunk.metadata.get("note_title", "")` and `heading = chunk.heading_path` (use `""` when it is `"root"`).
- Search ranks with `bm25(chunks_fts, 0.0, 1.0, 3.0, 2.0)` (the column order is chunk_id, text, title, heading). Keep `ORDER BY rank` and the `score = -rank` convention.
- **Migration in `initialize()`:** an existing DB has the old 2-column `chunks_fts`. Detect it with `PRAGMA table_xinfo(chunks_fts)` or by reading `sqlite_master.sql`. If `title` is missing:
  1. `DROP TABLE chunks_fts`.
  2. Create the new table.
  3. Repopulate it from the `chunks` table (`text`, plus `note_title` and `heading_path` from `metadata_json`) inside the same connection/transaction.
  
  No re-embedding is needed. This must be idempotent: a second `initialize()` is a no-op.

**Tests:**

1. `build_fts_match_query("What are my notes on system design?") == '"notes" OR "system" OR "design"'`. All-stopword fallback: `"what is it"` → `'"what" OR "is" OR "it"'`. FTS syntax characters are still neutralized (existing test stays green).
2. Search for a natural-language question returns a chunk that contains only some of the terms. This failed under AND semantics; assert it.
3. A chunk whose `note_title` is `"Kafka"` but whose text never says "kafka" is returned for query `"kafka"`.
4. Title and heading matches outrank a single body-text mention, given equal text otherwise. Construct the case to prove the column weights take effect.
5. Migration: create a DB with the *old* DDL and rows (write the old schema directly with `sqlite3` in the test), run `KeywordStore.initialize()`, and assert the new columns exist, the row count is preserved, and a title-only query hits. Then call `initialize()` again and assert nothing changed.

---

## Task 4: Skip boilerplate sections + contextual embedding header (spec A4)

**Goal:** stop navigation boilerplate from being embedded, and give each embedded chunk its note and heading context.

**Files:** modify `second_brain/config.py`, `second_brain/chunker.py`, `second_brain/indexer.py`, `second_brain/cli.py` (`DEFAULT_INIT_CONFIG`), `rag_config.example.toml`, `README.md`, `CHANGELOG.md`, and `tests/test_chunker.py`, plus `tests/test_indexer.py` or `tests/test_config_cli.py` as needed.

**Config (two new keys):**

- `skip_headings: list[str] = []`. In TOML: `skip_headings = []`. Validate with the existing `_validated_string_list`. A section whose heading text matches an entry, compared after `.strip().casefold()` on both sides, is **not chunked**. Its links and tags still count, because the parser reads the whole body. The README should show the example `skip_headings = ["🔗 Related Notes", "Navigation"]`.
- `embed_context_header: bool = True`. In TOML: `embed_context_header = true`.

**Chunker:**

- Signature becomes `chunk_note(note, chunk_size, chunk_overlap, skip_headings: Sequence[str] = ())`. The default keeps every existing caller and test working.
- Add `embedding_text(chunk: ChunkRecord) -> str`. It returns `f"{title} > {heading}\n{chunk.text}"`, using `chunk.metadata["note_title"]` and `chunk.heading_path`. When the heading is `"root"` or empty, it returns `f"{title}\n{chunk.text}"`. Docstring the *why*: short sections lack context, and the header anchors the embedding to the note's topic without polluting stored or keyword text.

**Indexer (`_upsert_parsed`):**

- Pass `skip_headings=self.config.skip_headings` to `chunk_note`.
- Embed `[embedding_text(c) for c in chunks]` when `config.embed_context_header`, otherwise `[c.text for c in chunks]`. Stored `text`, `bm25_text`, and the keyword index stay header-free.

**Tests:**

1. `chunk_note` with `skip_headings=["🔗 related notes"]` drops a section titled `🔗 Related Notes` (case and emoji handled) and keeps other sections.
2. `embedding_text` formats both the root case and the heading case.
3. Indexer: with a recording fake embedder (follow the existing `tests/test_indexer.py` pattern), the texts sent to `embed` start with the header when the flag is on and equal `chunk.text` when it is off. The keyword store's stored text has no header.
4. Config: `load_config` reads both keys and rejects a non-list `skip_headings` with `ValueError`.
