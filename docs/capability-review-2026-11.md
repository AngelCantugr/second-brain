# Second Brain capability review — 2026-11-02

Scope: what to improve to (1) raise retrieval quality on today's markdown vault,
(2) support code and other document types, (3) decide whether the graph earns its keep.
Every number below was measured against the live `learning-depot` index
(642 notes, 10,690 chunks, 5,661 edges). Section 5 has the scripts to reproduce them.

## TL;DR

| Question | Verdict |
|---|---|
| Does it work for code / other docs? | **No.** Only `.md` is scanned, and code inside markdown is already mangled. One clean seam (a loader registry) fixes it, but code should be indexed for *why*-knowledge, not to compete with Serena or grep. |
| Is the graph useful? | **Partly.** Links, tags, and co-mentions carry real signal. The semantic signal (half the composite weight) is nearly constant noise, and search never uses the graph. |
| Biggest cheap win | Make parsing and chunking aware of code fences, switch FTS from AND to OR, and add a small golden-query eval set so you can measure every change after it. |

---

## 1. Retrieval quality on today's markdown (fix first)

### A1. The chunker breaks code fences. 11% of chunks are affected.

`chunker._split_sections` and `parser.HEADING_RE` treat any line starting with `#` as a heading, including
comments inside fenced code. `section_text.split()` + `" ".join(...)` then flattens all newlines and indentation.

```text
input:   # Retry helper / ```python / # compute delay / def backoff(n): ... / ``` / #fff #include
tags  : ['fff', 'include']
heads : ['Retry helper', 'compute delay']
chunk : Retry helper  | 'Use backoff for flaky calls. ```python'
chunk : compute delay | 'def backoff(n): return min(2**n, 30) # cap at 30s ``` CSS color: #fff ...'
```

The code block is split in half, a comment became a heading, and indentation is gone.
**1,183 of 10,690 live chunks (11%)** contain a fence, so this already affects your vault today. Mermaid diagrams
and tables are affected the same way.

**Fix:** run one fence-aware tokenization pass (mask ```` ``` ```` / `~~~` blocks and inline code) before extracting
headings, tags, and links. Keep fenced blocks as atomic units when windowing, and preserve newlines in chunk text.
A small markdown AST library would also work, but masking is about 30 lines and keeps the dependency count flat.

### A2. Fake tags from code and mermaid colors feed the graph

Hex colors from mermaid `style fill:#e1f5ff` lines are indexed as tags: `e1f5ff`, `fff4e6`, `ff9900`, `0066cc`,
and at least 15 more. Tags feed the graph's tag-Jaccard signal and the `rag.map` cluster labels, so this is
graph noise as well as filter noise. The A1 masking pass fixes it.

### A3. The FTS keyword leg is an implicit AND of every token

`build_fts_match_query` quotes every token and joins them with spaces, so FTS5 requires **all** of them.
Live comparison:

- `"What are my notes on system design?"`: the keyword leg found exactly one chunk, a blog post that
  literally contains that sentence. The other 4 of the top 5 hits are semantic-only.
- `"system design"`: the keyword leg contributed to 3 of the top 5 hits.

Natural-language questions, which are the main MCP use case, therefore lose the keyword leg of the hybrid
almost entirely.

**Fix:** join the tokens with `OR` (BM25 still rewards chunks that match more terms) and drop a small
stopword list. Also index `note_title` and `heading_path` as their own FTS columns, weighted with
`bm25(chunks_fts, w_text, w_title, w_heading)`. Today only `text` is indexed, so a note titled
"Kafka" with a body that never says "Kafka" can't be found by keyword.

### A4. Navigation boilerplate dominates semantic hits

330 chunks contain 4 or more wikilinks. These are mostly `🔗 Related Notes` and `Navigation` sections from the
daily journal template. For the system-design question, **3 of the top 5 semantic hits were these link lists**
(cosine 0.66–0.68) with no system-design content.

**Fix:** add a configurable `skip_headings = ["🔗 Related Notes", "Navigation"]` setting, or drop sections where
link density is above roughly 50%. Keep their links in the graph metadata and leave them out of the chunks.
Separately, prepend `"{title} > {heading}\n"` to the text you **embed**, and keep the stored text clean.
This contextual-chunk header is cheap and improves recall on short sections.

### A5. Embedding throughput

`OllamaEmbedder.embed` sends one HTTP request per chunk to the legacy `/api/embeddings` endpoint.
Ollama's `/api/embed` accepts a batch (`"input": [...]`), and the measurement scripts below use it.
At ~10k chunks this is tolerable. Once code is indexed it won't be.

On nomic task prefixes (`search_document:` / `search_query:`): I tested them on 40 random chunks. They did **not**
widen the score spread (median query score 0.415 → 0.497, chunk-pair spread got narrower). Adopt them only
if the eval set in A6 shows a gain.

### A6. There is no retrieval-quality measurement

`integration/test_benchmarks.py` measures latency only. Nothing measures whether the right notes come back.
Add `eval/golden.toml` with 20–30 queries, each mapped to the note paths you expect, and a script that reports
recall@5, recall@10, and MRR for `search`. Every fix above, and every graph change below, should move that number.
This is the most important item because it turns the rest of this list from opinion into measurement.

---

## 2. Code and other document types

### Current state: markdown-only by construction

- `scanner.iter_markdown_files` (`rglob("*.md")`) and `scanner.is_eligible_markdown_path` (`suffix == ".md"`)
  are the only two gates. `indexer`, `watcher`, and `service.status` all go through them, so the seam is small and clean.
- `ParsedNote` is markdown-shaped (frontmatter, wikilinks, tasks), and `parse_note` reads UTF-8 text only.
- Your vault already has files the index can't see, outside `node_modules`: 69 `.py`, 57 `.html`, 41 `.astro`,
  23 `.ts`, 17 `.svelte`, 74 `.json`.
- Making `.py` eligible would not be enough: the chunker would flatten it as shown in A1.

### Recommended architecture: a loader registry

```python
class SourceLoader(Protocol):
    kind: str                                   # "note" | "code" | "pdf" | "canvas" | ...
    def matches(self, rel_path: Path) -> bool: ...
    def load(self, path: Path, vault_root: Path) -> ParsedDocument: ...
    def chunk(self, doc: ParsedDocument, cfg: RagConfig) -> list[ChunkRecord]: ...
```

- `ParsedDocument` generalizes `ParsedNote`: common fields (`path`, `title`, `content_hash`, `mtime`, `kind`) plus
  `sections` (headings, symbols, or pages) and `references` (wikilinks, imports, citations).
- The scanner asks the registry instead of checking the suffix. Make it configurable with `[[loaders]]` / `include_globs`
  in `rag_config.toml`, and keep markdown as the default so existing vaults don't change.
- Add `kind` to chunk metadata, so `filters={"kind": "code"}` works through the existing `matches_filters` path.
- If you use a code-tuned embedding model, put it in a separate Qdrant collection or a named vector per kind,
  because cosine scores across models aren't comparable. RRF fuses by rank, so cross-collection fusion still works.

### Strategy for code: index the *why*, not the *what*

You already have Serena (LSP symbols and references), grep, and the agent's own file reads for navigating code.
A RAG copy of a repository is a weaker version of those tools and goes stale quickly. This system can offer
something they can't: **connecting your notes, decisions, and learnings to the code they're about.** Suggested phases:

1. **Repo docs as first-class sources.** Index `README`, `docs/**`, ADRs, and `CHANGELOG` from repositories you list
   in config (`[[sources]] path=… kind="repo"`). These are mostly markdown, so this phase mainly needs a
   multi-root scanner.
2. **Symbol outlines, not file bodies.** Use a tree-sitter loader that emits one chunk per function or class:
   signature + docstring + path + symbol, with `heading_path = "module.Class.method"`. That keeps chunks small,
   stable across edits elsewhere in the file, and citeable.
3. **Note→code edges.** Detect code references in notes (`second_brain/graph.py`, `GraphBuilder`, GitHub blob URLs)
   and add typed graph edges. Then "what have I written about this module?" becomes a graph query.
   That's the capability nothing else in your toolchain has.

### Other formats, by value per effort

| Format | Value | Effort | Notes |
|---|---|---|---|
| `.canvas` (Obsidian) | High | Low | JSON with nodes and edges. They map directly onto graph edges, and text nodes become chunks. |
| PDF (papers, books) | High | Medium | Page-aware chunking with `page` in metadata for citations. Pick a PDF-to-markdown library after checking its current docs. |
| HTML | Medium | Low | Convert to markdown, then reuse the markdown loader. |
| Images (OCR) | Low | High | Defer. |

---

## 3. Graph: is it useful?

### What works

- Link, tag, and co-mention evidence is real and well-explained. `rag.related` returns **why** two notes are
  related (`links_to`, `shared_tags`, `comention_count`), which is rare and valuable.
- The incremental update design (affected paths plus reverse-kNN) is careful and well-tested.

### What doesn't

**G-issue 1: the semantic signal is effectively constant.** Note centroids are compressed toward the vault mean:

| Measure (all 205,761 note pairs) | p1 | p10 | median | p90 | p99 |
|---|---|---|---|---|---|
| Raw centroid cosine | 0.55 | 0.67 | **0.79** | 0.88 | 0.95 |
| Mean-centered centroid cosine | −0.38 | −0.23 | −0.02 | 0.26 | 0.65 |

- **100% of note pairs** clear `graph_semantic_min = 0.35`, so that threshold filters nothing, and every note
  gets exactly k=8 "semantic neighbors" whether or not any are actually related.
- Every pair's semantic term is at least 0.5 × 0.55 = 0.27, which is above `graph_min_edge_score = 0.15`, so that threshold never prunes either.
- The semantic term averages **0.44 of the 0.54 average composite (~81%)**.
- Cause: averaging many chunk vectors converges toward the corpus mean. Random *chunk* pairs have a median of 0.50,
  while random *centroid* pairs have a median of 0.79.
- Example from `rag.related` on the Backend Engineering note: every neighbor has semantic 0.82–0.92, and the ranking is
  decided entirely by the link signal. Daily journals that link to the note outrank topically closer notes.

**G-issue 2: search never uses the graph.** `RagService.search` is vector + FTS + recency. The graph is only
reachable through `rag.related`, `rag.connections`, and `rag.map`, so its main potential payoff, better retrieval,
goes unused.

**G-issue 3: communities are too coarse.** Greedy modularity on the stored graph yields 8 clusters for 627 notes,
and the top three hold 165, 165, and 150 notes. That's too broad to read as topics. With semantic-only edges removed,
you get 12 clusters. Hub notes dominate: `🎯 Principal Engineer Progression` has degree 208, and
`🎓 AI Agents for SDLC` has 193.

**G-issue 4: 26% of wikilinks are unresolved (720 of 2,730).** Some are genuinely dangling, but 138 are
path-qualified `[[folder/Note]]` links that Obsidian resolves and this code doesn't, because `title_key` is the stem only.
Markdown links `[text](note.md)` aren't parsed at all.

### Recommendations

- **G1. Calibrate the semantic signal.** Store the vault-mean vector and use mean-centered cosine. Alternatively,
  score semantic by rank or percentile per note rather than absolute cosine, or keep a semantic edge only when the
  notes are *mutual* kNN neighbors. Express `graph_semantic_min` as a percentile, which stays meaningful when you
  change embedding models. Max-chunk-pair similarity is another option if centroids stay too blunt.
- **G2. Use the graph inside retrieval.** Add a bounded `graph_boost` parameter, mirroring `recency_boost`: after RRF,
  1-hop neighbors of the top hits get at most one rank-1 RRF term, scaled by edge composite. Measure it with A6.
  This is the change that makes the graph pay for itself.
- **G3. Resolve links the way Obsidian does.** Handle path-qualified links, `aliases` frontmatter, and
  `[text](x.md)` links. Surface unresolved targets as **ghost nodes**: concepts you keep referencing but haven't
  written yet, ranked by inbound count. Today's top ones are `developer experience technical strategy` (51) and
  `project summer triangle` (30).
- **G4. Penalize hubs.** Apply an IDF-style damping to link and co-mention evidence from very-high-degree notes
  (daily journals, index notes, progression hubs) so that "everything links here" stops implying relatedness.
- **G5. Make clusters tunable.** networkx 3.6.1 (already locked) has `louvain_communities(resolution=…, seed=…)`.
  Expose `resolution` on `rag.map` and add a deterministic seed.
- **G6. Add typed edges once there are multiple formats.** Add `type` (`links`, `mentions_code`, `imports`, `cites`,
  `canvas`) next to the composite score, so `rag.related` can filter by relationship.

---

## 4. Prioritized roadmap

| Pri | Item | Why now |
|---|---|---|
| P0 | A6 golden-query eval | Lets you measure everything after it |
| P0 | A1 + A2 fence-aware parsing | Fixes 11% of chunks and removes graph tag noise |
| P0 | A3 FTS OR + title/heading columns | Restores the keyword leg for natural-language queries |
| P0 | A4 boilerplate skip + contextual header | Removes the largest source of semantic false positives |
| P1 | G1 semantic calibration | Makes 50% of the graph weight meaningful |
| P1 | G2 graph boost in search | Gives the graph a retrieval payoff |
| P1 | G3 link resolution + ghost nodes | Cheap, accurate, and immediately useful |
| P1 | A5 batched embedding | Required before scaling to code |
| P2 | Loader registry + `kind` metadata; `.canvas`, PDF, HTML loaders | Multi-format foundation |
| P3 | Repo-doc sources → tree-sitter symbol chunks → note↔code edges (G6) | The unique code capability |

---

## 5. Reproduce the measurements

Run from the repo root. All of these are read-only against the index.

```bash
DB=/Users/angelcantu/GithubRepositories/angelcantugr/learning-depot/data/fts.sqlite
# Semantic share of edges, bucketed
sqlite3 -readonly "$DB" "select round(semantic,1) b, count(*) from edges group by b order by b;"
# Fake hex tags
sqlite3 -readonly "$DB" "select value, count(*) from note_meta, json_each(tags_json) where length(value) in (3,6) and value glob '[0-9a-f][0-9a-f][0-9a-f]*' group by value order by 2 desc limit 10;"
```

```bash
# Centroid similarity spread, raw vs mean-centered
uv run --frozen python -I -c "
import sqlite3,numpy as np
c=sqlite3.connect('file:$DB?mode=ro',uri=True)
M=np.stack([np.frombuffer(b,np.float32) for (b,) in c.execute('select centroid from note_meta where centroid is not null')]).astype(float)
M/=np.linalg.norm(M,axis=1,keepdims=True); iu=np.triu_indices(len(M),1)
print('raw     ',np.percentile((M@M.T)[iu],[1,10,50,90,99]).round(3))
C=M-M.mean(0); C/=np.linalg.norm(C,axis=1,keepdims=True)
print('centered',np.percentile((C@C.T)[iu],[1,10,50,90,99]).round(3))"
```
