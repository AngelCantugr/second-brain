"""Chunk-parameter sweep harness for the autoresearch classic loop.

Builds a deterministic synthetic vault with a golden query set, indexes it
under a parameter set (chunk_size, chunk_overlap, embed_context_header,
skip_headings), and prints note-level recall@5 as the loop metric. Index
size and latency are reported alongside as STATS/sidecar so the loop log
captures the recall vs. size/latency trade-off.

Phase 1 (default) swaps in a deterministic hashed bag-of-words embedder so
iterations are cheap and offline; pass embedder = "ollama" in the params
file to confirm a winner against the real Ollama + Qdrant stack.

Usage: python harness.py --params params.toml [--json-out sidecar.json]
Last stdout line: METRIC <recall@5>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import time
import tomllib
from pathlib import Path

from second_brain.config import RagConfig
from second_brain.evaluation import evaluate, load_golden, report_to_dict
from second_brain.service import RagService

VAULT_NOTES: dict[str, str] = {
    "retrieval/reciprocal-rank-fusion.md": """---
tags: [retrieval, ranking]
---
# Reciprocal Rank Fusion

Reciprocal rank fusion merges semantic and keyword rankings without
calibrating their scores. Each candidate receives one contribution per
retrieval leg, computed as one divided by the rank plus a dampener.

## Why RRF

Score scales differ between vector cosine and keyword frequency, so adding
scores directly rewards whichever leg is over-confident. Rank-based fusion
treats both legs as equally trustworthy and is robust to outliers.

## Tuning the dampener

The dampener k controls how quickly contributions decay with rank. The
conventional default is sixty; larger values compress rank gaps, which helps
when both legs return noisy orderings.

## Failure modes

Duplicate chunks from one note consume result slots, so a note with many
near-identical chunks can crowd out distinct notes at small top_k values.

## 🔗 Related Notes
- [[hybrid-search]]
- [[note-graph]]
- [[recall-mrr]]
""",
    "retrieval/hybrid-search.md": """---
tags: [retrieval, vector, keyword]
---
# Hybrid Search

Hybrid search runs a dense vector leg and a sparse keyword leg over the same
candidate pool, then fuses the two rankings into one ordered result list.

## The semantic leg

Dense retrieval embeds the query and finds the chunks with the highest
cosine similarity. It paraphrases well: a query that never names the concept
literally still finds the right passage.

## The keyword leg

Sparse retrieval matches query terms against an inverted index. It is exact:
a rare token such as an error message or a filename is found precisely, even
when the surrounding prose says nothing similar.

## Why fuse

Each leg compensates for the other's blind spot. Semantic search misses
exact identifiers; keyword search misses paraphrase. Fusing both legs keeps
precision while recovering recall.

## 🔗 Related Notes
- [[reciprocal-rank-fusion]]
- [[embeddings]]
- [[chunking]]
""",
    "indexing/chunking.md": """---
tags: [indexing, parsing]
---
# Chunking

Chunking splits long notes into retrieval-sized passages before embedding.

## Windowing strategy

Notes are split on headings first, then windowed into fixed-size chunks
with a configurable overlap. Overlap preserves context at boundaries so a
sentence cut in half still has its setup in the neighbouring chunk.

## Size trade-offs

Small chunks isolate a fact, which helps precise questions, but multiply
the number of embeddings and shrink the context each passage carries.
Large chunks carry context, which helps conceptual questions, but dilute
the signal of any single fact and can exceed the embedding model's limit.

## Metadata per chunk

Every chunk records its source path, its headings, and whether its text
starts a section, so retrieval can show provenance and rerank by structure.

## 🔗 Related Notes
- [[embeddings]]
- [[markdown-parsing]]
- [[recall-mrr]]
""",
    "indexing/embeddings.md": """---
tags: [indexing, vector]
---
# Embeddings

Embeddings turn text into dense vectors so semantic similarity becomes
cosine distance in a high-dimensional space.

## Local embedding service

The indexer sends chunks to a local Ollama embedding model in batches.
Batching amortizes request overhead: embedding fifty chunks in one call is
far faster than fifty round trips.

## Dimension and model choice

The embedding model fixes the vector dimension. Switching models requires
re-indexing everything, because old and new vectors are not comparable.

## Mean centering

Note centroids used by the association graph subtract the vault mean, so
similarity spreads out and near-duplicate topical clusters separate.

## 🔗 Related Notes
- [[hybrid-search]]
- [[note-graph]]
- [[ollama-client]]
""",
    "indexing/sync-state.md": """---
tags: [indexing, storage]
---
# Sync State

The sync state records which files have been indexed and content hashes
for each, so re-syncing skips files that have not changed.

## Tracking renames

A file is tracked by its vault-relative path. When a file is renamed, the
old path disappears and the new path appears as an untracked file, so the
old chunks are deleted and the content is embedded once under its new name.

## Checkpoints

A checkpoint commits after each batch, so an interrupted sync resumes near
where it stopped instead of starting over. Files indexed after the last
checkpoint are re-processed.

## Deletion handling

Deleting a note removes its chunks from the vector index, its rows from the
keyword index, and its entry from the graph in the same pass.

## 🔗 Related Notes
- [[chunking]]
- [[toml-config]]
- [[backups]]
""",
    "vault/markdown-parsing.md": """---
tags: [vault, parsing]
---
# Markdown Parsing

The parser extracts frontmatter, headings, tags, and wikilinks from notes.

## Code fences

Fenced code blocks are scanned as literal text: markdown syntax inside a
fence is never mistaken for a heading or a tag, so a comment containing a
hash does not pollute the tag index.

## Heading levels

Headings at every level become section boundaries for chunking, and the
heading path of each chunk is recorded so a retrieval hit can show the
section it came from.

## Skip sections

Navigation sections that repeat on every note, such as related-notes
blocks, can be dropped before windowing so they never dominate a chunk.

## 🔗 Related Notes
- [[wikilinks]]
- [[chunking]]
- [[toml-config]]
""",
    "vault/wikilinks.md": """---
tags: [vault, graph]
---
# Wikilinks

Wikilinks connect notes with double-bracket references such as
[[recall-mrr]] and carry the note graph's explicit edges.

## Aliases

A link may display different text than its target: [[recall-mritic|recall
metrics]] still resolves to the note about evaluation. Aliases are parsed
so the displayed text is searchable while the target stays canonical.

## Path-qualified links

Notes in subfolders can be linked by vault-relative path when titles
collide, so two notes sharing one title still resolve unambiguously.

## Unresolved targets

A wikilink whose target matches no note is kept as a ghost: it contributes
no edge, but it is listed so a missing note is visible as a gap.

## 🔗 Related Notes
- [[note-graph]]
- [[markdown-parsing]]
- [[roadmap]]
""",
    "api/ollama-client.md": """---
tags: [api, embedding]
---
# Ollama Client

The Ollama client talks to a local embedding and generation server over
HTTP.

## Batching protocol

Embedding requests pack many chunks into one JSON payload. The server
embeds them in one pass and returns vectors in request order, so the client
can zip texts and vectors without matching by content.

## Health checks

Before syncing, the client checks that the embedding model is loaded. A
missing model fails fast with a clear error instead of a per-request
timeout during the sync.

## Retries

Transient connection failures are retried with backoff. A persistent
failure aborts the batch and reports the affected files in the sync result.

## 🔗 Related Notes
- [[embeddings]]
- [[mcp-server]]
- [[backups]]
""",
    "api/mcp-server.md": """---
tags: [api, mcp]
---
# MCP Server

The MCP server exposes search, query, and sync to assistants over the
Model Context Protocol.

## Tool surface

Search returns compact ranked chunks with citation metadata; query assembles
context from the top chunks; sync refreshes the index. Each tool validates
its arguments before touching the stores.

## Transport

The server speaks stdio, so an assistant launches it as a subprocess and
exchanges JSON-RPC messages over pipes. No network port is opened.

## Context budgets

Responses respect a max-context-chunks budget, truncating the lowest-ranked
chunks first so the most relevant text always fits.

## 🔗 Related Notes
- [[ollama-client]]
- [[recall-mrr]]
- [[roadmap]]
""",
    "config/toml-config.md": """---
tags: [config]
---
# TOML Configuration

The service is configured from a TOML file: store paths, chunking knobs,
graph weights, and the bank registry.

## Banks

Each bank declares an id, a type (knowledge-base or repo), and a root path.
Repo banks may restrict languages. Per-bank settings live inside the bank
table rather than at the top level.

## Chunk knobs

Chunk size and overlap are validated on load: overlap must be smaller than
the chunk size, and both must be positive. A bad value fails at startup,
not mid-sync.

## Redaction

Redaction patterns are applied before embedding, so a matched secret is
replaced by a token and never leaves the machine, even inside a vector.

## 🔗 Related Notes
- [[banks]]
- [[sync-state]]
- [[markdown-parsing]]
""",
    "config/banks.md": """---
tags: [config, banks]
---
# Banks

Banks partition the indexed corpus into independently rooted collections.

## Knowledge-base banks

A knowledge-base bank holds markdown notes. Its files are parsed, chunked,
embedded, and linked into the association graph like any vault.

## Repo banks

A repo bank holds source code, indexed as symbol outlines rather than
bodies, so a conceptual question finds the file that answers it without
drowning in implementation detail.

## Scoping retrieval

Searches can be scoped to specific banks, and filters such as modified-since
or frontmatter status narrow results further without rebuilding the index.

## 🔗 Related Notes
- [[toml-config]]
- [[sync-state]]
- [[note-graph]]
""",
    "graph/note-graph.md": """---
tags: [graph, associations]
---
# Note Graph

The note graph links notes by four signals: semantic similarity, wikilinks,
shared tags, and co-mentions by a third note.

## Edge composite

Each edge's composite score blends the four signals with configurable
weights. Semantic similarity contributes the vault-mean-centered cosine of
the note centroids; links and tags are exact; co-mentions scale with how
many notes name both endpoints.

## Neighbour queries

Given a note, the graph returns its strongest neighbours, which retrieval
can use to lift related notes past close competitors in the fused ranking.

## Maintenance

Edges are recomputed when affected notes change; an incremental sync only
rebuilds the neighbourhood of the notes that moved.

## 🔗 Related Notes
- [[edge-weights]]
- [[wikilinks]]
- [[embeddings]]
""",
    "graph/edge-weights.md": """---
tags: [graph, associations]
---
# Edge Weights

Edge weights decide which neighbours the graph reports and how strongly
retrieval lifts them.

## Signal weights

Semantic, link, tag, and co-mention signals each have a weight; the
composite is their weighted sum. A vault heavy on wikilinks can raise the
link weight so explicit references dominate accidental topical overlap.

## Minimum edge score

Edges below a minimum composite are dropped, keeping the graph sparse
enough that neighbour queries stay fast and noise links do not rerank
results.

## Tuning loop

Weights are tuned against golden queries: lift too aggressive and
off-topic neighbours pass relevant notes; too weak and the graph does
nothing at all.

## 🔗 Related Notes
- [[note-graph]]
- [[recall-mrr]]
- [[toml-config]]
""",
    "eval/recall-mrr.md": """---
tags: [evaluation]
---
# Recall and MRR

Golden queries score retrieval at the note level with recall at k and mean
reciprocal rank.

## Recall at k

Recall at k divides the expected notes found in the top k by the total
expected notes for the query. Note-level ranking collapses chunk hits, so a
note with three chunks in the results counts once, at its best rank.

## Mean reciprocal rank

Reciprocal rank rewards getting a relevant note to the top of the list:
rank one scores one, rank two one half, and so on, averaged over cases.

## Guarding changes

Every parsing or ranking change is checked against the golden set before it
lands, so a refactor that quietly shifts term weighting is caught by a
dropped recall number rather than a user complaint.

## 🔗 Related Notes
- [[reciprocal-rank-fusion]]
- [[edge-weights]]
- [[mcp-server]]
""",
    "ops/backups.md": """---
tags: [ops]
---
# Backups

Backups protect the index stores and the sync state.

## What to back up

The vector store, keyword index, sync state, and configuration together
describe the entire index. The vault itself is backed up separately by
whatever syncs the notes between machines.

## Restore procedure

Restoring from backup means copying the stores back and restarting the
service. A mismatch between sync state and stores is repaired by a full
re-index, which the sync command performs on demand.

## Rotation

Weekly rotating backups keep disk use bounded; a backup older than the
oldest vault change still restores correctly because the sync state treats
every file as dirty.

## 🔗 Related Notes
- [[sync-state]]
- [[ollama-client]]
""",
    "meta/roadmap.md": """---
tags: [meta]
---
# Roadmap

The roadmap tracks planned work on the retrieval pipeline.

## Current focus

Chunking quality and reranking are the active threads: sweeps over chunk
size and overlap, and boosts that lift related notes without letting the
graph overwhelm the fused ranking.

## Evaluation work

The golden set grows with every reported miss. A query that fails to
surface its note becomes a new case, so the metric tightens over time.

## Infrastructure work

Faster embedding batching and incremental graph rebuilds are queued behind
the quality work, since evaluation changes pay off for every later
optimization.

## 🔗 Related Notes
- [[recall-mrr]]
- [[wikilinks]]
- [[mcp-server]]
""",
    "distractors/cooking-fusion.md": """---
tags: [cooking]
---
# Fusion Cooking

Fusion cuisine combines culinary traditions: kimchi tacos, miso carbonara,
and curry pizza all remix techniques across cultures.

## Technique

The trick is respecting both sources: a fusion dish fails when it borrows
the name of a technique but not its discipline.

## 🔗 Related Notes
- [[chunky-knitting]]
- [[backup-generators]]
""",
    "distractors/chess-ranking.md": """---
tags: [chess]
---
# Chess Ranking

Chess players carry a numerical ranking that rises with wins and falls with
losses, administered by a national federation.

## Tournaments

A tournament report sends both players' rankings to the federation, which
recomputes them overnight; a draw moves them toward each other.

## 🔗 Related Notes
- [[recall-election]]
- [[graph-paper]]
""",
    "distractors/chunky-knitting.md": """---
tags: [crafts]
---
# Chunky Knitting

Chunky yarn knits up fast: the thick strand covers ground quickly and
forgives uneven tension.

## Gauge

Patterns specify a gauge swatch so the finished scarf matches the size on
the page, even when the knitter's hands pull tighter or looser.

## 🔗 Related Notes
- [[cooking-fusion]]
- [[embedding-emails]]
""",
    "distractors/embedding-emails.md": """---
tags: [email]
---
# Embedding Media in Emails

Embedding images in an email means attaching the file so the reader sees it
inline instead of fetching it from a remote server later.

## Size limits

Mail hosts cap attachment size, so a large image is compressed before it is
embedded, and the layout is checked in a preview pane.

## 🔗 Related Notes
- [[chunky-knitting]]
- [[mcp-server]]
""",
    "distractors/backup-generators.md": """---
tags: [home]
---
# Backup Generators

A backup generator keeps a home powered through an outage, burning propane
or natural gas behind an automatic transfer switch.

## Sizing

The generator is sized from the circuits that must stay live: a well pump,
a refrigerator, and a furnace, not every outlet in the house.

## 🔗 Related Notes
- [[cooking-fusion]]
- [[backups]]
""",
    "distractors/recall-election.md": """---
tags: [civics]
---
# Recall Elections

A recall election lets voters remove an elected official before the term
ends, by petitioning for a special vote.

## Thresholds

The petition needs signatures from a fraction of the district's registered
voters within a fixed window; if the threshold is met, the ballot asks
whether the official should be removed.

## 🔗 Related Notes
- [[chess-ranking]]
- [[graph-paper]]
""",
    "distractors/graph-paper.md": """---
tags: [stationery]
---
# Graph Paper

Graph paper prints a grid of fine lines for drawing charts, plotting
functions, and laying out cross-stitch patterns.

## Grids

Engineering paper uses a green tint so pencil sketches reproduce cleanly on
a copier, with the grid vanishing from the copy.

## 🔗 Related Notes
- [[chess-ranking]]
- [[recall-election]]
""",
    "distractors/vault-locks.md": """---
tags: [home]
---
# Vault Door Locks

A home vault door carries a combination lock with hardened bolts that seat
into the frame on every side.

## Maintenance

The lock mechanism wants an annual service: dust in the bolt works grinds
the gears, and the combination drifts as parts wear.

## 🔗 Related Notes
- [[backup-generators]]
- [[sync-state]]
""",
    "distractors/mcp-certification.md": """---
tags: [healthcare]
---
# Medical Certification Program

A medical certification program licenses technicians to draw blood and run
lab panels under a physician's oversight.

## Renewal

Certification renews on a two-year cycle with continuing education credits;
lapsed certification bars the technician from signing off results.

## 🔗 Related Notes
- [[recall-election]]
- [[embedding-emails]]
""",
}

GOLDEN_TOML = """
[[cases]]
query = "merge vector and keyword rankings with rank-based fusion"
expected = [{bank = "kb", path = "retrieval/reciprocal-rank-fusion.md"}]

[[cases]]
query = "why fuse dense and sparse retrieval legs"
expected = [{bank = "kb", path = "retrieval/hybrid-search.md"}]

[[cases]]
query = "what determines where one passage ends and the next begins"
expected = [{bank = "kb", path = "indexing/chunking.md"}]

[[cases]]
query = "how is embedding round-trip overhead amortized"
expected = [{bank = "kb", path = "indexing/embeddings.md"}, {bank = "kb", path = "api/ollama-client.md"}]

[[cases]]
query = "after renaming a note why does the indexer not embed it twice"
expected = [{bank = "kb", path = "indexing/sync-state.md"}]

[[cases]]
query = "a comment containing a hash sign must not become a tag"
expected = [{bank = "kb", path = "vault/markdown-parsing.md"}]

[[cases]]
query = "two notes sharing one title must still link unambiguously"
expected = [{bank = "kb", path = "vault/wikilinks.md"}]

[[cases]]
query = "assistants talk to the index over pipes with no port opened"
expected = [{bank = "kb", path = "api/mcp-server.md"}]

[[cases]]
query = "why do note centroids subtract the corpus average"
expected = [{bank = "kb", path = "indexing/embeddings.md"}, {bank = "kb", path = "graph/note-graph.md"}]

[[cases]]
query = "a bad windowing value must fail before syncing starts"
expected = [{bank = "kb", path = "config/toml-config.md"}]

[[cases]]
query = "score retrieval with recall at k and mean reciprocal rank"
expected = [{bank = "kb", path = "eval/recall-mrr.md"}]

[[cases]]
query = "recover the indexes after a disk failure without re-embedding everything"
expected = [{bank = "kb", path = "ops/backups.md"}]
"""


class HashingEmbedder:
    """Deterministic bag-of-words embedder: cheap, offline, discriminating."""

    DIM = 1024

    def __init__(self) -> None:
        self._pattern = re.compile(r"[a-z0-9]+")

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.DIM
        for token in self._pattern.findall(text.lower()):
            # Stable across processes: str hash() is randomized per run.
            slot = int.from_bytes(
                hashlib.blake2b(token.encode(), digest_size=8).digest(), "little"
            )
            vec[slot % self.DIM] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def health(self) -> bool:
        return True


def run(params: dict, workdir: Path) -> dict:
    vault = workdir / "vault"
    vault.mkdir(parents=True)
    for rel, content in VAULT_NOTES.items():
        target = vault / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    config = RagConfig(
        vault_path=vault,
        qdrant_path=workdir / "data" / "qdrant",
        fts_path=workdir / "data" / "fts.sqlite",
        sync_state_path=workdir / "data" / "sync_state.sqlite",
        ollama_url=params.get("ollama_url", "http://127.0.0.1:11434"),
        embedding_model=params.get("embedding_model", "nomic-embed-text"),
        chunk_size=params["chunk_size"],
        chunk_overlap=params["chunk_overlap"],
        embed_context_header=params.get("embed_context_header", True),
        skip_headings=list(params.get("skip_headings", [])),
    )
    # Register the vault as a knowledge-base bank; load_config normally does
    # this from [[banks]] tables, but RagConfig takes parsed BankConfig objects.
    from second_brain.banks import BankConfig

    config.banks = {
        # doc_globs=("*",) mirrors parse_banks for knowledge-base banks; the
        # dataclass default () would make every .md file ineligible.
        "kb": BankConfig(
            id="kb", name="kb", type="knowledge-base", path=vault, doc_globs=("*",)
        ),
    }

    service = RagService(config)
    if params.get("embedder", "hash") == "hash":
        service.embedder = service.indexer.embedder = HashingEmbedder()

    start = time.perf_counter()
    result = service.sync(mode="full")
    sync_seconds = time.perf_counter() - start
    if result["errors"]:
        raise RuntimeError(f"sync errors: {result['errors']}")

    chunk_count = sum(
        len(service.keyword_store.chunks_by_path(key))
        for key in service.sync_state.tracked_paths()
    )

    golden_path = workdir / "golden.toml"
    golden_path.write_text(GOLDEN_TOML, encoding="utf-8")
    cases = load_golden(golden_path)

    report = evaluate(service, cases, ks=(3, 5), depth=10)

    # Query latency: mean over the golden queries, 3 repeats each.
    lat_start = time.perf_counter()
    repeats = 3
    for _ in range(repeats):
        for case in cases:
            service.search(query=case.query, top_k=10, verbose=False)
    query_ms = (time.perf_counter() - lat_start) * 1000 / (len(cases) * repeats)

    service.vector_store.client.close()

    return {
        "params": {
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "embed_context_header": config.embed_context_header,
            "skip_headings": config.skip_headings,
            "embedder": params.get("embedder", "hash"),
        },
        "recall@3": round(report.recall[3], 4),
        "recall@5": round(report.recall[5], 4),
        "mrr": round(report.mrr, 4),
        "chunk_count": chunk_count,
        "sync_seconds": round(sync_seconds, 3),
        "query_ms": round(query_ms, 2),
        "unknown_expected": report_to_dict(report)["summary"]["unknown_expected"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    params = tomllib.loads(Path(args.params).read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="chunk-sweep-") as tmp:
        stats = run(params, Path(tmp))

    print("STATS " + json.dumps(stats))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"METRIC {stats['recall@5']}")


if __name__ == "__main__":
    main()
