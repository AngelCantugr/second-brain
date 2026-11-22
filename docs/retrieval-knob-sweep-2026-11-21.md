# Retrieval knob sweep — experiment log and analysis

- **Date:** 2026-11-21/22
- **Method:** autoresearch classic metric loop — 25 bounded iterations, 2 fork-and-flush rounds (8 fresh-context branch agents), keep/discard on guard
- **Metric:** MRR (note-level, `summary.mrr` of `second-brain eval`) over `eval/golden.code.toml` (12 cases); direction: higher is better
- **Guard:** full unit suite (989 tests at baseline)
- **Artifacts:** `autoresearch/loop-261121-2320/` — `results.tsv` (full iteration log), `per-case-ranks.txt`, `handoff.json`, `rag_config.eval.toml` (exact eval configuration)
- **Code outcome:** PR #74 (`feat(retrieval): lift implementation files above test/doc near-ties`)

## TL;DR

Baseline MRR **0.7083** → final MRR **1.0** (all 12 golden cases rank the expected file first; recall@10 was already saturated at 1.0, so every gain came from first-relevant rank). The win came from two post-fusion knobs — `repo_aux_weight` (demote repo test/prose chunks) and `filename_boost` (gated filename-stem boost) — discovered by random fork branches after every pre-existing fusion knob proved inert. The 1.0 is robust across wide knob ranges and deterministic across runs.

## Setup

This repository indexed as a repo bank (`second-brain`), 85 files, nomic-embed-text via local Ollama, chunk_size 500 / overlap 80, RRF fusion of semantic + keyword legs. Verify command:

```
uv run second-brain eval --golden eval/golden.code.toml | jq .summary.mrr
```

## Timeline

**Phase 1 — knob sweeps, all flat (iters 1–7).** Iteration 1 landed the enabler diff (from the experiment brief): `recency_boost` threaded through `evaluate()` + a `--recency-boost` CLI flag, and the hardcoded RRF `k=60` lifted to the `rrf_k` config key. Behavior-neutral, kept. Then the sweep grid: `graph_boost` ∈ {0.2, 0.5, 1.0}, `code_source_weight` 0.5, `rrf_k` ∈ {20, 120} — MRR 0.7083 every time. Six consecutive discards.

**Diagnosis (between phases).** Per-case ranks showed the real failure mode is not what the knobs address:

```
rank=2  expected=second_brain/redaction.py     top=tests/test_redaction.py
rank=3  expected=second_brain/embedder.py      top=docs/capability-review-2026-11.md
rank=2  expected=second_brain/parser.py        top=docs/capability-review-2026-11.md
rank=3  expected=second_brain/canvas.py        top=second_brain/parser.py
rank=3  expected=second_brain/keyword_store.py top=tests/test_keyword_store.py
rank=2  expected=second_brain/code.py          top=tests/test_code_symbols.py
```

Exact decomposition: 6×rank-1, 3×rank-2, 3×rank-3 → MRR = (6 + 1.5 + 1)/12 = 0.7083. Golden queries paraphrase *why* code exists, so test files and prose docs — which describe behavior in plain words — consistently edge out implementation outlines by one rank. Every fusion-layer knob re-weights signals shared by both competitors and cannot separate them.

**Phase 2 — fork-and-flush round 1 (iter 8).** Four branches × 2 iterations from the incumbent. Round-1 standings:

| Branch | Kind | Best MRR | Mechanism |
|---|---|---|---|
| fork-4 | random | **1.0** | `repo_aux_weight=0.5` (0.7083→0.9583) then gated `filename_boost=0.5` (→1.0) |
| fork-3 | random | 0.9583 | path-class demotion of `tests/`+`docs/` chunks (0.7) |
| fork-1 | guided | 0.9167 | bounded RRF bonus for documented-symbol chunks |
| fork-2 | guided | 0.7083 | chunk_size 120/200 — monotonically worse (0.575 / 0.6903) |

Three of four branches independently attacked the same tests/docs-vs-source competition — strong convergence on the diagnosis. Winner cherry-picked; new incumbent 1.0.

**Phase 3 — basin mapping (iters 9–15).** With the ceiling hit, remaining iterations mapped robustness: 1.0 holds for `repo_aux_weight` 0.3–0.7, `filename_boost` 0.3–0.7, `rrf_k` 10/40, `graph_boost` 0.5, `recency_boost` 0.3. Fusion-layer knobs remain inert under the new mechanism.

**Phase 4 — fork-and-flush round 2 (iter 16).** At the ceiling, ties go to branches; goal shifted to simplification and robustness. Round-2 standings:

| Branch | Kind | Result |
|---|---|---|
| fork-8 | random | **tie 1.0, selected** — ablated all 4 gate variants; the `kw_ratio<0.8` guard is redundant with `sem_gap ≤ 0.01`; kept the simpler single-gate mechanism |
| fork-5 | guided | tie 1.0 (alternate simplification), plus: `filename_boost` alone = 0.8194 |
| fork-6 | guided | tie 1.0 by adding the content-grounded symbol bonus (three mechanisms — rejected as more moving parts) |
| fork-7 | random | tie 1.0 at the untested joint corner (0.3, 0.7) |

Round-2 negative results are as valuable as the win: `filename_boost` alone collapses to 0.8194, so `repo_aux_weight` is load-bearing; the pair cannot be reduced to one knob.

**Phase 5 — endgame probes (iters 17–24).** `rrf_k=10` still 1.0; `embed_context_header=false` drops to 0.9583 (per-file header carries ranking signal; index restored); `code_source_weight` still inert; interior corner (0.4, 0.6) holds; **index-time alternative** — excluding `tests/**` and `docs/**` from the corpus — also reaches 1.0 but on a 39-file corpus; rank-time demotion preferred because auxiliary content stays retrievable. Three consecutive eval runs are bit-identical (deterministic). Final guard: 989 tests green.

## Findings

1. The golden-set failure mode is **corpus competition**, not fusion weighting: test/doc chunks share the paraphrase vocabulary of "why does this code exist" queries.
2. `graph_boost`, `code_source_weight`, `rrf_k` (10–120), and `recency_boost` are **inert** on this set, with or without the new knobs — 10 probes, zero movement.
3. `repo_aux_weight=0.5` is the single large win (+0.25); `filename_boost=0.5` fixes the one case the demotion leaves at rank 2 (+0.0417). The gate must be the semantic near-tie condition alone — the `kw_ratio` guard adds nothing (all 4 variants ablated in round 2).
4. Both chunk-side reductions hurt: smaller `chunk_size` destroys outline signal; `embed_context_header=false` drops one case.
5. The 1.0 basin is wide (both knobs independently 0.3–0.7, joint corners hold) and deterministic — the result is not a knife-edge tuning.

## Interpretation

- **Why the classic knobs failed:** every swept knob re-weights *signals*, but the errors were *candidate-class* errors — the right file and the wrong file both score well on every leg. Only a mechanism that distinguishes implementation chunks from auxiliary chunks (path class, or content like documented symbols) can separate them.
- **Random vs guided forks:** both rounds were won by random branches (2–0). Guided branches converged on the same diagnosis and produced viable-but-second mechanisms; the winning moves came from branches that measured fused-score gaps directly instead of reasoning from the Advisor's theory. Small sample — but consistent with the fork-and-flush literature.
- **Overfitting caveats:** 12 cases, one embedding model, one repository. `filename_boost`'s `sem_gap ≤ 0.01` threshold and the 0.5 weights are tuned constants; the golden set's queries all expect `.py` implementation files, so the aux demotion is validated only for "implementation is the answer" corpora. On a docs-first repo the demotion could hurt — that failure mode is invisible here.
- **Index-time vs rank-time:** excluding tests/docs at sync time is a legitimate simpler alternative (also 1.0) at the cost of shrinking the corpus 85→39 files. Rank-time demotion keeps auxiliary content retrievable and is reversible without re-embedding; preferred for a general-purpose tool.

## Reproduction

```
# from a checkout at the experiment branch (second-brain-experiment tip)
cp autoresearch/loop-261121-2320/rag_config.eval.toml rag_config.toml
uv sync && uv run second-brain sync --mode full
uv run second-brain eval --golden eval/golden.code.toml   # MRR 1.0
uv run pytest tests/ -q                                    # 989 passed
```

The full iteration-by-iteration record (every keep/discard, commit SHA, guard result) is in `autoresearch/loop-261121-2320/results.tsv`; machine-readable summary in `handoff.json`.

## Limitations

- Golden set is synthetic and written against this repository; queries deliberately avoid naming symbols.
- Eval requires local Ollama with nomic-embed-text; results may transfer differently across embedding models.
- The loop optimized a single scalar on 12 cases; per-case confidence intervals are meaningless at this granularity. Treat the mechanism as validated directionally, the constants as starting points.
