# Changelog

## Unreleased

- Add `second-brain eval --golden PATH` to score retrieval with note-level recall@5, recall@10 and MRR over a TOML golden query set (`eval/golden.example.toml`).
- Remove `answer_draft` from `rag.query`; callers receive retrieved chunks and citations without a truncated extractive concatenation.
