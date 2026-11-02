# Changelog

## Unreleased

- Keyword search now ORs query terms after dropping stopwords, so natural-language questions reach the FTS leg, and indexes note titles and headings with title/heading BM25 weights. The FTS table is rebuilt automatically on the next start from existing rows; no re-sync is needed.
- Ignore markdown syntax inside code fences and inline code when parsing headings, tags, links and tasks, and keep newlines and indentation in chunk text. Run `second-brain sync --mode full` once after upgrading to re-chunk unchanged notes.
- Add `second-brain eval --golden PATH` to score retrieval with note-level recall@5, recall@10 and MRR over a TOML golden query set (`eval/golden.example.toml`).
- Remove `answer_draft` from `rag.query`; callers receive retrieved chunks and citations without a truncated extractive concatenation.
