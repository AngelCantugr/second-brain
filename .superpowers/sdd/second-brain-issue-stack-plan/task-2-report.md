# Task 2 report: Issue #31 filtered retrieval

## Status
DONE

## Implementation
- Semantic search now applies the existing `matches_filters` predicate while collecting ranked candidates, before the requested limit is applied. In-memory results filter the complete sorted result set; local Qdrant results are queried in score-ordered pages until the filtered limit is filled or the collection is exhausted.
- SQLite FTS searches with filters read the complete ranked FTS candidate set before applying metadata predicates and the requested limit. Filtered wildcard listing similarly scans all indexed chunks. Unfiltered query behavior retains bounded retrieval.
- Native Qdrant filtering was not used because the supported filters include arbitrary frontmatter semantics. The complete pagination fallback preserves the Python reference predicate.

## TDD evidence
- RED: `.venv/bin/python -m pytest tests/test_service.py::test_search_applies_filter_before_semantic_candidate_limit tests/test_keyword_store.py::test_keyword_search_filters_before_candidate_limit -q` — both focused regressions failed as expected with no matching target returned beyond the old candidate windows.
- GREEN: `.venv/bin/python -m pytest tests/test_service.py::test_search_applies_filter_before_semantic_candidate_limit tests/test_keyword_store.py::test_keyword_search_filters_before_candidate_limit tests/test_vector_store.py::test_qdrant_search_pages_until_filtered_limit_is_filled tests/test_vector_store.py::test_qdrant_local_search_matches_filters_after_old_candidate_window -q` — `4 passed`.
- Full suite: `.venv/bin/python -m pytest -q` — `122 passed in 0.88s`.
- `git diff --check` — passed.

## Files changed
- `second_brain/service.py`
- `second_brain/vector_store.py`
- `second_brain/keyword_store.py`
- `tests/test_service.py`
- `tests/test_vector_store.py`
- `tests/test_keyword_store.py`
- `.superpowers/sdd/second-brain-issue-stack-plan/task-2-report.md`

## Self-review and concerns
- Tests cover semantic filtering beyond the old candidate window in the service, paginated fake Qdrant behavior, real local Qdrant pagination, and SQLite FTS filtering beyond twelve earlier unfiltered matches.
- Selective filters can require Qdrant to page through many high-ranked results. Filtered FTS queries materialize every ranked FTS candidate before Python filtering. Both are unbounded by an arbitrary ceiling to preserve parity and correctness; large collections with selective predicates may incur additional latency and memory use.
- Existing obsolete-chunk deletion changes in the base branch were preserved.

## Review round 1 fixes
- Added parameterized local-Qdrant parity comparisons that compute expected ranked IDs by applying `matches_filters` to unfiltered ranked results, then compare the paginated predicate search. Cases cover case-insensitive tag strings and lists, inclusive date boundaries and date-field priority, path prefixes, arbitrary frontmatter key/value checks, and derived-field equality.
- Added local-Qdrant exhaustion coverage for one result below the requested limit and no matching results. The parity cases also require two matches after the first page, covering a multi-hit result limit across pages.
- Production code was unchanged; review findings were test coverage gaps only.
- Validation: `.venv/bin/python -m pytest tests/test_vector_store.py -q` — `13 passed in 0.48s`.
- Validation: `git diff --check` — passed.
