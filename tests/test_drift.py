"""Drift between notes and the code they mention: state across syncs, and rag.drift."""

from pathlib import Path

import pytest

from second_brain.code_graph import MentionTarget, _CodeIndex, typed_graph
from second_brain.drift import (
    DANGLING,
    FILE_REMOVED,
    OK,
    STALE,
    SYMBOL_REMOVED,
    UNRESOLVED,
    MentionState,
    mention_level,
    mention_states,
    rename_candidates,
)

# -- mention -> symbol ----------------------------------------------------------------


def _index(*metas: dict) -> _CodeIndex:
    return _CodeIndex.build(list(metas))


def test_a_mention_names_a_symbol_only_when_it_spells_one() -> None:
    index = _index(
        {"path": "repo:sb/g.py", "code": {"symbols": ["GraphBuilder", "GraphBuilder.rebuild_full", "Store.rebuild_full"]}}
    )

    assert index.symbol_in("repo:sb/g.py", "GraphBuilder.rebuild_full()") == "GraphBuilder.rebuild_full"
    assert index.symbol_in("repo:sb/g.py", "graph.GraphBuilder") == "GraphBuilder"
    # Two methods share the terminal name, so the bare name picks neither.
    assert index.symbol_in("repo:sb/g.py", "rebuild_full") is None
    # Paths and URLs name the file.
    assert index.symbol_in("repo:sb/g.py", "second_brain/g.py") is None


def test_typed_graph_reports_what_each_mention_resolved_to() -> None:
    metas = [
        {"path": "repo:sb/g.py", "code": {"symbols": ["GraphBuilder"]}},
        {"path": "Note.md", "code": {"mentions": ["GraphBuilder", "g.py", "nothing_here"]}},
    ]

    _, resolved = typed_graph(metas, 0.25)

    assert resolved == {
        ("Note.md", "GraphBuilder"): MentionTarget("repo:sb/g.py", "GraphBuilder"),
        ("Note.md", "g.py"): MentionTarget("repo:sb/g.py", None),
    }


# -- state machine ----------------------------------------------------------------------


def _code(path: str, symbols: dict[str, str], file_hash: str, mtime: float = 100.0) -> dict:
    return {
        "path": path,
        "code": {"symbols": sorted(symbols), "symbol_hashes": symbols, "hash": file_hash, "mtime": mtime},
    }


def _note(path: str, mentions: list[str], note_hash: str) -> dict:
    return {"path": path, "code": {"mentions": mentions, "hash": note_hash, "mtime": 50.0}}


def _sync(metas: list[dict], previous: dict | None = None, now: float = 1000.0) -> dict:
    _, resolved = typed_graph(metas, 0.25)
    states = mention_states(metas, resolved, previous or {}, now)
    return {(s.src, s.mention): s for s in states}


def test_first_sync_records_a_baseline_with_nothing_stale() -> None:
    metas = [_code("repo:sb/r.py", {"compute_backoff": "h1"}, "f1"), _note("N.md", ["compute_backoff"], "n1")]

    state = _sync(metas)[("N.md", "compute_backoff")]

    assert (state.status, state.acked_hash, state.current_hash, state.changed_at) == (OK, "h1", "h1", None)
    assert (state.target, state.symbol) == ("repo:sb/r.py", "compute_backoff")


def test_code_changing_under_an_unchanged_note_is_stale_until_the_note_changes() -> None:
    first = _sync([_code("repo:sb/r.py", {"compute_backoff": "h1"}, "f1"), _note("N.md", ["compute_backoff"], "n1")])

    changed = [_code("repo:sb/r.py", {"compute_backoff": "h2"}, "f2", mtime=300.0), _note("N.md", ["compute_backoff"], "n1")]
    second = _sync(changed, first)
    state = second[("N.md", "compute_backoff")]
    assert (state.status, state.acked_hash, state.current_hash, state.changed_at) == (STALE, "h1", "h2", 300.0)

    # A full sync reprocesses the note without changing it: still stale.
    assert _sync(changed, second)[("N.md", "compute_backoff")].status == STALE

    edited = [changed[0], _note("N.md", ["compute_backoff"], "n2")]
    state = _sync(edited, second)[("N.md", "compute_backoff")]
    assert (state.status, state.acked_hash, state.changed_at) == (OK, "h2", 300.0)


def test_symbol_mentions_ignore_changes_elsewhere_in_the_file_but_file_mentions_do_not() -> None:
    before = [
        _code("repo:sb/r.py", {"compute_backoff": "h1", "jitter": "j1"}, "f1"),
        _note("N.md", ["compute_backoff", "r.py"], "n1"),
    ]
    after = [_code("repo:sb/r.py", {"compute_backoff": "h1", "jitter": "j2"}, "f2"), before[1]]

    states = _sync(after, _sync(before))

    assert states[("N.md", "compute_backoff")].status == OK
    assert states[("N.md", "r.py")].status == STALE


def test_a_lost_mention_dangles_from_the_sync_that_lost_it_until_the_note_drops_it() -> None:
    note = _note("N.md", ["compute_backoff"], "n1")
    first = _sync([_code("repo:sb/r.py", {"compute_backoff": "h1"}, "f1"), note])

    renamed = [_code("repo:sb/r.py", {"compute_backoff_ms": "h2"}, "f2"), note]
    second = _sync(renamed, first, now=2000.0)
    state = second[("N.md", "compute_backoff")]
    assert (state.status, state.reason, state.lost_at) == (DANGLING, SYMBOL_REMOVED, 2000.0)
    assert (state.target, state.symbol, state.current_hash) == ("repo:sb/r.py", "compute_backoff", None)
    assert rename_candidates(state, renamed) == ["repo:sb/r.py::compute_backoff_ms"]

    # Still dangling a sync later, dated from the first loss.
    assert _sync(renamed, second, now=3000.0)[("N.md", "compute_backoff")].lost_at == 2000.0

    fixed = [renamed[0], _note("N.md", ["compute_backoff_ms"], "n2")]
    assert set(_sync(fixed, second)) == {("N.md", "compute_backoff_ms")}


def test_dangling_reasons_distinguish_removed_files_from_ambiguous_names() -> None:
    note = _note("N.md", ["GraphBuilder", "util/helpers.py"], "n1")
    first = _sync(
        [
            _code("repo:sb/g.py", {"GraphBuilder": "g1"}, "f1"),
            _code("repo:sb/util/helpers.py", {}, "u1"),
            note,
        ]
    )

    later = [
        _code("repo:sb/g.py", {"GraphBuilder": "g1"}, "f1"),
        _code("repo:sb/legacy/g.py", {"GraphBuilder": "g9"}, "f9"),
        _code("repo:sb/lib/helpers.py", {}, "u2"),
        note,
    ]
    states = _sync(later, first)

    ambiguous = states[("N.md", "GraphBuilder")]
    assert (ambiguous.status, ambiguous.reason) == (DANGLING, UNRESOLVED)
    assert rename_candidates(ambiguous, later) == ["repo:sb/legacy/g.py::GraphBuilder"]
    removed = states[("N.md", "util/helpers.py")]
    assert (removed.status, removed.reason) == (DANGLING, FILE_REMOVED)
    assert rename_candidates(removed, later) == ["repo:sb/lib/helpers.py"]


def test_rows_indexed_before_hashes_existed_are_skipped_then_baselined() -> None:
    legacy_note = {"path": "N.md", "code": {"mentions": ["compute_backoff"]}}
    legacy_code = {"path": "repo:sb/r.py", "code": {"symbols": ["compute_backoff"]}}

    assert _sync([legacy_code, legacy_note]) == {}

    # The note is indexed again but the code file is not: tracked, hash unknown.
    first = _sync([legacy_code, _note("N.md", ["compute_backoff"], "n1")])
    assert first[("N.md", "compute_backoff")].current_hash is None
    # Once the code file is indexed, its hash is the baseline, not a change.
    state = _sync([_code("repo:sb/r.py", {"compute_backoff": "h1"}, "f1"), _note("N.md", ["compute_backoff"], "n1")], first)[
        ("N.md", "compute_backoff")
    ]
    assert (state.status, state.acked_hash) == (OK, "h1")


@pytest.mark.parametrize(
    ("mention", "symbol", "expected"),
    [
        ("compute_backoff", "compute_backoff", "symbol"),
        ("pkg/retry.py", None, "file"),
        ("https://github.com/me/sb/blob/main/pkg/retry.py", None, "file"),
        ("sb", None, "repo"),
        ("https://github.com/me/sb.", None, "repo"),
    ],
)
def test_mentions_are_levelled_by_what_they_name(mention: str, symbol: str | None, expected: str) -> None:
    state = MentionState("N.md", mention, "repo:sb/README.md", symbol, "n1", None, None, None, None, OK)

    assert mention_level(state) == expected


# -- fingerprints ----------------------------------------------------------------------

tree_sitter = pytest.importorskip("tree_sitter")

from second_brain.code import load_code_file  # noqa: E402
from second_brain.config import CodeSource, RagConfig  # noqa: E402
from second_brain.service import RagService  # noqa: E402


def _hashes(tmp_path: Path, source: str) -> dict[str, str]:
    path = tmp_path / "m.py"
    path.write_text(source, encoding="utf-8")
    return {s.qualname: s.body_hash for s in load_code_file(path, "m.py", "sb", "repo:sb/m.py").symbols}


def test_body_hashes_follow_the_code_not_its_position(tmp_path: Path) -> None:
    base = _hashes(tmp_path, 'class A:\n    def f(self):\n        """Doc."""\n        return 1\n\n\ndef g():\n    return 2\n')
    shifted = _hashes(tmp_path, '"""Module."""\n\nX = 1\n\n\nclass A:\n    def f(self):\n        """Doc."""\n        return 1   \n\n\ndef g():\n    return 2\n')
    edited = _hashes(tmp_path, 'class A:\n    def f(self):\n        """Doc."""\n        return 10\n\n\ndef g():\n    return 2\n')

    assert shifted["A.f"] == base["A.f"] and shifted["g"] == base["g"]
    assert edited["A.f"] != base["A.f"] and edited["A"] != base["A"]
    assert edited["g"] == base["g"]


# -- rag.drift end to end ---------------------------------------------------------------


class _StubEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0, 0.5] for text in texts]

    def health(self) -> bool:
        return True


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


_RETRY = 'def compute_backoff(n):\n    """Delay before retry n."""\n    return min(2 ** n, 30)\n\n\ndef jitter():\n    return 0\n'


def _service(tmp_path: Path) -> tuple[RagService, Path, Path]:
    vault, repo = tmp_path / "vault", tmp_path / "repo"
    _write(vault / "Retries.md", "# Retries\nI capped `compute_backoff` at 30s.\n\n# Clock\nTime comes from `pkg/clock.py`.\n")
    _write(repo / "pkg" / "retry.py", _RETRY)
    _write(repo / "pkg" / "clock.py", 'def now():\n    """Current time."""\n')
    config = RagConfig(
        vault_path=vault,
        qdrant_path=tmp_path / "q",
        fts_path=tmp_path / "fts.sqlite",
        sync_state_path=tmp_path / "s.sqlite",
        code_sources=[CodeSource("sb", repo, languages=("python",))],
    )
    service = RagService(config, use_in_memory_vector=True)
    service.embedder = service.indexer.embedder = _StubEmbedder()
    return service, vault, repo


def test_rag_drift_returns_stale_pairs_with_both_texts(tmp_path: Path) -> None:
    service, vault, repo = _service(tmp_path)
    assert service.sync(mode="full")["errors"] == []
    clean = service.drift()
    assert (clean["counts"], clean["pairs"]) == ({OK: 2, STALE: 0, DANGLING: 0}, [])

    _write(repo / "pkg" / "retry.py", _RETRY.replace("30)", "60)"))
    service.sync(mode="incremental")
    result = service.drift()

    assert result["counts"] == {OK: 1, STALE: 1, DANGLING: 0}
    [pair] = result["pairs"]
    assert pair["status"] == STALE
    assert pair["note"]["path"] == "Retries.md"
    assert pair["note"]["excerpts"] == ["I capped `compute_backoff` at 30s."]
    code = pair["code"]
    assert (code["path"], code["symbol"], code["start_line"], code["end_line"]) == (
        "repo:sb/pkg/retry.py", "compute_backoff", 1, 3,
    )
    assert "Delay before retry n." in code["text"]
    assert code["changed_at"] == pytest.approx((repo / "pkg" / "retry.py").stat().st_mtime)
    assert code["local_path"] == str(repo / "pkg" / "retry.py")
    assert pair["evidence"] == {"mention": "compute_backoff", "level": "symbol"}

    # A full sync does not count as the note being rewritten.
    service.sync(mode="full")
    assert service.drift()["counts"][STALE] == 1
    _write(vault / "Retries.md", "# Retries\nI capped `compute_backoff` at 60s.\n\n# Clock\nTime comes from `pkg/clock.py`.\n")
    service.sync(mode="incremental")
    assert service.drift()["counts"] == {OK: 2, STALE: 0, DANGLING: 0}


def test_rag_drift_remembers_a_renamed_symbol_and_suggests_the_new_name(tmp_path: Path) -> None:
    service, _, repo = _service(tmp_path)
    service.sync(mode="full")

    _write(repo / "pkg" / "retry.py", _RETRY.replace("compute_backoff", "compute_backoff_ms"))
    service.sync(mode="incremental")
    assert service.graph_store.edge_between("Retries.md", "repo:sb/pkg/retry.py") is None

    [pair] = service.drift(status=[DANGLING])["pairs"]
    assert pair["code"]["exists"] is True
    assert pair["evidence"]["reason"] == SYMBOL_REMOVED
    assert pair["evidence"]["candidates"] == ["repo:sb/pkg/retry.py::compute_backoff_ms"]
    assert pair["evidence"]["lost_at"] is not None


def test_rag_drift_filters_and_validates(tmp_path: Path) -> None:
    service, _, _ = _service(tmp_path)
    service.sync(mode="full")

    assert [p["evidence"]["level"] for p in service.drift(status=[OK])["pairs"]] == ["symbol", "file"]
    assert len(service.drift(status=[OK], levels=["file"])["pairs"]) == 1
    assert service.drift(status=[OK], repo="other")["pairs"] == []
    assert len(service.drift(status=[OK], note_path="Retries.md", limit=1)["pairs"]) == 1
    with pytest.raises(ValueError, match="status"):
        service.drift(status=["broken"])
    with pytest.raises(ValueError, match="levels"):
        service.drift(levels=["module"])
    with pytest.raises(ValueError, match="limit"):
        service.drift(limit=0)
