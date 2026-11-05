"""Drift between notes and the code they mention: state across syncs, and rag.drift."""

from pathlib import Path

import pytest

from second_brain.code_graph import MentionTarget, _CodeIndex, is_one_word_name, typed_graph
from second_brain.config import load_config
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
    record_summaries,
    rename_candidates,
)
from second_brain.doc_roles import (
    DEFAULT_DOC_ROLES,
    HISTORY,
    RECORD,
    REFERENCE,
    doc_role,
    validated_doc_roles,
)

# -- mention -> symbol ----------------------------------------------------------------


def _index(*metas: dict) -> _CodeIndex:
    return _CodeIndex.build(list(metas))


def _symbol(index: _CodeIndex, mention: str) -> str | None:
    [target] = index.resolve_mention_targets(mention, None)
    return target.symbol


def test_a_mention_names_a_symbol_only_when_it_spells_one() -> None:
    index = _index(
        {"path": "repo:sb/g.py", "code": {"symbols": ["GraphBuilder", "GraphBuilder.rebuild_full", "Store.rebuild_full"]}}
    )

    assert _symbol(index, "GraphBuilder.rebuild_full()") == "GraphBuilder.rebuild_full"
    assert _symbol(index, "graph.GraphBuilder") == "GraphBuilder"
    # Two methods share the terminal name, so the bare name picks neither.
    assert _symbol(index, "rebuild_full") is None
    # Paths and URLs name the file.
    assert _symbol(index, "sb/g.py") is None


def test_a_module_mention_names_its_file_even_when_a_symbol_shares_the_name() -> None:
    index = _index(
        {"path": "repo:sb/pkg/compute_backoff.py", "code": {"symbols": ["Retry.compute_backoff", "compute_backoff"]}},
    )

    # Resolved as the module pkg/compute_backoff.py, so every symbol counts.
    assert index.resolve_mention_targets("pkg.compute_backoff", None) == [
        MentionTarget("repo:sb/pkg/compute_backoff.py", None)
    ]


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


def _fp(digest: str, path: str = "repo:sb/r.py", symbol: str = "compute_backoff") -> str:
    return f"{digest}@{path}::{symbol}"


def _sync(metas: list[dict], previous: dict | None = None, now: float = 1000.0) -> dict:
    # As GraphBuilder.typed does: last sync's targets steer ambiguous names.
    previous = previous or {}
    _, resolved = typed_graph(metas, 0.25, previous={key: s.target for key, s in previous.items()})
    states = mention_states(metas, resolved, previous, now)
    return {(s.src, s.mention): s for s in states}


def test_first_sync_records_a_baseline_with_nothing_stale() -> None:
    metas = [_code("repo:sb/r.py", {"compute_backoff": "h1"}, "f1"), _note("N.md", ["compute_backoff"], "n1")]

    state = _sync(metas)[("N.md", "compute_backoff")]

    assert (state.status, state.acked_hash, state.current_hash, state.changed_at) == (OK, _fp("h1"), _fp("h1"), None)
    assert (state.target, state.symbol) == ("repo:sb/r.py", "compute_backoff")


def test_code_changing_under_an_unchanged_note_is_stale_until_the_note_changes() -> None:
    first = _sync([_code("repo:sb/r.py", {"compute_backoff": "h1"}, "f1"), _note("N.md", ["compute_backoff"], "n1")])

    changed = [_code("repo:sb/r.py", {"compute_backoff": "h2"}, "f2", mtime=300.0), _note("N.md", ["compute_backoff"], "n1")]
    second = _sync(changed, first)
    state = second[("N.md", "compute_backoff")]
    assert (state.status, state.acked_hash, state.current_hash, state.changed_at) == (
        STALE, _fp("h1"), _fp("h2"), 300.0,
    )

    # A full sync reprocesses the note without changing it: still stale.
    assert _sync(changed, second)[("N.md", "compute_backoff")].status == STALE

    edited = [changed[0], _note("N.md", ["compute_backoff"], "n2")]
    state = _sync(edited, second)[("N.md", "compute_backoff")]
    assert (state.status, state.acked_hash, state.changed_at) == (OK, _fp("h2"), 300.0)


def test_symbol_mentions_ignore_changes_elsewhere_in_the_file_but_file_mentions_do_not() -> None:
    before = [
        _code("repo:sb/r.py", {"compute_backoff": "h1", "jitter": "j1"}, "f1"),
        _note("N.md", ["compute_backoff", "r.py"], "n1"),
    ]
    after = [_code("repo:sb/r.py", {"compute_backoff": "h1", "jitter": "j2"}, "f2"), before[1]]

    states = _sync(after, _sync(before))

    assert states[("N.md", "compute_backoff")].status == OK
    assert states[("N.md", "r.py")].status == STALE


def test_a_symbol_moved_unchanged_to_another_file_is_stale_until_the_note_changes() -> None:
    note = _note("N.md", ["compute_backoff"], "n1")
    first = _sync([_code("repo:sb/a.py", {"compute_backoff": "h1"}, "fa"), note])

    moved = [_code("repo:sb/a.py", {}, "fa2"), _code("repo:sb/b.py", {"compute_backoff": "h1"}, "fb", mtime=400.0), note]
    second = _sync(moved, first)
    state = second[("N.md", "compute_backoff")]
    assert (state.status, state.target, state.changed_at) == (STALE, "repo:sb/b.py", 400.0)
    assert state.acked_hash == _fp("h1", "repo:sb/a.py")
    # Still stale on the next sync: the acknowledged baseline remembers a.py.
    assert _sync(moved, second)[("N.md", "compute_backoff")].status == STALE

    edited = [*moved[:2], _note("N.md", ["compute_backoff"], "n2")]
    assert _sync(edited, second)[("N.md", "compute_backoff")].status == OK


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


def test_a_name_that_becomes_ambiguous_keeps_the_file_it_named() -> None:
    note = _note("N.md", ["GraphBuilder", "helpers.py"], "n1")
    first = _sync(
        [
            _code("repo:sb/g.py", {"GraphBuilder": "g1"}, "f1"),
            _code("repo:sb/util/helpers.py", {}, "u1"),
            note,
        ]
    )

    # Another file starts defining the same names; the note meant the first.
    later = [
        _code("repo:sb/g.py", {"GraphBuilder": "g1"}, "f1"),
        _code("repo:sb/legacy/g.py", {"GraphBuilder": "g9"}, "f9"),
        _code("repo:sb/util/helpers.py", {}, "u1"),
        _code("repo:sb/lib/helpers.py", {}, "u2"),
        note,
    ]
    states = _sync(later, first)

    assert (states[("N.md", "GraphBuilder")].target, states[("N.md", "GraphBuilder")].status) == ("repo:sb/g.py", OK)
    assert (states[("N.md", "helpers.py")].target, states[("N.md", "helpers.py")].status) == ("repo:sb/util/helpers.py", OK)
    # Without a previous target the same names stay ambiguous.
    assert not {key for key in _sync(later) if key[1] in ("GraphBuilder", "helpers.py")}


def test_an_ambiguous_name_whose_file_lost_it_dangles_with_the_files_that_define_it() -> None:
    note = _note("N.md", ["GraphBuilder"], "n1")
    first = _sync([_code("repo:sb/g.py", {"GraphBuilder": "g1"}, "f1"), note])

    later = [
        _code("repo:sb/g.py", {}, "f2"),
        _code("repo:sb/a.py", {"GraphBuilder": "a1"}, "a1"),
        _code("repo:sb/b.py", {"GraphBuilder": "b1"}, "b1"),
        note,
    ]
    state = _sync(later, first)[("N.md", "GraphBuilder")]

    assert (state.status, state.reason) == (DANGLING, SYMBOL_REMOVED)
    assert rename_candidates(state, later) == ["repo:sb/a.py::GraphBuilder", "repo:sb/b.py::GraphBuilder"]


def test_a_mention_that_stops_resolving_while_its_code_remains_is_unresolved() -> None:
    metas = [_code("repo:sb/g.py", {"GraphBuilder": "g1"}, "f1"), _note("N.md", ["GraphBuilder"], "n1")]
    first = _sync(metas)

    # No resolution this sync (say, out of scope), but g.py still defines it.
    [state] = mention_states(metas, {}, first, 2000.0)

    assert (state.status, state.reason) == (DANGLING, UNRESOLVED)


def test_dangling_reasons_distinguish_removed_files() -> None:
    note = _note("N.md", ["util/helpers.py"], "n1")
    first = _sync([_code("repo:sb/util/helpers.py", {}, "u1"), note])

    later = [_code("repo:sb/lib/helpers.py", {}, "u2"), note]
    states = _sync(later, first)

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
    assert (state.status, state.acked_hash) == (OK, _fp("h1"))


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
    shifted = _hashes(tmp_path, '"""Module."""\n\nX = 1\n\n\nclass A:\n    def f(self):\n        """Doc."""\n        return 1\n\n\ndef g():\n    return 2\n')
    nested = _hashes(tmp_path, 'if True:\n    def g():\n        return 2\n')
    edited = _hashes(tmp_path, 'class A:\n    def f(self):\n        """Doc."""\n        return 10\n\n\ndef g():\n    return 2\n')

    assert shifted["A.f"] == base["A.f"] and shifted["g"] == base["g"]
    assert nested["g"] == base["g"]
    assert edited["A.f"] != base["A.f"] and edited["A"] != base["A"]
    assert edited["g"] == base["g"]


def test_body_hashes_keep_whitespace_inside_string_literals(tmp_path: Path) -> None:
    padded = _hashes(tmp_path, 'def greet():\n    return """hello \nworld"""\n')
    trimmed = _hashes(tmp_path, 'def greet():\n    return """hello\nworld"""\n')
    dedented = _hashes(tmp_path, 'def greet():\n    return """hello\n  world"""\n')

    assert padded["greet"] != trimmed["greet"]
    assert dedented["greet"] != trimmed["greet"]


# -- one-word names in vault notes --------------------------------------------------------


@pytest.mark.parametrize(
    ("mention", "expected"),
    [
        ("Task", True),
        ("base.py", True),
        ("util.go:12", True),
        ("Loader()", True),
        ("GraphBuilder", False),
        ("HTTPServer", False),
        ("compute_backoff", False),
        ("stack-ledger.py", False),
        ("pkg/base.py", False),
        ("graph.Edge", False),
        ("https://github.com/me/sb", False),
    ],
)
def test_one_word_names_are_bare_and_single_word(mention: str, expected: bool) -> None:
    assert is_one_word_name(mention) is expected


def _plan_code() -> dict:
    return _code("repo:market/plugins/desk/plan_state.py", {"Task": "t1", "load_plan": "l1"}, "p1")


def test_a_one_word_name_in_a_vault_note_needs_the_repo_named_elsewhere() -> None:
    alone = _note("Go.md", ["Task"], "n1")
    with_repo = _note("Desk.md", ["Task", "market"], "n2")
    with_long_name = _note("Plan.md", ["Task", "load_plan"], "n3")
    readme = _code("repo:market/README.md", {}, "r1")

    edges, resolved = typed_graph([_plan_code(), readme, alone, with_repo, with_long_name], 0.25)

    assert resolved[("Go.md", "Task")] is None
    assert not any("Go.md" in (edge.src, edge.dst) for edge in edges)
    assert resolved[("Desk.md", "Task")] == MentionTarget("repo:market/plugins/desk/plan_state.py", "Task")
    assert resolved[("Plan.md", "Task")] == MentionTarget("repo:market/plugins/desk/plan_state.py", "Task")


@pytest.mark.parametrize("named", ["market", "https://github.com/me/market", "https://github.com/me/market."])
def test_naming_the_repo_counts_as_context_without_an_indexed_readme(named: str) -> None:
    # No README is indexed for "market", so the repo mention itself links nothing.
    note = _note("Desk.md", ["Task", named], "n1")

    edges, resolved = typed_graph([_plan_code(), note], 0.25)

    assert resolved[("Desk.md", "Task")] == MentionTarget("repo:market/plugins/desk/plan_state.py", "Task")
    assert ("Desk.md", named) not in resolved
    assert [(edge.src, edge.dst) for edge in edges] == [("Desk.md", "repo:market/plugins/desk/plan_state.py")]


def test_repo_docs_resolve_one_word_names_in_their_own_repo() -> None:
    doc = {"path": "repo:market/plugins/desk/SKILL.md", "code": {"mentions": ["Task"], "hash": "d1"}}

    _, resolved = typed_graph([_plan_code(), doc], 0.25)

    assert resolved[(doc["path"], "Task")] == MentionTarget("repo:market/plugins/desk/plan_state.py", "Task")


def test_a_one_word_name_no_longer_believed_drops_its_row_instead_of_dangling() -> None:
    code = _plan_code()
    # Recorded by an earlier release, which linked one-word names on their own.
    earlier = MentionState(
        src="Go.md",
        mention="Task",
        target=code["path"],
        symbol="Task",
        src_hash="n1",
        acked_hash=_fp("t1", code["path"], "Task"),
        current_hash=_fp("t1", code["path"], "Task"),
        changed_at=None,
        lost_at=None,
        status=OK,
    )

    assert _sync([code, _note("Go.md", ["Task"], "n1")], {("Go.md", "Task"): earlier}) == {}


# -- doc roles ----------------------------------------------------------------------------


def _roles_config(tmp_path: Path, extra: str = "", source_extra: str = "") -> object:
    path = tmp_path / "rag_config.toml"
    path.write_text(
        'vault_path = "./v"\nqdrant_path = "./q"\nfts_path = "./f.sqlite"\n'
        + extra
        + '[[code_sources]]\nname = "market"\npath = "./m"\n'
        + source_extra,
        encoding="utf-8",
    )
    return load_config(path)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("repo:market/README.md", REFERENCE),
        ("repo:market/plugins/desk/skills/plan/SKILL.md", REFERENCE),
        ("repo:market/docs/guide.md", REFERENCE),
        ("repo:market/docs/planning.md", REFERENCE),
        ("repo:market/docs/specs/quant/design.md", RECORD),
        ("repo:market/docs/specs/quant/plan-pr2.md", RECORD),
        ("repo:market/docs/superpowers/plans/x.md", RECORD),
        ("repo:market/docs/advance-design.md", RECORD),
        ("repo:market/docs/audit-fix-2026-09-15.md", RECORD),
        ("repo:market/docs/2026-10-26-hardening/notes.md", RECORD),
        ("repo:market/adr/0001-use-sqlite.md", RECORD),
        # An explicit reference glob wins over a record folder.
        ("repo:market/docs/specs/README.md", REFERENCE),
        ("repo:market/CHANGELOG.md", HISTORY),
        ("repo:market/plugins/desk/CHANGELOG.md", HISTORY),
        # Vault notes get the same defaults.
        ("docs/specs/2026-10-26-hardening/design.md", RECORD),
        ("Journal/2026-11-04.md", RECORD),
        ("Learning/Retries.md", REFERENCE),
    ],
)
def test_default_doc_roles(tmp_path: Path, path: str, expected: str) -> None:
    assert doc_role(path, _roles_config(tmp_path)) == expected


def test_configured_doc_roles_replace_that_roles_defaults(tmp_path: Path) -> None:
    config = _roles_config(
        tmp_path,
        extra='doc_roles = { record = ["docs/specs/**"] }\n',
        source_extra='doc_roles = { record = ["rfc-*.md"] }\n',
    )

    assert doc_role("docs/specs/x/design.md", config) == RECORD
    assert doc_role("Learning/Retries.md", config) == REFERENCE
    assert doc_role("repo:market/rfc-7.md", config) == RECORD
    # The record defaults were replaced; the other roles keep theirs.
    assert doc_role("repo:market/docs/specs/x/design.md", config) == REFERENCE
    assert doc_role("repo:market/CHANGELOG.md", config) == HISTORY


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ('"docs/**"', "must be a table"),
        ("{ draft = [] }", "unknown roles"),
        ('{ record = "docs/**" }', r"doc_roles\.record must be an array"),
    ],
)
def test_doc_roles_are_validated(tmp_path: Path, value: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _roles_config(tmp_path, extra=f"doc_roles = {value}\n")
    with pytest.raises(ValueError, match=message):
        _roles_config(tmp_path, source_extra=f"doc_roles = {value}\n")


def _state(src: str, mention: str, status: str, when: float | None = None, symbol: str | None = "f") -> MentionState:
    return MentionState(
        src=src,
        mention=mention,
        target="repo:sb/r.py",
        symbol=symbol,
        src_hash="h",
        acked_hash="a",
        current_hash=None if status == DANGLING else "c",
        changed_at=when if status == STALE else None,
        lost_at=when if status == DANGLING else None,
        status=status,
    )


def test_record_summaries_rank_records_by_how_far_their_code_moved() -> None:
    states = [
        # Half moved: one changed, one gone.
        _state("spec.md", "alpha_fn", STALE, 10.0),
        _state("spec.md", "beta_fn", DANGLING, 30.0),
        _state("spec.md", "gamma_fn", OK),
        _state("spec.md", "delta_fn", OK),
        # Everything moved, but only one mention.
        _state("plan.md", "alpha_fn", STALE, 20.0),
        # Nothing moved: not listed.
        _state("old.md", "alpha_fn", OK),
        # A file-level mention, left out when only symbols are asked for.
        _state("spec.md", "r.py", STALE, 40.0, symbol=None),
    ]

    summaries = record_summaries(states, ["symbol", "file"])
    # 3 of 5 moved outranks 1 of 1: a one-mention record is weak evidence.
    assert [(s["path"], s["moved_share"]) for s in summaries] == [("spec.md", 0.6), ("plan.md", 1.0)]
    spec = summaries[0]
    assert (spec["tracked"], spec["changed"], spec["gone"], spec["changed_at"]) == (5, 2, 1, 40.0)
    assert spec["gone_mentions"] == ["beta_fn"]

    only_symbols = record_summaries(states, ["symbol"])
    assert {(s["path"], s["moved_share"]) for s in only_symbols} == {("plan.md", 1.0), ("spec.md", 0.5)}


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
    assert pair["evidence"] == {"mention": "compute_backoff", "level": "symbol", "role": REFERENCE}

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
    # The symbol is gone: no other declaration's outline stands in for it.
    assert pair["code"]["text"] is None and "start_line" not in pair["code"]
    assert pair["evidence"]["reason"] == SYMBOL_REMOVED
    assert pair["evidence"]["candidates"] == ["repo:sb/pkg/retry.py::compute_backoff_ms"]
    assert pair["evidence"]["lost_at"] is not None


def test_note_excerpts_keep_a_mention_deep_in_a_long_chunk() -> None:
    from second_brain.service import DRIFT_EXCERPT_CHARS, _excerpt_around

    text = "filler " * 400 + "I capped `compute_backoff` at 30s." + " tail" * 400

    excerpt = _excerpt_around(text, "compute_backoff")

    assert "I capped `compute_backoff` at 30s." in excerpt
    assert excerpt.startswith("…") and excerpt.endswith("…")
    assert len(excerpt) == DRIFT_EXCERPT_CHARS + 2
    assert _excerpt_around("short `x`", "x") == "short `x`"
    assert _excerpt_around("ab" * 1000 + "x", "x").endswith("x")


def test_rag_drift_lists_reference_pairs_and_summarizes_records(tmp_path: Path) -> None:
    service, vault, repo = _service(tmp_path)
    _write(vault / "plans" / "Retry-plan.md", "# Plan\nRewrite `compute_backoff` and `pkg/clock.py`.\n")
    _write(repo / "CHANGELOG.md", "# Changelog\n- Cap `compute_backoff` at 30s.\n")
    service.config.doc_roles = validated_doc_roles({"record": ["plans/**"]}, "doc_roles", DEFAULT_DOC_ROLES)
    service.sync(mode="full")
    # The changelog is history: its mention is never tracked.
    assert {key[0] for key in service.graph_store.code_mentions()} == {"Retries.md", "plans/Retry-plan.md"}

    _write(repo / "pkg" / "retry.py", _RETRY.replace("30)", "60)"))
    service.sync(mode="incremental")
    result = service.drift()

    assert [(p["note"]["path"], p["evidence"]["role"]) for p in result["pairs"]] == [("Retries.md", REFERENCE)]
    assert result["counts"] == {OK: 1, STALE: 1, DANGLING: 0}
    assert result["record_total"] == 1
    [record] = result["records"]
    assert (record["path"], record["tracked"], record["changed"], record["gone"], record["moved_share"]) == (
        "plans/Retry-plan.md", 2, 1, 0, 0.5,
    )

    as_records = service.drift(roles=[RECORD])
    assert [p["note"]["path"] for p in as_records["pairs"]] == ["plans/Retry-plan.md"]
    both = service.drift(roles=[REFERENCE, RECORD])
    assert {p["note"]["path"] for p in both["pairs"]} == {"Retries.md", "plans/Retry-plan.md"}
    with pytest.raises(ValueError, match="roles"):
        service.drift(roles=[HISTORY])


def test_search_filters_by_doc_role(tmp_path: Path) -> None:
    service, vault, _ = _service(tmp_path)
    _write(vault / "plans" / "Retry-plan.md", "# Plan\nRewrite the retry backoff.\n")
    service.config.doc_roles = validated_doc_roles({"record": ["plans/**"]}, "doc_roles", DEFAULT_DOC_ROLES)
    service.sync(mode="full")

    def paths(role) -> set[str]:
        hits = service.search("retry backoff", filters={"doc_role": role}, top_k=20)["hits"]
        return {hit["metadata"]["path"] for hit in hits}

    assert paths("record") == {"plans/Retry-plan.md"}
    reference = paths(["reference"])
    assert "Retries.md" in reference and "plans/Retry-plan.md" not in reference
    # Code chunks are not docs and match no role.
    assert not any(path.endswith(".py") for path in reference)
    with pytest.raises(ValueError, match="doc_role"):
        service.search("retry", filters={"doc_role": "draft"})


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
