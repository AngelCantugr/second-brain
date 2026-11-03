"""Typed graph edges (G6): mention extraction, note->code and import edges, rag.related output."""

import sqlite3
from pathlib import Path

import pytest

from second_brain.code_graph import IMPORTS, LINKS, MENTIONS_CODE, code_edges, looks_like_symbol
from second_brain.graph import GraphStore
from second_brain.parser import extract_code_mentions

# -- mention extraction ------------------------------------------------------------


def test_mentions_come_from_inline_code_urls_and_prose_paths_not_fences() -> None:
    body = (
        "Rewrote `GraphBuilder.rebuild_full()` in second_brain/graph.py today.\n"
        "See https://github.com/me/second-brain/blob/main/docs/x.md and `sb`.\n"
        "```python\nfrom inside_fence import `Hidden`\nother/path.py\n```\n"
        "A url like https://example.com/a/b.py is not a path.\n"
    )

    mentions = extract_code_mentions(body)

    assert mentions == [
        "GraphBuilder.rebuild_full()",
        "sb",
        "https://github.com/me/second-brain/blob/main/docs/x.md",
        "second_brain/graph.py",
    ]


@pytest.mark.parametrize(
    ("name", "expected"),
    [("chunk_note", True), ("GraphBuilder", True), ("Indexer", True), ("sync", False), ("path", False),
     ("__init__", False), ("ok", False)],
)
def test_only_specific_bare_names_can_match_symbols(name: str, expected: bool) -> None:
    assert looks_like_symbol(name) is expected


# -- code_edges ---------------------------------------------------------------------


def _meta(path: str, **code) -> dict:
    return {"path": path, "code": code}


def _code_population(*notes: dict) -> list[dict]:
    return [
        _meta("repo:sb/README.md", mentions=["GraphBuilder", "other_repo_fn", "docs/design.md"]),
        _meta("repo:sb/docs/design.md"),
        _meta(
            "repo:sb/second_brain/graph.py",
            symbols=["GraphBuilder", "GraphBuilder.rebuild_full", "Loader"],
            imports=["second_brain.models", "second_brain.models.Edge", "numpy", "json"],
        ),
        _meta("repo:sb/second_brain/models.py", symbols=["Edge"], imports=[".graph"]),
        _meta("repo:sb/second_brain/loaders.py", symbols=["Loader", "load"]),
        _meta("repo:sb/second_brain/json.py"),
        _meta("repo:sb/second_brain/__init__.py"),
        _meta("repo:other/README.md"),
        _meta("repo:other/pkg/util.go", symbols=["other_repo_fn"]),
        _meta("repo:other/cmd/main.go", imports=["github.com/me/other/pkg", "fmt", "errors"]),
        _meta("repo:other/errors/errors.go"),
        _meta("repo:other/web/app.ts", imports=["./api.js", "react"]),
        _meta("repo:other/web/api.ts"),
        *notes,
    ]


def _by_pair(edges) -> dict:
    return {(edge.src, edge.dst): edge for edge in edges}


def test_note_mentions_link_to_the_one_file_that_defines_or_is_them() -> None:
    note = _meta(
        "Notes/Graph.md",
        mentions=[
            "GraphBuilder",            # unique symbol
            "graph.GraphBuilder.rebuild_full()",  # dotted, leading module dropped
            "Loader",                  # defined in two files: ambiguous
            "load",                    # bare lowercase word: never a symbol
            "second_brain/models.py",  # path suffix
            "other",                   # repo name -> its README
            "https://github.com/me/other/blob/main/web/api.ts#L3",
            "util.go:12",              # basename with line suffix
            "README.md",               # bare markdown name: whose README?
            "__init__.py",             # generic name, even if unique here
            "nothing_like_this",
        ],
    )

    edges = _by_pair(code_edges(_code_population(note), link_weight=0.25))
    from_note = {pair: edge for pair, edge in edges.items() if "Notes/Graph.md" in pair}

    assert set(from_note) == {
        ("Notes/Graph.md", "repo:sb/second_brain/graph.py"),
        ("Notes/Graph.md", "repo:sb/second_brain/models.py"),
        ("Notes/Graph.md", "repo:other/README.md"),
        ("Notes/Graph.md", "repo:other/web/api.ts"),
        ("Notes/Graph.md", "repo:other/pkg/util.go"),
    }
    graph_edge = from_note[("Notes/Graph.md", "repo:sb/second_brain/graph.py")]
    assert graph_edge.type == MENTIONS_CODE
    assert graph_edge.code_refs == ["GraphBuilder", "graph.GraphBuilder.rebuild_full()"]
    assert graph_edge.link_src_to_dst and not graph_edge.link_dst_to_src
    assert graph_edge.link == 0.7 and graph_edge.composite == pytest.approx(0.25 * 0.7)
    assert graph_edge.semantic == 0.0 and graph_edge.shared_tags == []


def test_module_qualified_symbols_and_github_urls_to_root_files_resolve() -> None:
    note = _meta(
        "Notes/Qualified.md",
        mentions=[
            "graph.Edge",       # module.Class for a top-level symbol
            "self.load",        # terminal is a bare lowercase word: no match
            "https://github.com/me/sb/blob/main/README.md",  # root generic file, repo named
        ],
    )

    edges = _by_pair(code_edges(_code_population(note), link_weight=0.25))
    targets = {dst if src == "Notes/Qualified.md" else src for src, dst in edges if "Notes/Qualified.md" in (src, dst)}

    assert targets == {"repo:sb/second_brain/models.py", "repo:sb/README.md"}


def test_repo_docs_mention_only_their_own_repo() -> None:
    edges = _by_pair(code_edges(_code_population(), link_weight=0.25))

    assert edges[("repo:sb/README.md", "repo:sb/second_brain/graph.py")].type == MENTIONS_CODE
    assert ("repo:sb/README.md", "repo:sb/docs/design.md") in edges
    # other_repo_fn is defined only in repo "other"; README of "sb" cannot reach it.
    assert not any("util.go" in dst for (src, dst) in edges if src == "repo:sb/README.md")


def test_imports_resolve_python_typescript_and_go_within_a_repo() -> None:
    edges = _by_pair(code_edges(_code_population(), link_weight=0.25))

    mutual = edges[("repo:sb/second_brain/graph.py", "repo:sb/second_brain/models.py")]
    assert mutual.type == IMPORTS
    assert mutual.link_src_to_dst and mutual.link_dst_to_src and mutual.link == 1.0
    # second_brain.models.Edge names a class, not a module file, so it matches nothing.
    assert mutual.code_refs == [".graph", "second_brain.models"]
    assert edges[("repo:other/web/api.ts", "repo:other/web/app.ts")].link_dst_to_src
    assert ("repo:other/cmd/main.go", "repo:other/pkg/util.go") in edges
    # Standard-library ``import "errors"`` must not claim the repo's errors/ package.
    assert ("repo:other/cmd/main.go", "repo:other/errors/errors.go") not in edges
    # ``import json`` must not claim the repo's own second_brain/json.py.
    assert not any("json.py" in pair[0] or "json.py" in pair[1] for pair in edges)


def test_go_root_packages_and_emitted_typescript_extensions_resolve() -> None:
    population = [
        _meta("repo:widget/widget.go", symbols=["Widget"]),
        _meta("repo:widget/pkg/util/util.go"),
        # Learns the module path from its own subpackage import, then resolves the root.
        _meta("repo:widget/cmd/tool/main.go", imports=["github.com/acme/widget", "github.com/acme/widget/pkg/util"]),
        # Matches by repo name even with a /v2 major suffix.
        _meta("repo:widget/cmd/v2tool/main.go", imports=["github.com/acme/widget/v2"]),
        # A stranger's module that is not this repo's name or learned path.
        _meta("repo:widget/cmd/other/main.go", imports=["github.com/acme/gadget"]),
        _meta("repo:web/src/main.mts", imports=["./helper.mjs", "./legacy.cjs", "./view.jsx"]),
        _meta("repo:web/src/helper.mts"),
        _meta("repo:web/src/legacy.cts"),
        _meta("repo:web/src/view.tsx"),
    ]

    pairs = set(_by_pair(code_edges(population, link_weight=0.25)))

    assert ("repo:widget/cmd/tool/main.go", "repo:widget/widget.go") in pairs
    assert ("repo:widget/cmd/v2tool/main.go", "repo:widget/widget.go") in pairs
    assert not any("cmd/other" in src or "cmd/other" in dst for src, dst in pairs)
    assert {("repo:web/src/helper.mts", "repo:web/src/main.mts"), ("repo:web/src/legacy.cts", "repo:web/src/main.mts"),
            ("repo:web/src/main.mts", "repo:web/src/view.tsx")} <= pairs


def test_paths_differing_only_in_case_stay_distinct() -> None:
    population = [
        _meta("repo:sb/pkg/Helper.py"),
        _meta("repo:sb/pkg/helper.py"),
        _meta("repo:sb/pkg/main.py", imports=[".Helper"]),
        _meta(
            "Notes/Case.md",
            mentions=["pkg/helper.py", "PKG/HELPER.PY"],  # exact case wins; folded is ambiguous
        ),
    ]

    pairs = set(_by_pair(code_edges(population, link_weight=0.25)))

    assert ("repo:sb/pkg/Helper.py", "repo:sb/pkg/main.py") in pairs
    assert ("repo:sb/pkg/helper.py", "repo:sb/pkg/main.py") not in pairs
    note_targets = {dst for src, dst in pairs if src == "Notes/Case.md"}
    assert note_targets == {"repo:sb/pkg/helper.py"}


def test_python_stubs_resolve_and_win_only_for_stub_importers() -> None:
    population = [
        _meta("repo:sb/pkg/main.pyi", imports=[".helper"]),
        _meta("repo:sb/pkg/run.py", imports=[".helper"]),
        _meta("repo:sb/pkg/helper.pyi"),
        _meta("repo:sb/pkg/helper.py"),
        _meta("repo:sb/stubs/only.pyi", imports=["stubs.lonely"]),
        _meta("repo:sb/stubs/lonely/__init__.pyi"),
    ]

    pairs = set(_by_pair(code_edges(population, link_weight=0.25)))

    assert ("repo:sb/pkg/helper.pyi", "repo:sb/pkg/main.pyi") in pairs
    assert ("repo:sb/pkg/helper.py", "repo:sb/pkg/run.py") in pairs
    assert ("repo:sb/pkg/helper.pyi", "repo:sb/pkg/run.py") not in pairs
    assert ("repo:sb/stubs/lonely/__init__.pyi", "repo:sb/stubs/only.pyi") in pairs


def test_github_urls_ending_a_sentence_drop_the_punctuation() -> None:
    mentions = extract_code_mentions(
        "See https://github.com/me/sb/blob/main/pkg/helper.py. Also https://github.com/me/sb, ok?\n"
    )
    assert mentions == ["https://github.com/me/sb/blob/main/pkg/helper.py", "https://github.com/me/sb"]

    # Mentions stored before the parser trimmed punctuation still resolve.
    population = [
        _meta("repo:sb/README.md"),
        _meta("repo:sb/pkg/helper.py"),
        _meta("Notes/Old.md", mentions=["https://github.com/me/sb/blob/main/pkg/helper.py."]),
    ]
    assert ("Notes/Old.md", "repo:sb/pkg/helper.py") in _by_pair(code_edges(population, 0.25))


def test_no_code_sources_means_no_typed_edges() -> None:
    vault_only = [_meta("A.md", mentions=["GraphBuilder"]), _meta("B.md")]

    assert code_edges(vault_only, link_weight=0.25) == []


# -- storage -------------------------------------------------------------------------


def test_edges_written_before_edge_types_read_back_as_links(tmp_path: Path) -> None:
    db = tmp_path / "fts.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE edges (src TEXT NOT NULL, dst TEXT NOT NULL, semantic REAL NOT NULL DEFAULT 0,"
            " link REAL NOT NULL DEFAULT 0, tag REAL NOT NULL DEFAULT 0, comention REAL NOT NULL DEFAULT 0,"
            " comention_count INTEGER NOT NULL DEFAULT 0, link_src_to_dst INTEGER NOT NULL DEFAULT 0,"
            " link_dst_to_src INTEGER NOT NULL DEFAULT 0, shared_tags_json TEXT NOT NULL DEFAULT '[]',"
            " composite REAL NOT NULL, updated_at REAL, PRIMARY KEY (src, dst))"
        )
        conn.execute("INSERT INTO edges(src, dst, link, composite) VALUES ('a.md', 'b.md', 1.0, 0.4)")
        conn.execute(
            "CREATE TABLE note_meta (path TEXT PRIMARY KEY, title TEXT NOT NULL, title_key TEXT NOT NULL,"
            " tags_json TEXT NOT NULL, links_json TEXT NOT NULL, centroid BLOB, dim INTEGER, updated_at REAL)"
        )
        conn.execute("INSERT INTO note_meta VALUES ('a.md', 'a', 'a', '[]', '[]', NULL, NULL, 0)")

    store = GraphStore(db)
    store.initialize()

    (edge,) = store.all_edges()
    assert (edge.type, edge.code_refs) == (LINKS, [])
    assert store.note_meta_for("a.md")["code"] == {}


# -- end to end ------------------------------------------------------------------------

tree_sitter = pytest.importorskip("tree_sitter")

from second_brain.config import CodeSource, RagConfig  # noqa: E402
from second_brain.service import RagService  # noqa: E402


class _StubEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0, 0.5] for text in texts]

    def health(self) -> bool:
        return True


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _service(tmp_path: Path) -> tuple[RagService, Path, Path]:
    vault, repo = tmp_path / "vault", tmp_path / "repo"
    _write(vault / "Retries.md", "# Retries\nI capped `compute_backoff` at 30s. See [[Ops]].\n")
    _write(vault / "Ops.md", "# Ops\nOn-call notes. See [[Retries]].\n")
    _write(repo / "README.md", "# Repo\nRetry helpers.\n")
    _write(
        repo / "pkg" / "retry.py",
        'from pkg.clock import now\n\n\ndef compute_backoff(n):\n    """Delay before retry n."""\n',
    )
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


def test_related_reports_edge_types_and_filters_by_them(tmp_path: Path) -> None:
    service, _, _ = _service(tmp_path)
    assert service.sync(mode="full")["errors"] == []

    neighbors = service.related("Retries.md")["neighbors"]
    by_path = {n["path"]: n for n in neighbors}
    assert by_path["Ops.md"]["type"] == LINKS
    assert "code_refs" not in by_path["Ops.md"]["evidence"]
    code = by_path["repo:sb/pkg/retry.py"]
    assert code["type"] == MENTIONS_CODE
    assert code["evidence"]["code_refs"] == ["compute_backoff"]
    assert code["evidence"]["links_to"] is True

    only_code = service.related("Retries.md", edge_types=[MENTIONS_CODE])["neighbors"]
    assert [n["path"] for n in only_code] == ["repo:sb/pkg/retry.py"]
    # The reverse question: which notes discuss this file, and what does it import?
    from_code = {n["path"]: n["type"] for n in service.related("repo:sb/pkg/retry.py")["neighbors"]}
    assert from_code == {"Retries.md": MENTIONS_CODE, "repo:sb/pkg/clock.py": IMPORTS}

    with pytest.raises(ValueError, match="edge_types"):
        service.related("Retries.md", edge_types=["cites"])


def test_connections_carry_edge_types_along_the_path(tmp_path: Path) -> None:
    service, _, _ = _service(tmp_path)
    service.sync(mode="full")

    direct = service.connections("Retries.md", "repo:sb/pkg/retry.py")
    via = service.connections("Ops.md", "repo:sb/pkg/clock.py")

    assert direct["direct_edge"]["type"] == MENTIONS_CODE
    assert direct["direct_edge"]["evidence"]["code_refs"] == ["compute_backoff"]
    assert via["path"] == ["Ops.md", "Retries.md", "repo:sb/pkg/retry.py", "repo:sb/pkg/clock.py"]
    assert [hop["type"] for hop in via["path_edges"]] == [LINKS, MENTIONS_CODE, IMPORTS]


def test_incremental_sync_retargets_mentions_when_only_the_code_changes(tmp_path: Path) -> None:
    service, vault, repo = _service(tmp_path)
    service.sync(mode="full")

    # Rename the function: the unchanged note's mention no longer matches.
    _write(repo / "pkg" / "retry.py", 'def backoff_delay(n):\n    """Delay before retry n."""\n')
    service.sync(mode="incremental")
    assert service.graph_store.edge_between("Retries.md", "repo:sb/pkg/retry.py") is None
    links_edge = service.graph_store.edge_between("Ops.md", "Retries.md")
    assert links_edge is not None and links_edge.type == LINKS

    # Mention the new name: the edge comes back from the note's side.
    _write(vault / "Retries.md", "# Retries\nNow `backoff_delay`. See [[Ops]].\n")
    service.sync(mode="incremental")
    edge = service.graph_store.edge_between("Retries.md", "repo:sb/pkg/retry.py")
    assert edge is not None and edge.code_refs == ["backoff_delay"]


def test_map_stays_vault_only(tmp_path: Path) -> None:
    service, _, _ = _service(tmp_path)
    service.sync(mode="full")

    graph_map = service.graph_map()

    assert graph_map["note_count"] == 2
    assert not any(o.startswith("repo:") for o in graph_map["orphans"])
