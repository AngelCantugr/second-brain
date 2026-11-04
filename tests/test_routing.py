"""Query routing: repo-intent terms and code_source_weight."""

from pathlib import Path

import pytest

from second_brain.config import CodeSource, RagConfig, load_config
from second_brain.models import RetrievalHit
from second_brain.retrieval import apply_code_source_weight, validate_code_source_weight
from second_brain.routing import build_repo_terms, repo_intent_terms
from second_brain.service import RagService


def _vault(path: str, text: str) -> tuple[dict, str]:
    return {"path": path}, text


def _repo(path: str, text: str = "", **extra) -> tuple[dict, str]:
    return {"path": f"repo:market/{path}", "repo": "market", **extra}, text


# -- repo terms -------------------------------------------------------------


def test_repo_terms_come_from_repo_name_path_components_and_symbols() -> None:
    terms = build_repo_terms(
        [
            _repo(
                "plugins/career-desk/scripts/ledger_tool.py",
                symbol_name="parse_evidence",
            ),
            _repo("docs/memory-tiers.md"),
        ]
    )

    assert {
        "market",
        "career-desk",
        "ledger_tool",
        "parse_evidence",
        "memory-tiers",
    } <= terms
    # File extensions are not part of the term.
    assert "ledger_tool.py" not in terms


def test_repo_terms_drop_names_common_in_the_vault() -> None:
    vault = [
        _vault(f"Note{i}.md", "My plugins and scripts, plus a ledger.")
        for i in range(3)
    ]
    vault.append(_vault("Other.md", "The career-desk idea."))

    terms = build_repo_terms([*vault, _repo("plugins/career-desk/scripts/ledger.py")])

    # In three vault notes: too common to signal the repo.
    assert not {"plugins", "scripts", "ledger"} & terms
    # In one vault note: still distinctive.
    assert "career-desk" in terms


def test_repo_terms_count_whole_words_not_substrings() -> None:
    vault = [_vault(f"Note{i}.md", "The career-desk-design doc.") for i in range(3)]

    terms = build_repo_terms([*vault, _repo("plugins/career-desk/x.md")])

    assert "career-desk" in terms


def test_repo_terms_keep_private_and_dunder_symbols() -> None:
    terms = build_repo_terms(
        [
            _repo("pkg/chunker.py", symbol_name="_extract_python"),
            _repo("pkg/models.py", symbol_name="__init__"),
        ]
    )

    assert {"_extract_python", "__init__"} <= terms


def test_repo_terms_count_names_written_with_spaces_or_in_italics() -> None:
    vault = [
        _vault(f"Spaced{i}.md", "Notes on the Career Desk plan.") for i in range(2)
    ]
    vault.append(_vault("Italic.md", "See _career-desk_ for details."))

    terms = build_repo_terms([*vault, _repo("plugins/career-desk/x.md")])

    # Two notes spell it with spaces and one in italics: three vault notes.
    assert "career-desk" not in terms


def test_repo_terms_skip_short_and_untokenizable_names() -> None:
    terms = build_repo_terms(
        [_repo("src/app.test.ts", symbol_name="run"), _repo("docs/2026/v1.md")]
    )

    assert not {"src", "run", "app.test", "2026"} & terms


# -- repo intent ------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("why does career-desk keep a ledger", ["career-desk"]),
        ("Why does Career-Desk keep a ledger?", ["career-desk"]),
        ("why does the career desk keep a ledger", ["career-desk"]),
        ("what calls parse_evidence", ["parse_evidence"]),
        ("what does ledger_tool.py do", ["ledger_tool"]),
        ("what does _extract_python skip", ["_extract_python"]),
        ("which classes define __init__", ["__init__"]),
        ("the _career-desk_ plan", ["career-desk"]),
        ("what is promotion evidence", []),
        # A term inside a longer hyphenated word is a different name.
        ("the career-desk-v2 plan", []),
    ],
)
def test_repo_intent_matches_whole_terms(query: str, expected: list[str]) -> None:
    terms = frozenset(
        {"career-desk", "parse_evidence", "ledger_tool", "_extract_python", "__init__"}
    )

    assert repo_intent_terms(query, terms) == expected


# -- weighting --------------------------------------------------------------


def _hit(chunk_id: str, score: float, repo: str | None = None) -> RetrievalHit:
    metadata = {"path": f"{chunk_id}.md"}
    if repo:
        metadata["repo"] = repo
    return RetrievalHit(chunk_id, score, "hybrid", "", metadata)


def test_code_source_weight_scales_and_reorders_code_source_hits() -> None:
    hits = [_hit("repo", 0.030, repo="market"), _hit("vault", 0.020)]

    weighted = apply_code_source_weight(hits, 0.5)

    assert [hit.chunk_id for hit in weighted] == ["vault", "repo"]
    assert weighted[1].score == pytest.approx(0.015)
    # The input hits are not mutated.
    assert hits[0].score == 0.030


def test_full_code_source_weight_keeps_fused_order() -> None:
    hits = [_hit("repo", 0.030, repo="market"), _hit("vault", 0.020)]

    assert apply_code_source_weight(hits, 1.0) == hits


@pytest.mark.parametrize("weight", [0.0, -0.1, 1.01, float("nan"), True, "0.5"])
def test_code_source_weight_must_be_in_unit_interval(weight) -> None:
    with pytest.raises(ValueError, match="code_source_weight"):
        validate_code_source_weight(weight)


# -- config -----------------------------------------------------------------


def _config_file(tmp_path: Path, extra: str = "") -> Path:
    path = tmp_path / "rag_config.toml"
    path.write_text(
        'vault_path = "./vault"\nqdrant_path = "./q"\nfts_path = "./f.sqlite"\n'
        + extra,
        encoding="utf-8",
    )
    return path


def test_code_source_weight_defaults_to_one(tmp_path: Path) -> None:
    assert load_config(_config_file(tmp_path)).code_source_weight == 1.0


def test_code_source_weight_loads_from_config(tmp_path: Path) -> None:
    config = load_config(_config_file(tmp_path, "code_source_weight = 0.5\n"))

    assert config.code_source_weight == 0.5


@pytest.mark.parametrize("value", ["0", "1.5", '"0.5"', "true"])
def test_invalid_code_source_weight_fails_at_config_load(
    tmp_path: Path, value: str
) -> None:
    with pytest.raises(ValueError, match="code_source_weight"):
        load_config(_config_file(tmp_path, f"code_source_weight = {value}\n"))


# -- service ----------------------------------------------------------------


class _StubEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _routed_service(tmp_path: Path, monkeypatch, weight: float = 0.5) -> RagService:
    vault = tmp_path / "vault"
    for i in range(3):
        _write(vault / f"Note{i}.md", f"# Note {i}\nA design for weekly reviews.\n")
    repo = tmp_path / "repo"
    _write(repo / "docs" / "design.md", "# Design\nWhy the market exists.\n")
    _write(repo / "docs" / "ledger-tool.md", "# Ledger tool\nHow the ledger works.\n")
    config = RagConfig(
        vault_path=vault,
        qdrant_path=tmp_path / "qdrant",
        fts_path=tmp_path / "fts.sqlite",
        sync_state_path=tmp_path / "sync_state.sqlite",
        code_sources=[CodeSource("market", repo)],
        code_source_weight=weight,
    )
    service = RagService(config, use_in_memory_vector=True)
    service.embedder = _StubEmbedder()
    service.indexer.embedder = service.embedder
    service.sync(mode="full")

    hits = [
        RetrievalHit(
            "repo",
            1.0,
            "semantic",
            "",
            {"path": "repo:market/docs/design.md", "repo": "market"},
        ),
        RetrievalHit("vault", 0.9, "semantic", "", {"path": "Note0.md"}),
    ]
    monkeypatch.setattr(service.vector_store, "search", lambda *args, **kwargs: hits)
    monkeypatch.setattr(service.keyword_store, "search", lambda *args, **kwargs: [])
    return service


def _ranked(service: RagService, query: str, **kwargs) -> list[str]:
    return [hit["chunk_id"] for hit in service.search(query, top_k=2, **kwargs)["hits"]]


def test_search_down_weights_code_sources_for_queries_without_repo_intent(
    tmp_path: Path, monkeypatch
) -> None:
    service = _routed_service(tmp_path, monkeypatch)

    # "design" is in three vault notes, so it does not signal the repo.
    assert _ranked(service, "weekly design") == ["vault", "repo"]


@pytest.mark.parametrize(
    "query", ["the market design", "how does the ledger tool work"]
)
def test_search_keeps_code_source_rank_when_the_query_names_the_repo(
    tmp_path: Path, monkeypatch, query: str
) -> None:
    service = _routed_service(tmp_path, monkeypatch)

    assert _ranked(service, query) == ["repo", "vault"]


def test_search_code_source_weight_argument_overrides_config(
    tmp_path: Path, monkeypatch
) -> None:
    service = _routed_service(tmp_path, monkeypatch)

    assert _ranked(service, "weekly design", code_source_weight=1.0) == [
        "repo",
        "vault",
    ]
    with pytest.raises(ValueError, match="code_source_weight"):
        service.search("weekly design", code_source_weight=0.0)


def test_default_config_leaves_ranking_unchanged(tmp_path: Path, monkeypatch) -> None:
    service = _routed_service(tmp_path, monkeypatch, weight=1.0)

    assert _ranked(service, "weekly design") == ["repo", "vault"]


def test_sync_refreshes_repo_terms(tmp_path: Path, monkeypatch) -> None:
    service = _routed_service(tmp_path, monkeypatch)
    assert _ranked(service, "the quarry notes") == ["vault", "repo"]

    _write(tmp_path / "repo" / "docs" / "quarry.md", "# Quarry\nNew doc.\n")
    service.sync(mode="incremental")

    assert _ranked(service, "the quarry notes") == ["repo", "vault"]
