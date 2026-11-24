import math
from pathlib import Path

import pytest

from second_brain.config import load_config
from second_brain.models import RetrievalHit
from second_brain.retrieval import (
    _is_repo_aux_path,
    apply_filename_boost,
    apply_repo_aux_weight,
    validate_repo_aux_weight,
)


def _hit(chunk_id: str, score: float, path: str, bank_type: str = "repo") -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        score=score,
        source="hybrid",
        text="",
        metadata={"path": path, "bank_type": bank_type},
    )


# -- repo_aux_weight ---------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "tests/test_graph.py",
        "src/test_graph.py",
        "test_graph.py",
        "graph_test.py",
        "docs/usage.md",
        "README.markdown",
        "guide.rst",
        "notes.txt",
    ],
)
def test_repo_aux_paths_are_tests_or_prose(path: str) -> None:
    assert _is_repo_aux_path(path) is True


@pytest.mark.parametrize(
    "path",
    ["second_brain/graph.py", "src/graph.py", "graph.pyi", "", 42, None],
)
def test_implementation_paths_are_not_repo_aux(path: object) -> None:
    assert _is_repo_aux_path(path) is False


def test_repo_aux_weight_lets_implementation_pass_auxiliary_near_tie() -> None:
    aux = _hit("aux", 0.8, "tests/test_graph.py")
    impl = _hit("impl", 0.5, "second_brain/graph.py")

    weighted = apply_repo_aux_weight([aux, impl], 0.5)

    assert [hit.chunk_id for hit in weighted] == ["impl", "aux"]
    assert weighted[1].score == pytest.approx(0.4)


def test_repo_aux_weight_keeps_clearly_winning_auxiliary_first() -> None:
    aux = _hit("aux", 1.0, "docs/usage.md")
    impl = _hit("impl", 0.5, "second_brain/graph.py")

    weighted = apply_repo_aux_weight([aux, impl], 0.5)

    assert [hit.chunk_id for hit in weighted] == ["aux", "impl"]


def test_repo_aux_weight_leaves_vault_hits_untouched() -> None:
    note = _hit("note", 0.9, "tests/notes.md", bank_type="knowledge-base")
    impl = _hit("impl", 0.5, "second_brain/graph.py")

    weighted = apply_repo_aux_weight([note, impl], 0.5)

    assert [hit.chunk_id for hit in weighted] == ["note", "impl"]
    assert weighted[0].score == 0.9


def test_full_repo_aux_weight_keeps_fused_order() -> None:
    hits = [_hit("a", 0.9, "tests/test_a.py"), _hit("b", 0.8, "tests/test_b.py")]

    assert apply_repo_aux_weight(hits, 1.0) == hits


@pytest.mark.parametrize("weight", [0.0, 1.5, True, "0.5", math.nan])
def test_repo_aux_weight_must_be_in_unit_interval(weight: object) -> None:
    with pytest.raises(ValueError, match="repo_aux_weight"):
        validate_repo_aux_weight(weight)


# -- filename_boost ----------------------------------------------------------


def _sem_hit(
    chunk_id: str, score: float, path: str, semantic_score: float
) -> RetrievalHit:
    hit = _hit(chunk_id, score, path)
    hit.semantic_score = semantic_score
    return hit


def test_filename_boost_passes_stem_match_within_semantic_near_tie() -> None:
    leader = _sem_hit("leader", 0.500, "second_brain/parser.py", 0.90)
    challenger = _sem_hit("challenger", 0.495, "second_brain/canvas.py", 0.895)

    boosted = apply_filename_boost([leader, challenger], {"canvas"}, 0.5)

    assert boosted[0].chunk_id == "challenger"
    assert boosted[0].score == pytest.approx(0.495 + 0.5 / 61)


def test_filename_boost_respects_semantic_gap() -> None:
    leader = _sem_hit("leader", 0.500, "second_brain/parser.py", 0.90)
    challenger = _sem_hit("challenger", 0.495, "second_brain/canvas.py", 0.87)

    boosted = apply_filename_boost([leader, challenger], {"canvas"}, 0.5)

    assert boosted[0].chunk_id == "leader"


def test_filename_boost_ignores_leader_without_semantic_score() -> None:
    leader = _hit("leader", 0.500, "second_brain/parser.py")
    challenger = _sem_hit("challenger", 0.495, "second_brain/canvas.py", 0.895)

    boosted = apply_filename_boost([leader, challenger], {"canvas"}, 0.5)

    assert boosted[0].chunk_id == "leader"


def test_filename_boost_without_matching_tokens_returns_hits() -> None:
    hits = [_sem_hit("a", 0.5, "second_brain/graph.py", 0.9)]

    assert apply_filename_boost(hits, {"unrelated"}, 0.5) == hits
    assert apply_filename_boost(hits, set(), 0.5) == hits
    assert apply_filename_boost(hits, {"graph"}, 0.0) == hits


# -- config ------------------------------------------------------------------


def _config_file(tmp_path: Path, extra: str = "") -> Path:
    path = tmp_path / "rag_config.toml"
    path.write_text(
        'vault_path = "./vault"\nqdrant_path = "./q"\nfts_path = "./f.sqlite"\n'
        + extra,
        encoding="utf-8",
    )
    return path


def test_repo_aux_weight_defaults_to_one(tmp_path: Path) -> None:
    assert load_config(_config_file(tmp_path)).repo_aux_weight == 1.0


def test_repo_aux_weight_loads_from_config(tmp_path: Path) -> None:
    config = load_config(_config_file(tmp_path, "repo_aux_weight = 0.5\n"))

    assert config.repo_aux_weight == 0.5


@pytest.mark.parametrize("value", ["0", "1.5", '"0.5"', "true"])
def test_invalid_repo_aux_weight_fails_at_config_load(
    tmp_path: Path, value: str
) -> None:
    with pytest.raises(ValueError, match="repo_aux_weight"):
        load_config(_config_file(tmp_path, f"repo_aux_weight = {value}\n"))


def test_filename_boost_defaults_to_zero(tmp_path: Path) -> None:
    assert load_config(_config_file(tmp_path)).filename_boost == 0.0


def test_filename_boost_loads_from_config(tmp_path: Path) -> None:
    config = load_config(_config_file(tmp_path, "filename_boost = 0.5\n"))

    assert config.filename_boost == 0.5


@pytest.mark.parametrize("value", ["-0.1", "1.5", '"0.5"', "true"])
def test_invalid_filename_boost_fails_at_config_load(
    tmp_path: Path, value: str
) -> None:
    with pytest.raises(ValueError, match="filename_boost"):
        load_config(_config_file(tmp_path, f"filename_boost = {value}\n"))
