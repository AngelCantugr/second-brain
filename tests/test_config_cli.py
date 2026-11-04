from pathlib import Path

import pytest

from second_brain import cli
from second_brain.cli import _write_init_config
from second_brain.config import load_config


def test_load_config_expands_cwd_and_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "rag_config.toml"
    config_path.write_text(
        (
            'vault_path = "$CWD"\n'
            'qdrant_path = "$CWD/data/qdrant"\n'
            'fts_path = "./data/fts.sqlite"\n'
            'sync_state_path = "./data/sync_state.sqlite"'
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.vault_path == workspace
    assert config.qdrant_path == workspace / "data" / "qdrant"
    assert config.fts_path == config_dir / "data" / "fts.sqlite"
    assert config.sync_state_path == config_dir / "data" / "sync_state.sqlite"


def test_load_config_defaults_graph_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "rag_config.toml"
    config_path.write_text(
        (
            'vault_path = "$CWD"\n'
            'qdrant_path = "$CWD/data/qdrant"\n'
            'fts_path = "./data/fts.sqlite"\n'
            'sync_state_path = "./data/sync_state.sqlite"'
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.graph_enabled is True
    assert config.graph_knn_k == 8
    assert config.graph_semantic_min == 0.35
    assert config.graph_semantic_center is True
    assert config.graph_min_edge_score == 0.15
    assert config.graph_weight_semantic == 0.5
    assert config.graph_weight_link == 0.25
    assert config.graph_weight_tag == 0.15
    assert config.graph_weight_comention == 0.10
    assert config.graph_comention_cap == 3
    assert config.graph_comention_max_fanout == 20


def test_load_config_uses_shared_indexing_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "rag_config.toml"
    config_path.write_text(
        (
            'vault_path = "$CWD"\n'
            'qdrant_path = "$CWD/data/qdrant"\n'
            'fts_path = "./data/fts.sqlite"\n'
            'sync_state_path = "./data/sync_state.sqlite"'
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.exclude_globs == [
        ".obsidian/**", ".git/**", "Templates/**", "_types/**",
        "_templates/**", "**/_templates*/**", "CLAUDE.md", "AGENTS.md", "GEMINI.md",
    ]
    assert config.exclude_status == []


def test_load_config_preserves_explicit_indexing_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "rag_config.toml"
    config_path.write_text(
        (
            'vault_path = "$CWD"\n'
            'qdrant_path = "$CWD/data/qdrant"\n'
            'fts_path = "./data/fts.sqlite"\n'
            'exclude_globs = []\n'
            'exclude_status = ["superseded"]'
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.exclude_globs == []
    assert config.exclude_status == ["superseded"]


@pytest.mark.parametrize("value", ['"superseded"', '["superseded", 2]'])
def test_load_config_rejects_malformed_exclude_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "rag_config.toml"
    config_path.write_text(
        '\n'.join([
            'vault_path = "$CWD"',
            'qdrant_path = "$CWD/data/qdrant"',
            'fts_path = "./data/fts.sqlite"',
            f"exclude_status = {value}",
        ]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exclude_status"):
        load_config(config_path)


def test_init_writes_config_and_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "rag_config.toml"

    result = _write_init_config(config_path)

    assert result["initialized"] is True
    assert result["config_path"] == str(config_path)
    assert result["overwritten"] is False
    assert config_path.exists()
    assert (tmp_path / "data").exists()
    assert "$CWD/data/qdrant" in config_path.read_text(encoding="utf-8")
    assert "skip_headings = []" in config_path.read_text(encoding="utf-8")
    generated = load_config(config_path)
    assert "_types/**" in generated.exclude_globs
    assert "**/_templates*/**" in generated.exclude_globs
    assert "CLAUDE.md" in generated.exclude_globs


def test_init_refuses_overwrite_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "rag_config.toml"
    config_path.write_text('vault_path = "$CWD"\n', encoding="utf-8")

    with pytest.raises(FileExistsError):
        _write_init_config(config_path)


def test_init_overwrites_with_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "rag_config.toml"
    config_path.write_text('vault_path = "/tmp/old"\n', encoding="utf-8")

    result = _write_init_config(config_path, force=True)

    assert result["overwritten"] is True
    assert "$CWD" in config_path.read_text(encoding="utf-8")


def test_load_config_reads_skip_headings_and_context_header_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "rag_config.toml"
    config_path.write_text(
        (
            'vault_path = "$CWD"\n'
            'qdrant_path = "$CWD/data/qdrant"\n'
            'fts_path = "./data/fts.sqlite"\n'
            'skip_headings = ["🔗 Related Notes", "Navigation"]\n'
            "embed_context_header = false\n"
            "graph_semantic_center = false"
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.skip_headings == ["🔗 Related Notes", "Navigation"]
    assert config.embed_context_header is False
    assert config.graph_semantic_center is False


def test_load_config_defaults_skip_headings_and_context_header_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "rag_config.toml"
    config_path.write_text(
        'vault_path = "$CWD"\nqdrant_path = "$CWD/data/qdrant"\nfts_path = "./data/fts.sqlite"\n',
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.skip_headings == []
    assert config.embed_context_header is True


def test_load_config_rejects_non_list_skip_headings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "rag_config.toml"
    config_path.write_text(
        (
            'vault_path = "$CWD"\n'
            'qdrant_path = "$CWD/data/qdrant"\n'
            'fts_path = "./data/fts.sqlite"\n'
            'skip_headings = "Navigation"'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="skip_headings"):
        load_config(config_path)


def _forbid_service(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("RagService must not be built before eval input is validated")

    monkeypatch.setattr(cli, "RagService", fail)


def test_eval_rejects_bad_golden_file_before_building_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    golden = tmp_path / "golden.toml"
    golden.write_text('title = "no cases"\n', encoding="utf-8")
    _forbid_service(monkeypatch)
    monkeypatch.setattr(
        "sys.argv", ["second-brain", "--config", str(tmp_path / "missing.toml"),
                     "eval", "--golden", str(golden)],
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 2
    assert "cases" in capsys.readouterr().err


def test_eval_rejects_depth_below_largest_k_before_building_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    golden = tmp_path / "golden.toml"
    golden.write_text('[[cases]]\nquery = "q"\nexpected = ["a.md"]\n', encoding="utf-8")
    _forbid_service(monkeypatch)
    monkeypatch.setattr(
        "sys.argv", ["second-brain", "--config", str(tmp_path / "missing.toml"),
                     "eval", "--golden", str(golden), "--depth", "3"],
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 2
    assert "depth" in capsys.readouterr().err


def test_eval_rejects_out_of_range_code_source_weight_before_building_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    golden = tmp_path / "golden.toml"
    golden.write_text('[[cases]]\nquery = "q"\nexpected = ["a.md"]\n', encoding="utf-8")
    _forbid_service(monkeypatch)
    monkeypatch.setattr(
        "sys.argv", ["second-brain", "--config", str(tmp_path / "missing.toml"),
                     "eval", "--golden", str(golden), "--code-source-weight", "0"],
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 2
    assert "code_source_weight" in capsys.readouterr().err


def test_eval_rejects_out_of_range_graph_boost_before_building_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    golden = tmp_path / "golden.toml"
    golden.write_text('[[cases]]\nquery = "q"\nexpected = ["a.md"]\n', encoding="utf-8")
    _forbid_service(monkeypatch)
    monkeypatch.setattr(
        "sys.argv", ["second-brain", "--config", str(tmp_path / "missing.toml"),
                     "eval", "--golden", str(golden), "--graph-boost", "1.5"],
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 2
    assert "graph_boost" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ('"sk-[a-z]+"', "redact_patterns must be an array of strings"),
        ('["sk-[a-z]+", 3]', "redact_patterns must be an array of strings"),
        ('["sk-[a-z]+", "token=(unclosed"]', r"redact_patterns: invalid regex 'token=\(unclosed'"),
    ],
)
def test_load_config_rejects_malformed_redact_patterns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str, message: str
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "rag_config.toml"
    config_path.write_text(
        '\n'.join([
            'vault_path = "$CWD"',
            'qdrant_path = "$CWD/data/qdrant"',
            'fts_path = "./data/fts.sqlite"',
            f"redact_patterns = {value}",
        ]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_config(config_path)


def test_load_config_reads_valid_redact_patterns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "rag_config.toml"
    config_path.write_text(
        (
            'vault_path = "$CWD"\n'
            'qdrant_path = "$CWD/data/qdrant"\n'
            'fts_path = "./data/fts.sqlite"\n'
            "redact_patterns = ['sk-[A-Za-z0-9]+', '\\d{3}-\\d{4}']"
        ),
        encoding="utf-8",
    )

    assert load_config(config_path).redact_patterns == ["sk-[A-Za-z0-9]+", "\\d{3}-\\d{4}"]


def test_load_config_reads_and_defaults_embed_batch_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    base = 'vault_path = "$CWD"\nqdrant_path = "$CWD/data/qdrant"\nfts_path = "./data/fts.sqlite"\n'
    config_path = tmp_path / "rag_config.toml"

    config_path.write_text(base, encoding="utf-8")
    assert load_config(config_path).embed_batch_size == 32

    config_path.write_text(base + "embed_batch_size = 8\n", encoding="utf-8")
    assert load_config(config_path).embed_batch_size == 8


@pytest.mark.parametrize("value", ["0", "-4", '"32"', "true", "2.5"])
def test_load_config_rejects_invalid_embed_batch_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "rag_config.toml"
    config_path.write_text(
        'vault_path = "$CWD"\nqdrant_path = "$CWD/data/qdrant"\nfts_path = "./data/fts.sqlite"\n'
        f"embed_batch_size = {value}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="embed_batch_size"):
        load_config(config_path)
