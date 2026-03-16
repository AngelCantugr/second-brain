from pathlib import Path

import pytest

from obsidian_rag.cli import _write_init_config
from obsidian_rag.config import load_config


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
        "\n".join(
            [
                'vault_path = "$CWD"',
                'qdrant_path = "$CWD/data/qdrant"',
                'fts_path = "./data/fts.sqlite"',
                'sync_state_path = "./data/sync_state.sqlite"',
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.vault_path == workspace
    assert config.qdrant_path == workspace / "data" / "qdrant"
    assert config.fts_path == config_dir / "data" / "fts.sqlite"
    assert config.sync_state_path == config_dir / "data" / "sync_state.sqlite"


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
