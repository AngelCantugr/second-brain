from pathlib import Path

import pytest

from second_brain.banks import FileRef, select_banks
from second_brain.config import load_config


def test_registry_accepts_multiple_roots_and_repo_only(tmp_path: Path) -> None:
    config_path = tmp_path / "rag.toml"
    store_settings = 'qdrant_path = "vec"\nfts_path = "fts.sqlite"\n'
    repo = '[[banks]]\nid = "repo"\ntype = "repo"\npath = "./unmounted"\n'
    config_path.write_text(store_settings + repo, encoding="utf-8")
    assert set(load_config(config_path).banks) == {"repo"}
    config_path.write_text(
        store_settings + repo + '[[banks]]\nid = "notes"\ntype = "obsidian"\npath = "./notes"\n',
        encoding="utf-8",
    )
    banks = load_config(config_path).banks
    assert set(banks) == {"repo", "notes"}
    assert banks["repo"].path == tmp_path / "unmounted"
    assert banks["notes"].name == "notes"
    assert select_banks(banks, None) == {"repo", "notes"}


def test_duplicate_bank_id_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "rag.toml"
    bank = '[[banks]]\nid = "same"\ntype = "obsidian"\npath = "./notes"\n'
    config_path.write_text('qdrant_path = "vec"\nfts_path = "fts.sqlite"\n' + bank * 2)
    with pytest.raises(ValueError, match="same"):
        load_config(config_path)


@pytest.mark.parametrize("selector", [[], ["unknown"], "notes"])
def test_selector_rejects_empty_unknown_and_scalar(selector) -> None:
    with pytest.raises(ValueError):
        select_banks({}, selector)


def test_file_ref_is_case_preserving_and_collision_free() -> None:
    note = FileRef("learning", "Projects/Roadmap.md")
    assert note.to_dict() == {"bank": "learning", "path": "Projects/Roadmap.md"}
    assert note.to_key() != FileRef("other", note.path).to_key()
    punctuation = FileRef("learning", "docs/日本語 [a, b].md")
    assert FileRef.from_key(punctuation.to_key()) == punctuation


@pytest.mark.parametrize("path", ["/note.md", "../note.md", "dir/./note.md", "dir//note.md", "a\\b.md", "a\0b"])
def test_file_ref_rejects_noncanonical_paths(path: str) -> None:
    with pytest.raises(ValueError):
        FileRef("learning", path)
