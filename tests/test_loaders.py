"""Loader registry, include_extensions config, scanner extensions, and the kind filter."""

from pathlib import Path

import pytest

from second_brain.config import RagConfig, load_config
from second_brain.keyword_store import matches_filters
from second_brain.loaders import (
    CanvasLoader,
    MarkdownLoader,
    load_source,
    loader_for,
)
from second_brain.parser import parse_note
from second_brain.scanner import is_eligible_source_path, iter_source_files


def test_loader_for_picks_by_included_extension() -> None:
    both = [".md", ".canvas"]

    assert isinstance(loader_for("notes/a.md", both), MarkdownLoader)
    assert isinstance(loader_for("boards/b.canvas", both), CanvasLoader)
    assert loader_for("boards/b.canvas", [".md"]) is None
    assert loader_for("doc.pdf", [".md", ".pdf"]) is None  # included but no loader
    assert loader_for("README", both) is None
    # Suffixes match exactly, as the scanner always has for ``.md``.
    assert loader_for("UPPER.MD", both) is None


def test_markdown_loader_matches_parse_note(tmp_path: Path) -> None:
    note = tmp_path / "Folder" / "Note.md"
    note.parent.mkdir()
    note.write_text("---\ntags: [x]\n---\n# Title\nbody [[Other]] #y\n", encoding="utf-8")

    loaded = load_source(note, tmp_path, [".md"])

    assert loaded == parse_note(note, tmp_path)
    assert loaded.kind == "markdown"
    assert loaded.sections is None
    assert loaded.file_links == [] and loaded.link_pairs == []


def test_load_source_rejects_excluded_extension(tmp_path: Path) -> None:
    board = tmp_path / "b.canvas"
    board.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="no loader for .canvas"):
        load_source(board, tmp_path, [".md"])


def _write_config(tmp_path: Path, extra: str = "") -> Path:
    config = tmp_path / "rag_config.toml"
    config.write_text(
        'vault_path = "./vault"\nqdrant_path = "./q"\nfts_path = "./f.sqlite"\n' + extra,
        encoding="utf-8",
    )
    return config


def test_include_extensions_defaults_to_markdown_only(tmp_path: Path) -> None:
    assert load_config(_write_config(tmp_path)).include_extensions == [".md"]
    assert RagConfig(Path("v"), Path("q"), Path("f"), Path("s")).include_extensions == [".md"]


def test_include_extensions_normalizes_and_dedupes(tmp_path: Path) -> None:
    config = load_config(
        _write_config(tmp_path, 'include_extensions = ["md", ".CANVAS", ".md"]\n')
    )

    assert config.include_extensions == [".md", ".canvas"]


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ('[".pdf"]', "no loader for '.pdf'; supported: .canvas, .md"),
        ("[]", "at least one extension"),
        ('".md"', "must be an array of strings"),
    ],
)
def test_include_extensions_rejects_bad_values(tmp_path: Path, value: str, message: str) -> None:
    with pytest.raises(ValueError, match=message.replace(".", r"\.")):
        load_config(_write_config(tmp_path, f"include_extensions = {value}\n"))


def test_scanner_finds_only_included_extensions(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "b.canvas").write_text("{}", encoding="utf-8")
    (tmp_path / "c.txt").write_text("c", encoding="utf-8")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "d.canvas").write_text("{}", encoding="utf-8")

    assert iter_source_files(tmp_path, []) == [tmp_path / "a.md"]
    assert iter_source_files(tmp_path, [], [".md", ".canvas"]) == [
        tmp_path / "a.md",
        tmp_path / "b.canvas",
    ]
    assert is_eligible_source_path("b.canvas", [], [".canvas"])
    assert not is_eligible_source_path("b.canvas", ["*.canvas"], [".canvas"])
    assert not is_eligible_source_path("b.canvas", [])


@pytest.mark.parametrize(
    ("metadata", "kind", "expected"),
    [
        ({"kind": "canvas"}, "canvas", True),
        ({"kind": "canvas"}, "markdown", False),
        ({"kind": "canvas"}, ["markdown", "canvas"], True),
        ({"kind": "markdown"}, [], False),
        # Chunks indexed before kind existed are markdown.
        ({}, "markdown", True),
        ({}, "canvas", False),
    ],
)
def test_kind_filter(metadata: dict, kind, expected: bool) -> None:
    assert matches_filters(metadata, {"kind": kind}) is expected


@pytest.mark.parametrize("bad", [3, {"kind": "canvas"}, ["canvas", 1]])
def test_kind_filter_rejects_non_string_values(bad) -> None:
    with pytest.raises(ValueError, match="filters\\['kind'\\]"):
        matches_filters({"kind": "canvas"}, {"kind": bad})
