from pathlib import Path

import pytest

from second_brain.config import DEFAULT_EXCLUDE_GLOBS
from second_brain.scanner import iter_markdown_files, path_is_excluded


def test_default_patterns_exclude_root_and_nested_agent_instructions_and_templates(
    tmp_path: Path,
) -> None:
    paths = [
        "CLAUDE.md",
        "AGENTS.md",
        "GEMINI.md",
        "nested/CLAUDE.md",
        "_types/Person.md",
        "_templates/Meeting.md",
        "_templates-old/Meeting.md",
        "nested/_templates-old/Meeting.md",
        "nested/keep.md",
    ]
    for relative_path in paths:
        target = tmp_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("content", encoding="utf-8")

    files = iter_markdown_files(
        tmp_path,
        [
            ".obsidian/**", ".git/**", "Templates/**", "_types/**",
            "_templates/**", "**/_templates*/**", "CLAUDE.md", "AGENTS.md", "GEMINI.md",
        ],
    )

    assert [path.relative_to(tmp_path).as_posix() for path in files] == [
        "nested/keep.md"
    ]


def test_explicit_empty_globs_include_agent_instruction_files(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("instructions", encoding="utf-8")

    assert iter_markdown_files(tmp_path, []) == [tmp_path / "AGENTS.md"]


@pytest.mark.parametrize(
    "relative_path",
    [
        "_templates/Meeting.md",
        "_templates-old/deep/Meeting.md",
        "nested/_templates-old/deep/Meeting.md",
        "nested/_templates/deep/Meeting.md",
        "alpha/beta/_templates-kit/Meeting.md",
    ],
)
def test_recursive_template_default_matches_root_and_nested_subtrees(
    relative_path: str,
) -> None:
    assert path_is_excluded(relative_path, list(DEFAULT_EXCLUDE_GLOBS))


@pytest.mark.parametrize(
    "relative_path",
    [
        "Templates-notes/keep.md",
        "_types-extra/keep.md",
        "nested/templates-old/deep/Meeting.md",
        "nested/_templating/deep/Meeting.md",
    ],
)
def test_default_patterns_keep_similarly_named_regular_paths(relative_path: str) -> None:
    assert not path_is_excluded(relative_path, list(DEFAULT_EXCLUDE_GLOBS))


def test_scanner_and_path_helper_agree_on_recursive_template_paths(tmp_path: Path) -> None:
    relative_paths = [
        "_templates-old/deep/Meeting.md",
        "nested/_templates/deep/Meeting.md",
        "Templates-notes/keep.md",
        "_types-extra/keep.md",
    ]
    for relative_path in relative_paths:
        target = tmp_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("content", encoding="utf-8")

    helper_exclusions = {
        relative_path
        for relative_path in relative_paths
        if path_is_excluded(relative_path, list(DEFAULT_EXCLUDE_GLOBS))
    }
    scanner_inclusions = {
        path.relative_to(tmp_path).as_posix()
        for path in iter_markdown_files(tmp_path, list(DEFAULT_EXCLUDE_GLOBS))
    }

    assert helper_exclusions == {"_templates-old/deep/Meeting.md", "nested/_templates/deep/Meeting.md"}
    assert scanner_inclusions == set(relative_paths) - helper_exclusions
