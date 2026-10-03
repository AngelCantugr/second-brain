from pathlib import Path

from second_brain.scanner import iter_markdown_files


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
