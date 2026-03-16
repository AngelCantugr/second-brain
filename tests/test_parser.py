from pathlib import Path

from obsidian_rag.parser import parse_note


def test_parse_note_extracts_frontmatter_links_tags_tasks_and_headings(tmp_path: Path) -> None:
    note_path = tmp_path / "Project.md"
    note_path.write_text(
        """---
status: doing
due: 2026-02-28
project: Second Brain
custom_key:
  nested: value
---
# Project Alpha

- [ ] First task #task

Reference [[Daily/2026-02-24]] and [[Project Beta|beta]].

## Next Steps
Ship feature #important
""",
        encoding="utf-8",
    )

    parsed = parse_note(note_path)

    assert parsed.title == "Project"
    assert parsed.frontmatter["status"] == "doing"
    assert parsed.frontmatter["custom_key"]["nested"] == "value"
    assert "task" in parsed.tags
    assert "important" in parsed.tags
    assert parsed.links == ["Daily/2026-02-24", "Project Beta"]
    assert parsed.headings == ["Project Alpha", "Next Steps"]
    assert parsed.tasks == ["First task #task"]


def test_parse_note_handles_malformed_frontmatter_without_failure(tmp_path: Path) -> None:
    note_path = tmp_path / "Broken.md"
    note_path.write_text(
        """---
status: [invalid
---
# Heading
Body
""",
        encoding="utf-8",
    )

    parsed = parse_note(note_path)

    assert parsed.frontmatter == {}
    assert "Body" in parsed.body
