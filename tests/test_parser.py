import json
from pathlib import Path

from second_brain.parser import derive_metadata, parse_note


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


def test_derive_metadata_captures_plain_date_key_for_daily_journal_notes() -> None:
    from datetime import date, datetime

    assert derive_metadata({"date": "2026-06-16"})["date_date"] == "2026-06-16"
    assert derive_metadata({"date": date(2026, 6, 16)})["date_date"] == "2026-06-16"
    assert (
        derive_metadata({"date": datetime(2026, 6, 16, 10, 0)})["date_date"]
        == "2026-06-16 10:00:00"
    )


def test_parse_note_merges_frontmatter_list_tags(tmp_path: Path) -> None:
    note_path = tmp_path / "Sifter.md"
    note_path.write_text(
        """---
tags: [sifter, metrics]
---
# Sifter Metrics Platform
Body text with #inline tag.
""",
        encoding="utf-8",
    )

    parsed = parse_note(note_path)

    assert "sifter" in parsed.tags
    assert "metrics" in parsed.tags
    assert "inline" in parsed.tags


def test_parse_note_merges_frontmatter_comma_string_tags(tmp_path: Path) -> None:
    note_path = tmp_path / "LangGraph-Patterns.md"
    note_path.write_text(
        """---
tags: langgraph, ai-agents, learning, patterns
---
# LangGraph Patterns
Body text.
""",
        encoding="utf-8",
    )

    parsed = parse_note(note_path)

    assert set(parsed.tags) == {"langgraph", "ai-agents", "learning", "patterns"}


def test_parse_note_ignores_hash_fragments_in_pasted_urls(tmp_path: Path) -> None:
    note_path = tmp_path / "AI-Agents.md"
    note_path.write_text(
        """---
tags: [ai-agents]
---
# AI Agents

Task link: https://ticktick.com/webapp/#q/today/task/6919276a8f088d0075bc5ce9

Another: https://ticktick.com/webapp/#q/today/tasks/69b3936c176651c9b5afedba

Body text with real #inline-tag.
""",
        encoding="utf-8",
    )

    parsed = parse_note(note_path)

    assert "inline-tag" in parsed.tags
    assert "ai-agents" in parsed.tags
    for tag in parsed.tags:
        assert "task" not in tag or tag == "inline-tag"
        assert "/" not in tag
        assert not tag.startswith("q")


def test_parse_note_ignores_purely_numeric_inline_references(tmp_path: Path) -> None:
    note_path = tmp_path / "Strategy.md"
    note_path.write_text(
        """---
---
# Strategy

See item #3 above, and also #1 and #9801 for context.

Real tag here: #priority-1
""",
        encoding="utf-8",
    )

    parsed = parse_note(note_path)

    assert "priority-1" in parsed.tags
    assert "3" not in parsed.tags
    assert "1" not in parsed.tags
    assert "9801" not in parsed.tags


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


def test_parse_note_stringifies_unquoted_frontmatter_dates(tmp_path: Path) -> None:
    note_path = tmp_path / "Daily" / "2026-06-16.md"
    note_path.parent.mkdir(parents=True)
    note_path.write_text(
        """---
date: 2026-06-16
due: 2026-02-28
created: 2026-06-16 08:30:00
nested:
  reviewed: 2026-06-17
2026-06-16: today's log
tags: [journal]
---
# Daily Note
Body text.
""",
        encoding="utf-8",
    )

    parsed = parse_note(note_path)

    assert parsed.frontmatter["date"] == "2026-06-16"
    assert parsed.frontmatter["due"] == "2026-02-28"
    assert parsed.frontmatter["created"] == "2026-06-16T08:30:00"
    assert parsed.frontmatter["2026-06-16"] == "today's log"
    assert parsed.frontmatter["nested"]["reviewed"] == "2026-06-17"
    # json.dumps must not raise for frontmatter carrying unquoted YAML dates.
    json.dumps(parsed.frontmatter, sort_keys=True)
