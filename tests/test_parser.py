import json
from pathlib import Path

from second_brain.parser import (
    derive_metadata,
    extract_note_fields,
    frontmatter_aliases,
    mask_code,
    parse_note,
)


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


def _assert_mask_keeps_shape(text: str, masked: str) -> None:
    assert len(masked) == len(text)
    assert [i for i, ch in enumerate(masked) if ch == "\n"] == [
        i for i, ch in enumerate(text) if ch == "\n"
    ]


def test_mask_code_blanks_backtick_fence_contents_and_keeps_newlines() -> None:
    text = "before\n```python\n# compute delay\nx = 1\n```\n# real heading\n"

    masked = mask_code(text)

    _assert_mask_keeps_shape(text, masked)
    assert "compute delay" not in masked
    assert "x = 1" not in masked
    assert "# real heading" in masked


def test_mask_code_blanks_tilde_fence_contents() -> None:
    text = "~~~\n# inside tilde\n~~~\n# outside\n"

    masked = mask_code(text)

    _assert_mask_keeps_shape(text, masked)
    assert "inside tilde" not in masked
    assert "# outside" in masked


def test_mask_code_four_backtick_fence_ignores_three_backtick_lines() -> None:
    text = "````\n```\n# inside\n````\n# outside\n"

    masked = mask_code(text)

    _assert_mask_keeps_shape(text, masked)
    assert "# inside" not in masked
    assert "# outside" in masked


def test_mask_code_unclosed_fence_extends_to_end_of_text() -> None:
    text = "```\n# inside\nstill code\n"

    masked = mask_code(text)

    _assert_mask_keeps_shape(text, masked)
    assert "inside" not in masked
    assert "still code" not in masked


def test_mask_code_blanks_fence_nested_in_list_item() -> None:
    text = (
        "- item\n  - sub\n    ```mermaid\n    style A fill:#e1f5ff\n"
        "    #include <x>\n    ```\n# after\n"
    )

    masked = mask_code(text)

    _assert_mask_keeps_shape(text, masked)
    assert "e1f5ff" not in masked
    assert "#include" not in masked
    assert "# after" in masked


def test_mask_code_blanks_fence_inside_blockquote() -> None:
    text = "> ```py\n> # comment\n> ```\n> > ```\n> > #fff\n> > ```\n# after\n"

    masked = mask_code(text)

    _assert_mask_keeps_shape(text, masked)
    assert "# comment" not in masked
    assert "#fff" not in masked
    assert "# after" in masked


def test_mask_code_blanks_fence_inside_obsidian_callout() -> None:
    text = "> [!note]\n> ```py\n> # comment\n> ```\n> after fence #kept\n"

    masked = mask_code(text)

    _assert_mask_keeps_shape(text, masked)
    assert "# comment" not in masked
    assert "[!note]" in masked
    assert "#kept" in masked


def test_mask_code_ends_unclosed_blockquote_fence_when_quote_ends() -> None:
    text = "> ```python\n> print(1)\n\n# Real section\n[[RealLink]] #real-tag\n"

    masked = mask_code(text)

    _assert_mask_keeps_shape(text, masked)
    assert "print(1)" not in masked
    assert "# Real section" in masked
    assert "[[RealLink]] #real-tag" in masked


def test_mask_code_unquoted_fence_does_not_close_blockquote_fence() -> None:
    # The quote ends at the unquoted line, which then opens its own fence.
    text = "> ```\n> #quoted\n```\n#top-code\n```\n#after\n"

    masked = mask_code(text)

    _assert_mask_keeps_shape(text, masked)
    assert "#quoted" not in masked
    assert "#top-code" not in masked
    assert "#after" in masked


def test_mask_code_deeper_quote_fence_line_does_not_close_fence() -> None:
    text = "> ```\n> > ```\n> #still-code\n> ```\n#after\n"

    masked = mask_code(text)

    _assert_mask_keeps_shape(text, masked)
    assert "#still-code" not in masked
    assert "#after" in masked


def test_mask_code_list_fence_keeps_masking_code_lines_that_start_with_quote() -> None:
    # "> " inside the code is a REPL prompt, not a blockquote ending the item.
    text = "* item\n\n  ```\n  > run('#x')\n  {\n    _: '#y'\n  }\n  ```\n#after\n"

    masked = mask_code(text)

    _assert_mask_keeps_shape(text, masked)
    assert "#x" not in masked
    assert "#y" not in masked
    assert "#after" in masked


def test_mask_code_recognizes_fence_opened_on_list_marker_line() -> None:
    text = "- ```python\n  print('#x')\n  ```\n\n  Reference [[RealLink]] #real-tag\n"

    masked = mask_code(text)

    _assert_mask_keeps_shape(text, masked)
    assert "#x" not in masked
    assert "Reference [[RealLink]] #real-tag" in masked


def test_mask_code_list_marker_fence_ends_with_its_item() -> None:
    text = "> 1. ```\n>    #inside\n> #after\n"

    masked = mask_code(text)

    _assert_mask_keeps_shape(text, masked)
    assert "#inside" not in masked
    assert "#after" in masked


def test_mask_code_ends_unclosed_list_fence_at_dedented_line() -> None:
    text = "- item\n  ```sh\n  echo #x\n\n  echo #y\n# Real section\n- [ ] real task\n"

    masked = mask_code(text)

    _assert_mask_keeps_shape(text, masked)
    assert "#x" not in masked
    assert "#y" not in masked
    assert "# Real section" in masked
    assert "- [ ] real task" in masked


def test_mask_code_indented_top_level_fence_still_masks_unindented_lines() -> None:
    # No list item precedes it, so the fence belongs to the document.
    text = "Para\n\n  ```\n#inside\n  ```\n#after\n"

    masked = mask_code(text)

    _assert_mask_keeps_shape(text, masked)
    assert "#inside" not in masked
    assert "#after" in masked


def test_parse_note_keeps_metadata_after_unclosed_blockquote_fence(tmp_path: Path) -> None:
    note_path = tmp_path / "Quoted.md"
    note_path.write_text(
        "> ```python\n> print(1)\n\n# Real section\n[[RealLink]] #real-tag\n- [ ] real task\n",
        encoding="utf-8",
    )

    parsed = parse_note(note_path)

    assert "Real section" in parsed.headings
    assert "RealLink" in parsed.links
    assert "real-tag" in parsed.tags
    assert "real task" in parsed.tasks


def test_parse_note_ignores_syntax_in_nested_list_fence(tmp_path: Path) -> None:
    note_path = tmp_path / "Nested.md"
    note_path.write_text(
        "- item\n  - sub\n    ```mermaid\n    style A fill:#e1f5ff\n"
        "    #include <x>\n    ```\n",
        encoding="utf-8",
    )

    parsed = parse_note(note_path)

    assert "e1f5ff" not in parsed.tags
    assert "include" not in parsed.tags


def test_mask_code_blanks_inline_code_spans() -> None:
    text = "use `#fff` and ``x # y`` here\n"

    masked = mask_code(text)

    _assert_mask_keeps_shape(text, masked)
    assert "#fff" not in masked
    assert "# y" not in masked
    assert "use" in masked
    assert "here" in masked


def test_mask_code_leaves_unmatched_backtick_intact() -> None:
    text = "a ` b #tag\n"

    assert mask_code(text) == text


def test_parse_note_ignores_code_fence_and_inline_code_syntax(tmp_path: Path) -> None:
    note_path = tmp_path / "Code.md"
    note_path.write_text(
        """# Real Heading

- [ ] real task #outside-tag

Reference [[RealLink]].

```python
# compute delay
#include <x>
[[NotALink]]
- [ ] not a task
```

```mermaid
style A fill:#e1f5ff
```

Inline `#fff` here.
""",
        encoding="utf-8",
    )

    parsed = parse_note(note_path)

    assert parsed.headings == ["Real Heading"]
    assert parsed.tasks == ["real task #outside-tag"]
    assert parsed.links == ["RealLink"]
    assert "outside-tag" in parsed.tags
    for fake_tag in ("include", "e1f5ff", "fff"):
        assert fake_tag not in parsed.tags
    assert parsed.body.count("# compute delay") == 1


def test_frontmatter_aliases_accepts_list_string_and_legacy_key() -> None:
    assert frontmatter_aliases({"aliases": ["K8s", " Kube ", None, "", "K8s"]}) == ["K8s", "Kube"]
    assert frontmatter_aliases({"aliases": "k8s, kube"}) == ["k8s", "kube"]
    assert frontmatter_aliases({"alias": "Solo"}) == ["Solo"]
    assert frontmatter_aliases({"aliases": 3}) == []
    assert frontmatter_aliases({}) == []


def test_heading_links_and_html_entities_are_not_tags() -> None:
    """``#`` inside a wikilink or a numeric character reference is not a tag."""

    body = (
        "See [[#Summary]], [[Note#Details]] and [[Note|#1 pick]].\n"
        "It&#x27;s &#39;quoted&#39; and tagged #real-tag.\n"
    )

    fields = extract_note_fields(body, {})

    assert fields.tags == ["real-tag"]
    assert fields.links == ["#Summary", "Note#Details", "Note"]


def test_tag_after_a_wikilink_whose_alias_is_a_url_is_kept() -> None:
    """Masking the URL first ate the link's ``]]``, so it ran on to the next link."""

    body = "[[Source|https://example.com]]\n#keep\n[[Other]] and https://example.com/page#frag\n"

    fields = extract_note_fields(body, {})

    assert fields.tags == ["keep"]


def test_wikilink_alias_with_escaped_pipe_in_a_table_keeps_a_clean_target() -> None:
    """Obsidian requires ``\\|`` for an alias inside a table cell."""

    body = "| col |\n| --- |\n| [[Target\\|shown]] and [[Folder/Other#Part\\|x]] |\n"

    fields = extract_note_fields(body, {})

    assert fields.links == ["Target", "Folder/Other#Part"]


def test_wikilink_without_alias_keeps_a_backslash_in_its_target() -> None:
    assert extract_note_fields("[[a\\b]] and [[Plain|alias]]", {}).links == ["a\\b", "Plain"]
