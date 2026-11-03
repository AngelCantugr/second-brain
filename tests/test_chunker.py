from second_brain.chunker import chunk_note, embedding_text
from second_brain.models import ParsedNote


def test_chunk_note_preserves_heading_path_and_overlap() -> None:
    body = "\n".join(
        [
            "# H1",
            " ".join([f"word{i}" for i in range(250)]),
            "## H2",
            " ".join([f"word{i}" for i in range(250, 500)]),
        ]
    )
    note = ParsedNote(
        note_id="n1",
        path="Project.md",
        title="Project",
        body=body,
        frontmatter={},
        tags=["task"],
        links=["Other"],
        headings=["H1", "H2"],
        tasks=[],
        mtime=1.0,
        content_hash="h1",
    )

    chunks = chunk_note(note, chunk_size=120, chunk_overlap=20)

    assert len(chunks) > 1
    assert chunks[0].heading_path
    assert chunks[0].metadata["path"] == "Project.md"
    first_tokens = chunks[0].text.split()
    second_tokens = chunks[1].text.split()
    assert first_tokens[-20:] == second_tokens[:20]


def test_chunk_note_scopes_tasks_to_their_own_section() -> None:
    body = "\n".join(
        [
            "# H1",
            "- [ ] task in h1",
            "## H2",
            "no checkboxes here",
        ]
    )
    note = ParsedNote(
        note_id="n1",
        path="Project.md",
        title="Project",
        body=body,
        frontmatter={},
        tags=[],
        links=[],
        headings=["H1", "H2"],
        tasks=["task in h1"],
        mtime=1.0,
        content_hash="h1",
    )

    chunks = chunk_note(note, chunk_size=120, chunk_overlap=20)

    h1_chunk = next(c for c in chunks if c.heading_path == "H1")
    h2_chunk = next(c for c in chunks if c.heading_path == "H2")
    assert h1_chunk.metadata["tasks"] == ["task in h1"]
    assert h2_chunk.metadata["tasks"] == []


def test_chunk_note_keeps_fenced_comment_inside_its_real_section() -> None:
    body = "\n".join(
        [
            "# Real Heading",
            "Intro line",
            "```python",
            "# compute delay",
            "def delay():",
            "    return 1",
            "```",
            "After block",
        ]
    )
    note = ParsedNote(
        note_id="n1",
        path="Code.md",
        title="Code",
        body=body,
        frontmatter={},
        tags=[],
        links=[],
        headings=["Real Heading"],
        tasks=[],
        mtime=1.0,
        content_hash="h1",
    )

    chunks = chunk_note(note, chunk_size=500, chunk_overlap=0)

    assert [c.heading_path for c in chunks] == ["Real Heading"]
    assert "\n" in chunks[0].text
    assert "    return" in chunks[0].text
    assert "# compute delay" in chunks[0].text


def test_chunk_note_skips_configured_headings_ignoring_case_and_padding() -> None:
    body = "\n".join(
        [
            "# Daily",
            "Worked on the parser today.",
            "## 🔗 Related Notes",
            "- [[Alpha]]",
            "- [[Beta]]",
            "## Reflection",
            "Learned about chunking.",
        ]
    )
    note = ParsedNote(
        note_id="n1",
        path="Journal.md",
        title="Journal",
        body=body,
        frontmatter={},
        tags=[],
        links=["Alpha", "Beta"],
        headings=["Daily", "🔗 Related Notes", "Reflection"],
        tasks=[],
        mtime=1.0,
        content_hash="h1",
    )

    chunks = chunk_note(
        note, chunk_size=500, chunk_overlap=0, skip_headings=["  🔗 related notes "]
    )

    assert [c.heading_path for c in chunks] == ["Daily", "Reflection"]
    assert all("[[Alpha]]" not in c.text for c in chunks)
    assert chunks[0].metadata["links"] == ["Alpha", "Beta"]


def test_chunk_note_keeps_first_line_indentation_and_skips_leading_blank_lines() -> None:
    note = ParsedNote(
        note_id="n1",
        path="Indented.md",
        title="Indented",
        body="# H\r\n\r\n   \r\n    indented code\r\n    second\r\n",
        frontmatter={},
        tags=[],
        links=[],
        headings=["H"],
        tasks=[],
        mtime=1.0,
        content_hash="h1",
    )

    chunks = chunk_note(note, chunk_size=500, chunk_overlap=0)

    assert [c.text for c in chunks] == ["    indented code\n    second"]


def test_embedding_text_prefixes_note_title_and_heading() -> None:
    note = ParsedNote(
        note_id="n1",
        path="Project.md",
        title="Project",
        body="# Plan\nShip the retrieval fix.\nNo heading here.",
        frontmatter={},
        tags=[],
        links=[],
        headings=["Plan"],
        tasks=[],
        mtime=1.0,
        content_hash="h1",
    )
    heading_chunk = chunk_note(note, chunk_size=500, chunk_overlap=0)[0]
    root_note = ParsedNote(
        note_id="n2",
        path="Root.md",
        title="Root Note",
        body="Plain intro without heading.",
        frontmatter={},
        tags=[],
        links=[],
        headings=[],
        tasks=[],
        mtime=1.0,
        content_hash="h2",
    )
    root_chunk = chunk_note(root_note, chunk_size=500, chunk_overlap=0)[0]

    assert embedding_text(heading_chunk) == "Project > Plan\nShip the retrieval fix.\nNo heading here."
    assert embedding_text(root_chunk) == "Root Note\nPlain intro without heading."
