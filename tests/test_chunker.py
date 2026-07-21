from second_brain.chunker import chunk_note
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
