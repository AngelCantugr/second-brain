from pathlib import Path

from obsidian_rag.chunker import chunk_note
from obsidian_rag.parser import parse_note


def test_chunker_respects_heading_structure_and_overlap(tmp_path: Path) -> None:
    note_path = tmp_path / "Test.md"
    note_path.write_text(
        """# Section A
word1 word2 word3 word4 word5

## Subsection A1
word6 word7 word8 word9 word10

# Section B
word11 word12 word13
""",
        encoding="utf-8",
    )

    parsed = parse_note(note_path)
    chunks = chunk_note(parsed, chunk_size=3, chunk_overlap=1)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.text.strip()  # All chunks have text
        assert chunk.heading_path in ["Section A", "Subsection A1", "Section B", "root"]
        assert chunk.metadata["note_title"] == "Test"


def test_chunker_handles_empty_sections_gracefully(tmp_path: Path) -> None:
    note_path = tmp_path / "Empty.md"
    note_path.write_text(
        """# Section

## Empty subsection

# Another section
Actual content
""",
        encoding="utf-8",
    )

    parsed = parse_note(note_path)
    chunks = chunk_note(parsed, chunk_size=10, chunk_overlap=0)

    assert all(chunk.text.strip() for chunk in chunks)
