from pathlib import Path

from vault_mcp.config import Settings
from vault_mcp.metadata import iter_markdown_files, parse_note


def test_settings_defaults_exist() -> None:
    settings = Settings()
    assert settings.collection_name == "vault_chunks"


def test_iter_markdown_files_skips_excluded_directories(tmp_path: Path) -> None:
    (tmp_path / "_templates").mkdir()
    (tmp_path / "Archive").mkdir()
    (tmp_path / "Topic").mkdir()

    keep = tmp_path / "Topic" / "keep.md"
    skip_template = tmp_path / "_templates" / "skip.md"
    skip_archive = tmp_path / "Archive" / "skip.md"

    keep.write_text("# keep", encoding="utf-8")
    skip_template.write_text("# skip", encoding="utf-8")
    skip_archive.write_text("# skip", encoding="utf-8")

    found = [p.relative_to(tmp_path).as_posix() for p in iter_markdown_files(tmp_path)]
    assert found == ["Topic/keep.md"]


def test_parse_note_extracts_frontmatter_tags_and_mtime(tmp_path: Path) -> None:
    note_path = tmp_path / "Topic" / "note.md"
    note_path.parent.mkdir()
    note_path.write_text(
        "---\ntags:\n  - ai\n  - mcp\n---\n# Heading\nBody",
        encoding="utf-8",
    )

    note = parse_note(note_path, tmp_path)

    assert note.relative_path == "Topic/note.md"
    assert note.tags == ["ai", "mcp"]
    assert note.content.startswith("# Heading")
    assert note.date_modified > 0
