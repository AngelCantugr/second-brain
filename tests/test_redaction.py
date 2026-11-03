import pytest

from second_brain.models import ParsedNote
from second_brain.parser import extract_note_fields
from second_brain.redaction import REDACTION_TOKEN, Redactor


def test_redactor_replaces_every_match_with_fixed_token() -> None:
    redactor = Redactor([r"sk-[A-Za-z0-9]+", r"\d{3}-\d{4}"])

    assert redactor.text("key sk-abc123 and sk-XYZ, call 555-0199") == (
        f"key {REDACTION_TOKEN} and {REDACTION_TOKEN}, call {REDACTION_TOKEN}"
    )


def test_redactor_ignores_empty_matches() -> None:
    assert Redactor([r"\d*"]).text("pin 1234 ok") == f"pin {REDACTION_TOKEN} ok"


def test_redactor_names_the_invalid_pattern() -> None:
    with pytest.raises(ValueError, match=r"invalid regex '\(oops'"):
        Redactor(["fine", "(oops"])


def test_redactor_merges_overlapping_matches_from_different_patterns() -> None:
    redactor = Redactor([r"secret", r"secret-\w+"])

    assert redactor.text("x secret-hunter2 y") == f"x {REDACTION_TOKEN} y"


def test_redactor_redacts_matching_numbers_in_frontmatter() -> None:
    redactor = Redactor([r"555\d{4}"])

    assert redactor.value({"phone": 5550199, "count": 3, "ok": True}) == {
        "phone": REDACTION_TOKEN,
        "count": 3,
        "ok": True,
    }


def _note(body: str, frontmatter: dict | None = None, title: str = "Note") -> ParsedNote:
    frontmatter = frontmatter or {}
    fields = extract_note_fields(body, frontmatter)
    return ParsedNote(
        note_id="n1",
        path="Ops/secret-sk-path.md",
        title=title,
        body=body,
        frontmatter=frontmatter,
        tags=fields.tags,
        links=fields.links,
        headings=fields.headings,
        tasks=fields.tasks,
        mtime=1.0,
        content_hash="hash",
    )


def test_redactor_covers_note_fields_but_keeps_path_identity() -> None:
    note = _note(
        "\n".join(
            [
                "# Rotate sk-heading",
                "body sk-body #sk-tag [[sk-link]]",
                "- [ ] revoke sk-task",
            ]
        ),
        frontmatter={"api_key": "sk-front", "nested": [{"v": "sk-deep"}], "n": 7},
        title="sk-title",
    )

    redacted = Redactor([r"sk-[a-z]+"]).note(note)

    dumped = repr(
        [redacted.title, redacted.body, redacted.frontmatter, redacted.tags,
         redacted.links, redacted.headings, redacted.tasks]
    )
    assert "sk-" not in dumped
    assert redacted.title == REDACTION_TOKEN
    assert redacted.frontmatter == {
        "api_key": REDACTION_TOKEN,
        "nested": [{"v": REDACTION_TOKEN}],
        "n": 7,
    }
    assert redacted.headings == [f"Rotate {REDACTION_TOKEN}"]
    assert redacted.tasks == [f"revoke {REDACTION_TOKEN}"]
    assert (redacted.path, redacted.note_id, redacted.content_hash) == (
        note.path,
        note.note_id,
        note.content_hash,
    )
    assert note.body.startswith("# Rotate sk-heading")


def test_redactor_rederives_fields_when_pattern_includes_markdown_context() -> None:
    note = _note(
        "Team #private-team met. See [[Secret Project|the plan]] and [[Public]].",
        frontmatter={"tags": ["#private-board", "ops"]},
    )
    assert "private-team" in note.tags and "Secret Project" in note.links

    redacted = Redactor([r"#private-\S+", r"\[\[Secret[^\]]*\]\]"]).note(note)

    assert redacted.tags == [REDACTION_TOKEN, "ops"]
    assert redacted.links == ["Public"]
    assert "private" not in redacted.body and "Secret" not in redacted.body


def test_redactor_without_patterns_returns_note_unchanged() -> None:
    note = ParsedNote("n1", "a.md", "A", "body", {}, [], [], [], [], 1.0, "h")

    assert Redactor([]).note(note) is note
