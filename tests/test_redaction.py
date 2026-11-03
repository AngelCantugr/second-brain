import pytest

from second_brain.models import ParsedNote
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


def test_redactor_covers_note_fields_but_keeps_path_identity() -> None:
    note = ParsedNote(
        note_id="n1",
        path="Ops/secret-sk-path.md",
        title="sk-title1",
        body="body sk-body1",
        frontmatter={"api_key": "sk-front1", "nested": [{"v": "sk-deep1"}], "n": 7},
        tags=["sk-tag1"],
        links=["sk-link1"],
        headings=["sk-heading1"],
        tasks=["- [ ] rotate sk-task1"],
        mtime=1.0,
        content_hash="hash",
    )

    redacted = Redactor([r"sk-[a-z]+1"]).note(note)

    assert redacted.title == REDACTION_TOKEN
    assert redacted.body == f"body {REDACTION_TOKEN}"
    assert redacted.frontmatter == {
        "api_key": REDACTION_TOKEN,
        "nested": [{"v": REDACTION_TOKEN}],
        "n": 7,
    }
    assert redacted.tags == [REDACTION_TOKEN]
    assert redacted.links == [REDACTION_TOKEN]
    assert redacted.headings == [REDACTION_TOKEN]
    assert redacted.tasks == [f"- [ ] rotate {REDACTION_TOKEN}"]
    assert (redacted.path, redacted.note_id, redacted.content_hash) == (
        note.path,
        note.note_id,
        note.content_hash,
    )
    assert note.body == "body sk-body1"


def test_redactor_without_patterns_returns_note_unchanged() -> None:
    note = ParsedNote("n1", "a.md", "A", "body", {}, [], [], [], [], 1.0, "h")

    assert Redactor([]).note(note) is note
