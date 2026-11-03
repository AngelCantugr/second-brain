"""Regex redaction applied to note content before it reaches any index."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Sequence
from typing import Any

from second_brain.models import ParsedNote

REDACTION_TOKEN = "[REDACTED]"


def compile_redact_patterns(patterns: Sequence[str]) -> list[re.Pattern[str]]:
    """Compile every pattern up front so a typo fails at startup, not mid-sync."""

    compiled: list[re.Pattern[str]] = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern))
        except re.error as exc:
            raise ValueError(
                f"redact_patterns: invalid regex {pattern!r}: {exc}"
            ) from exc
    return compiled


class Redactor:
    """Replace every match of the configured patterns with ``REDACTION_TOKEN``."""

    def __init__(self, patterns: Sequence[str]) -> None:
        self._patterns = compile_redact_patterns(patterns)

    def text(self, value: str) -> str:
        """Redact one string.

        Empty matches are left alone: a pattern like ``\\d*`` matches the empty
        string at every position and would otherwise insert the token between
        every character.
        """

        for pattern in self._patterns:
            value = pattern.sub(
                lambda match: REDACTION_TOKEN if match.group(0) else "", value
            )
        return value

    def value(self, value: Any) -> Any:
        """Redact strings inside nested frontmatter values, keeping keys and non-strings."""

        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, dict):
            return {key: self.value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.value(item) for item in value]
        return value

    def note(self, note: ParsedNote) -> ParsedNote:
        """Return a copy of ``note`` with every indexed free-text field redacted.

        Redaction runs on the whole note before chunking, so a multi-word match
        cannot be split across two chunk windows and escape the pattern, and
        everything derived later (chunk text, embedding header, chunk metadata,
        FTS title/heading columns, graph note metadata) sees redacted values.

        Redacted: body, title, frontmatter string values, tags, links, headings
        and tasks. Not redacted: the vault-relative path, note id, content hash
        and mtime. The path is the identity key for sync state, deletes and the
        graph, so rewriting it would break reindexing; notes whose file names
        are sensitive belong in ``exclude_globs`` instead.
        """

        if not self._patterns:
            return note
        return dataclasses.replace(
            note,
            title=self.text(note.title),
            body=self.text(note.body),
            frontmatter=self.value(note.frontmatter),
            tags=[self.text(tag) for tag in note.tags],
            links=[self.text(link) for link in note.links],
            headings=[self.text(heading) for heading in note.headings],
            tasks=[self.text(task) for task in note.tasks],
        )
