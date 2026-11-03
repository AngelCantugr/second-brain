"""Regex redaction applied to note content before it reaches any index."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Sequence
from typing import Any

from second_brain.models import ParsedNote
from second_brain.parser import extract_note_fields

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

        Every pattern is matched against the original string and overlapping or
        touching spans are merged before replacing. Applying patterns one after
        another would let an earlier, shorter match hide part of a later one:
        ``secret`` then ``secret-\\w+`` on ``secret-hunter2`` would leave
        ``[REDACTED]-hunter2``. Empty matches are skipped: a pattern like ``\\d*``
        matches the empty string at every position and would otherwise insert
        the token between every character.
        """

        spans = sorted(
            match.span()
            for pattern in self._patterns
            for match in pattern.finditer(value)
            if match.end() > match.start()
        )
        if not spans:
            return value

        merged: list[list[int]] = [list(spans[0])]
        for start, end in spans[1:]:
            if start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])

        parts: list[str] = []
        cursor = 0
        for start, end in merged:
            parts.append(value[cursor:start])
            parts.append(REDACTION_TOKEN)
            cursor = end
        parts.append(value[cursor:])
        return "".join(parts)

    def value(self, value: Any) -> Any:
        """Redact nested frontmatter values, keeping keys.

        Numbers are checked through their string form, since YAML turns an
        unquoted ``phone: 5550199`` into an int; a matching number becomes the
        redacted string. Booleans and ``None`` are left alone.
        """

        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, dict):
            return {key: self.value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.value(item) for item in value]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            redacted = self.text(str(value))
            return value if redacted == str(value) else redacted
        return value

    def note(self, note: ParsedNote) -> ParsedNote:
        """Return a copy of ``note`` with every indexed free-text field redacted.

        Redaction runs on the whole note before chunking, so a multi-word match
        cannot be split across two chunk windows and escape the pattern, and
        everything derived later (chunk text, embedding header, chunk metadata,
        FTS title/heading columns, graph note metadata) sees redacted values.

        Links, tags, headings and tasks are re-extracted from the redacted body
        and frontmatter rather than redacted in place. The parser strips
        context such as ``#`` or ``[[ ]]`` while extracting, so a pattern like
        ``#private-\\S+`` would match the body but not the extracted tag.

        Redacted: body, title, frontmatter values, and the links, tags,
        headings and tasks derived from them. Not redacted: the vault-relative
        path, note id, content hash and mtime. The path is the identity key for
        sync state, deletes and the graph, so rewriting it would break
        reindexing; notes whose file names are sensitive belong in
        ``exclude_globs`` instead.
        """

        if not self._patterns:
            return note
        body = self.text(note.body)
        frontmatter = self.value(note.frontmatter)
        fields = extract_note_fields(body, frontmatter)
        # File links are not in the body, so re-extraction cannot see them;
        # redact them in place, as a matching wikilink target would be.
        file_links = [self.text(link) for link in note.file_links]
        sections = (
            None
            if note.sections is None
            else [(self.text(heading), self.text(text)) for heading, text in note.sections]
        )
        return dataclasses.replace(
            note,
            title=self.text(note.title),
            body=body,
            frontmatter=frontmatter,
            tags=fields.tags,
            links=sorted(set(fields.links) | set(file_links)) if file_links else fields.links,
            headings=fields.headings,
            tasks=fields.tasks,
            sections=sections,
            file_links=file_links,
            link_pairs=[(self.text(a), self.text(b)) for a, b in note.link_pairs],
        )
