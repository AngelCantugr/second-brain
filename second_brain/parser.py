"""Obsidian note parsing utilities.

This module extracts:
- YAML frontmatter
- wiki links
- tags
- headings
- checklist tasks
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from second_brain.models import ParsedNote

# The target is matched lazily so the backslash of an escaped alias pipe
# (``[[Target\|alias]]``, required inside a table cell) stays out of it.
WIKILINK_RE = re.compile(r"\[\[([^\]|]+?)(?:\\?\|[^\]]+)?\]\]")
# ``&`` is excluded so a character reference such as ``&#x27;`` is not a tag.
TAG_RE = re.compile(r"(?<![\w&])#([A-Za-z0-9_/-]+)")
URL_RE = re.compile(r"https?://\S+")
GITHUB_URL_RE = re.compile(r"(https?://github\.com/[\w.-]+/[\w.-]+(?:/(?:blob|tree)/[^\s#?)\]>]+)?)")
# A path written in prose: at least one folder, then a source or markdown file.
CODE_PATH_RE = re.compile(
    r"(?<![\w/.:-])((?:[\w.-]+/)+[\w.-]+\.(?:py|pyi|go|ts|tsx|mts|cts|md))(?![\w/])"
)
_MAX_CODE_MENTION = 200
_MAX_CODE_MENTIONS = 500
HEADING_RE = re.compile(r"^#{1,6}\s+(.*)$", re.MULTILINE)
TASK_RE = re.compile(r"^\s*[-*]\s+\[(?: |x|X)\]\s+(.*)$", re.MULTILINE)

# Matched after blockquote and list markers are stripped. A fence may sit at
# any indent (nested list items put fences 4+ spaces in). CommonMark's "at
# most 3 spaces" rule is relaxed on purpose: missing a list-nested fence leaks
# code syntax into tags, while treating a rare indented-code line that starts
# with ``` as a fence only blanks extra text.
FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})(.*)$")
QUOTE_MARKER_RE = re.compile(r"[ \t]*>[ \t]?")
LIST_ITEM_RE = re.compile(r"^[ \t]*(?:[-*+]|\d{1,9}[.)])[ \t]+")
# Opener and closer must each be a whole backtick run of the same length.
# Without the opener's (?!`), the regex backtracks to a shorter run and
# matches inside ``a` (a two-backtick opener with no closer).
INLINE_CODE_RE = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)")
# Exactly the characters str.splitlines() breaks on. Masking must keep these,
# or masked and original text would split into different line lists.
_LINE_BREAK_CHARS = frozenset("\n\r\x0b\x0c\x1c\x1d\x1e\x85  ")


def mask_code(text: str) -> str:
    """Blank fenced code blocks and inline code so syntax regexes see only prose.

    Code can contain ``# comment``, ``#fff`` or ``[[x]]`` that look like
    markdown syntax. Blanking it (instead of removing it) keeps every offset
    and line the same as the original, so a match in the masked copy can be
    reported from the original text.
    """

    masked_lines: list[str] = []
    open_fence: _OpenFence | None = None
    # Content indent of the latest list item, as (quote depth, indent). A
    # fence indented at least this far belongs to that item.
    list_item: tuple[int, int] | None = None
    prev_blank = True

    for line in text.splitlines(keepends=True):
        # Inside a fence, a ``>`` past the fence's own quote depth is code
        # (a REPL prompt, a diff), not a deeper blockquote.
        max_depth = open_fence.depth if open_fence is not None else None
        depth, rest = _split_quote(line.rstrip(), max_depth)
        indent = _indent(rest)
        blank = not rest.strip()

        # CommonMark gives fenced code no lazy continuation: when the quote
        # or list item holding an unclosed fence ends, the fence ends too.
        if open_fence is not None and open_fence.container_ended(depth, indent, blank):
            open_fence = None

        if open_fence is not None:
            # Match the closer on the text after this fence's own markers, so
            # a deeper ``> ```` line inside the code does not close it.
            fence = FENCE_RE.match(rest)
            if fence and _closes_fence(fence, open_fence):
                open_fence = None
            masked_lines.append(_blank(line))
            prev_blank = blank
            continue

        list_item = _track_list_item(list_item, depth, rest, indent, blank, prev_blank)
        # A fence can open on the list marker's own line (``- ```py``).
        marker = LIST_ITEM_RE.match(rest)
        fence = FENCE_RE.match(rest[marker.end() :] if marker else rest)
        if fence and _may_open_fence(fence):
            container_indent = 0
            if list_item is not None and list_item[0] == depth and (marker or indent >= list_item[1]):
                container_indent = list_item[1]
            open_fence = _OpenFence(
                fence.group(1)[0], len(fence.group(1)), depth, container_indent
            )
            masked_lines.append(_blank(line))
        else:
            masked_lines.append(_mask_inline_code(line))
        prev_blank = blank

    return "".join(masked_lines)


@dataclass(frozen=True, slots=True)
class _OpenFence:
    char: str
    length: int
    depth: int
    container_indent: int

    def container_ended(self, depth: int, indent: int, blank: bool) -> bool:
        if depth < self.depth:
            return True
        return not blank and indent < self.container_indent


def _split_quote(line: str, max_depth: int | None = None) -> tuple[int, str]:
    """Return ``(blockquote depth, text after the markers)`` for one line.

    ``max_depth`` stops after that many markers, leaving any further ``>``
    in the returned text.
    """

    depth = 0
    pos = 0
    while max_depth is None or depth < max_depth:
        marker = QUOTE_MARKER_RE.match(line, pos)
        if not marker:
            break
        depth += 1
        pos = marker.end()
    return depth, line[pos:]


def _indent(text: str) -> int:
    return len(text) - len(text.lstrip(" \t"))


def _track_list_item(
    current: tuple[int, int] | None,
    depth: int,
    rest: str,
    indent: int,
    blank: bool,
    prev_blank: bool,
) -> tuple[int, int] | None:
    item = LIST_ITEM_RE.match(rest)
    if item:
        return depth, len(item.group(0))
    if current is None or blank:
        return current
    # A dedented line after a blank line, or at another quote depth, ends
    # the item. Without a blank line it is a lazy paragraph continuation.
    if depth != current[0] or (prev_blank and indent < current[1]):
        return None
    return current


def findall_outside_code(pattern: re.Pattern[str], masked: str, original: str) -> list[str]:
    """Match on masked text but return group 1 from the original text.

    Matching masked text keeps code out of the results. Reading the original
    keeps code the author wrote in a heading or task, such as the ``foo()``
    in ``- [ ] fix `foo()` ``, instead of blanks.
    """

    return [original[m.start(1) : m.end(1)] for m in pattern.finditer(masked)]


def _blank(text: str) -> str:
    return "".join(ch if ch in _LINE_BREAK_CHARS else " " for ch in text)


def _mask_inline_code(line: str) -> str:
    return INLINE_CODE_RE.sub(
        lambda m: m.group(1) + _blank(m.group(2)) + m.group(1),
        line,
    )


def _may_open_fence(fence: re.Match[str]) -> bool:
    # CommonMark: a backtick fence's info string cannot contain a backtick.
    # Without this, a line like ```foo``` would open a fence instead of being
    # inline code.
    return fence.group(1)[0] == "~" or "`" not in fence.group(2)


def _closes_fence(fence: re.Match[str], open_fence: _OpenFence) -> bool:
    run = fence.group(1)
    return (
        run[0] == open_fence.char
        and len(run) >= open_fence.length
        and not fence.group(2).strip()
    )


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Return ``(frontmatter_dict, body_text)`` from markdown input.

    Frontmatter parsing is intentionally tolerant. If parsing fails,
    the function returns an empty metadata map and the original text body.
    """

    if not text.startswith("---\n"):
        return {}, text

    try:
        _, rest = text.split("---\n", 1)
        yaml_text, body = rest.split("\n---\n", 1)
    except ValueError:
        return {}, text

    try:
        loaded = yaml.safe_load(yaml_text)
        if not isinstance(loaded, dict):
            loaded = {}
    except Exception:
        loaded = {}

    return _json_safe_frontmatter(loaded), body


def _json_safe_frontmatter(value: Any) -> Any:
    """Recursively convert YAML-inferred ``date``/``datetime`` values to ISO strings.

    PyYAML's safe loader parses unquoted frontmatter dates (e.g. ``date: 2026-06-16``)
    into native ``date``/``datetime`` objects, which ``json.dumps`` cannot serialize.
    Sanitizing here keeps ``raw_frontmatter`` JSON-safe everywhere it's consumed.
    """

    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {
            _json_safe_frontmatter(key): _json_safe_frontmatter(v)
            for key, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_frontmatter(v) for v in value]
    return value


def _frontmatter_tags(frontmatter: dict[str, Any]) -> list[str]:
    """Normalize the frontmatter ``tags`` field into a flat list of strings.

    Obsidian allows frontmatter tags as either a YAML list or a single
    comma-separated string, so both forms are accepted here.
    """

    raw = frontmatter.get("tags")
    if raw is None:
        return []
    if isinstance(raw, str):
        return [t.strip().lstrip("#") for t in raw.split(",") if t.strip().lstrip("#")]
    if isinstance(raw, (list, tuple, set)):
        return [
            str(t).strip().lstrip("#")
            for t in raw
            if t is not None and str(t).strip().lstrip("#")
        ]
    return []


def frontmatter_aliases(frontmatter: dict[str, Any]) -> list[str]:
    """Return a note's alternative names from ``aliases`` (or legacy ``alias``).

    Accepts a YAML list or one comma-separated string, as Obsidian does, so
    the graph can resolve ``[[Alias]]`` to the note that declares it.
    """

    raw = frontmatter.get("aliases", frontmatter.get("alias"))
    if isinstance(raw, str):
        raw = raw.split(",")
    if not isinstance(raw, (list, tuple, set)):
        return []
    return sorted({str(a).strip() for a in raw if a is not None and str(a).strip()})


def _inline_tags(body: str) -> set[str]:
    """Extract genuine Obsidian ``#tags`` from note body text.

    URLs and wikilinks are masked out first so ``#`` fragments (TickTick task
    links, GitHub anchors, ``[[Note#Heading]]``) never get mistaken for tags. Purely numeric
    matches (``#1``, ``#3``) are also dropped, matching Obsidian's own
    tag rule that a tag must contain at least one non-digit character —
    otherwise inline prose references like "see item #3" pollute tags.
    """

    # A wikilink's ``#`` introduces a heading (``[[#Summary]]``), not a tag.
    prose = WIKILINK_RE.sub(" ", URL_RE.sub(" ", body))
    return {
        match
        for match in TAG_RE.findall(prose)
        if not match.isdigit()
    }


def derive_metadata(frontmatter: dict[str, Any]) -> dict[str, Any]:
    """Compute normalized, query-friendly metadata fields.

    Raw frontmatter is kept as-is elsewhere; these are best-effort
    derived fields used for consistent filtering.
    """

    derived: dict[str, Any] = {}
    date_keys = ("due", "deadline", "start", "created", "date")
    for key in date_keys:
        value = frontmatter.get(key)
        if isinstance(value, (str, date, datetime)):
            derived[f"{key}_date"] = str(value)

    status = frontmatter.get("status")
    if status is not None:
        derived["status"] = str(status).lower()

    for key in ("project", "context"):
        if key in frontmatter:
            derived[key] = str(frontmatter[key])

    return derived


@dataclass(slots=True)
class NoteFields:
    """Structured values extracted from a note's body and frontmatter."""

    links: list[str]
    tags: list[str]
    headings: list[str]
    tasks: list[str]
    code_mentions: list[str] = field(default_factory=list)


def extract_note_fields(body: str, frontmatter: dict[str, Any]) -> NoteFields:
    """Extract links, tags, headings and tasks, ignoring markdown inside code.

    Exposed separately from ``parse_note`` so redaction can re-derive these
    fields from redacted content instead of rewriting already-extracted values.
    """

    masked = mask_code(body)
    return NoteFields(
        links=findall_outside_code(WIKILINK_RE, masked, body),
        tags=sorted(_inline_tags(masked) | set(_frontmatter_tags(frontmatter))),
        headings=findall_outside_code(HEADING_RE, masked, body),
        tasks=findall_outside_code(TASK_RE, masked, body),
        code_mentions=extract_code_mentions(body, masked),
    )


def extract_section_fields(texts: list[str], frontmatter: dict[str, Any]) -> NoteFields:
    """Extract fields from each independent text, then merge them in order.

    For formats whose sections are separate documents (canvas cards), so an
    unclosed code fence in one section cannot swallow the next one's links
    and tags, as it would if the texts were joined first.
    """

    parts = [extract_note_fields(text, frontmatter) for text in texts]
    if not parts:
        parts = [extract_note_fields("", frontmatter)]
    return NoteFields(
        links=[link for part in parts for link in part.links],
        tags=sorted({tag for part in parts for tag in part.tags}),
        headings=[heading for part in parts for heading in part.headings],
        tasks=[task for part in parts for task in part.tasks],
        code_mentions=list(
            dict.fromkeys(mention for part in parts for mention in part.code_mentions)
        ),
    )


def extract_code_mentions(body: str, masked: str | None = None) -> list[str]:
    """Return what a note says that may name code, in first-seen order.

    Candidates are inline code spans (such as ``GraphBuilder`` or
    ``graph.py`` in backticks), GitHub file and repo URLs, and slash-separated paths with a source or
    markdown extension written in prose (``second_brain/graph.py``). The
    graph keeps only those that match an indexed code-source file, symbol or
    repo, so anything else here is harmless. Fenced blocks are skipped: a
    pasted snippet mentions every name it uses.
    """

    masked = mask_code(body) if masked is None else masked
    mentions: list[str] = []
    for match in INLINE_CODE_RE.finditer(body):
        # Inline code keeps its backticks in the masked copy; a fenced line
        # is blanked whole, backticks included.
        if masked[match.start()] != "`":
            continue
        span = match.group(2).strip()
        if span and len(span) <= _MAX_CODE_MENTION and "\n" not in span:
            mentions.append(span)
    # A URL that ends a sentence must not carry its period into the path.
    mentions.extend(
        body[m.start(1) : m.end(1)].rstrip(".,;:!?") for m in GITHUB_URL_RE.finditer(masked)
    )
    mentions.extend(body[m.start(1) : m.end(1)] for m in CODE_PATH_RE.finditer(masked))
    return list(dict.fromkeys(mentions))[:_MAX_CODE_MENTIONS]


def parse_note(path: Path, vault_root: Path | None = None) -> ParsedNote:
    """Parse a note with identity scoped to its vault-relative path.

    Content hashes remain content-only for change detection. Including the
    path in note identity prevents identical files from sharing ownership.
    """

    text = path.read_text(encoding="utf-8")
    frontmatter, body = parse_frontmatter(text)

    rel_path = str(path if vault_root is None else path.relative_to(vault_root))
    title = path.stem
    fields = extract_note_fields(body, frontmatter)

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    identity = hashlib.sha256(f"{rel_path}\0{digest}".encode("utf-8")).hexdigest()

    return ParsedNote(
        note_id=identity[:16],
        path=rel_path,
        title=title,
        body=body,
        frontmatter=frontmatter,
        tags=fields.tags,
        links=fields.links,
        headings=fields.headings,
        tasks=fields.tasks,
        code_mentions=fields.code_mentions,
        mtime=path.stat().st_mtime,
        content_hash=digest,
    )
