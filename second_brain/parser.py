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
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from second_brain.models import ParsedNote

WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
TAG_RE = re.compile(r"(?<!\w)#([A-Za-z0-9_/-]+)")
URL_RE = re.compile(r"https?://\S+")
HEADING_RE = re.compile(r"^#{1,6}\s+(.*)$", re.MULTILINE)
TASK_RE = re.compile(r"^\s*[-*]\s+\[(?: |x|X)\]\s+(.*)$", re.MULTILINE)

# A fence line may sit at any indent (nested list items put fences 4+ spaces
# in) and behind blockquote markers (``> ``, ``> > ``, Obsidian callouts).
# CommonMark's "at most 3 spaces" rule is relaxed on purpose: missing a
# list-nested fence leaks code syntax into tags, while treating a rare
# indented-code line that starts with ``` as a fence only blanks extra text.
FENCE_RE = re.compile(r"^[ \t]*(?:>[ \t]?)*[ \t]*(`{3,}|~{3,})(.*)$")
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
    open_fence: tuple[str, int] | None = None

    for line in text.splitlines(keepends=True):
        fence = FENCE_RE.match(line.rstrip())
        if open_fence is None:
            if fence and _may_open_fence(fence):
                open_fence = (fence.group(1)[0], len(fence.group(1)))
                masked_lines.append(_blank(line))
            else:
                masked_lines.append(_mask_inline_code(line))
            continue

        if fence and _closes_fence(fence, *open_fence):
            open_fence = None
        masked_lines.append(_blank(line))

    return "".join(masked_lines)


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


def _closes_fence(fence: re.Match[str], fence_char: str, fence_len: int) -> bool:
    run = fence.group(1)
    return run[0] == fence_char and len(run) >= fence_len and not fence.group(2).strip()


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


def _inline_tags(body: str) -> set[str]:
    """Extract genuine Obsidian ``#tags`` from note body text.

    URLs are masked out first so ``#`` fragments (TickTick task links,
    GitHub anchors, etc.) never get mistaken for tags. Purely numeric
    matches (``#1``, ``#3``) are also dropped, matching Obsidian's own
    tag rule that a tag must contain at least one non-digit character —
    otherwise inline prose references like "see item #3" pollute tags.
    """

    body_without_urls = URL_RE.sub(" ", body)
    return {
        match
        for match in TAG_RE.findall(body_without_urls)
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


def parse_note(path: Path, vault_root: Path | None = None) -> ParsedNote:
    """Parse a note with identity scoped to its vault-relative path.

    Content hashes remain content-only for change detection. Including the
    path in note identity prevents identical files from sharing ownership.
    """

    text = path.read_text(encoding="utf-8")
    frontmatter, body = parse_frontmatter(text)

    rel_path = str(path if vault_root is None else path.relative_to(vault_root))
    title = path.stem
    masked = mask_code(body)
    links = findall_outside_code(WIKILINK_RE, masked, body)
    tags = sorted(_inline_tags(masked) | set(_frontmatter_tags(frontmatter)))
    headings = findall_outside_code(HEADING_RE, masked, body)
    tasks = findall_outside_code(TASK_RE, masked, body)

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    identity = hashlib.sha256(f"{rel_path}\0{digest}".encode("utf-8")).hexdigest()

    return ParsedNote(
        note_id=identity[:16],
        path=rel_path,
        title=title,
        body=body,
        frontmatter=frontmatter,
        tags=tags,
        links=links,
        headings=headings,
        tasks=tasks,
        mtime=path.stat().st_mtime,
        content_hash=digest,
    )
