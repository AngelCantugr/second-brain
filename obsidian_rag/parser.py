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
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from obsidian_rag.models import ParsedNote

WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
TAG_RE = re.compile(r"(?<!\w)#([A-Za-z0-9_/-]+)")
HEADING_RE = re.compile(r"^#{1,6}\s+(.*)$", re.MULTILINE)
TASK_RE = re.compile(r"^\s*[-*]\s+\[(?: |x|X)\]\s+(.*)$", re.MULTILINE)


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

    return loaded, body


def derive_metadata(frontmatter: dict[str, Any]) -> dict[str, Any]:
    """Compute normalized, query-friendly metadata fields.

    Raw frontmatter is kept as-is elsewhere; these are best-effort
    derived fields used for consistent filtering.
    """

    derived: dict[str, Any] = {}
    date_keys = ("due", "deadline", "start", "created")
    for key in date_keys:
        value = frontmatter.get(key)
        if isinstance(value, (str, datetime)):
            derived[f"{key}_date"] = str(value)

    status = frontmatter.get("status")
    if status is not None:
        derived["status"] = str(status).lower()

    for key in ("project", "context"):
        if key in frontmatter:
            derived[key] = str(frontmatter[key])

    return derived


def parse_note(path: Path, vault_root: Path | None = None) -> ParsedNote:
    """Parse a markdown file into a ``ParsedNote`` object."""

    text = path.read_text(encoding="utf-8")
    frontmatter, body = parse_frontmatter(text)

    rel_path = str(path if vault_root is None else path.relative_to(vault_root))
    title = path.stem
    links = WIKILINK_RE.findall(body)
    tags = sorted(set(TAG_RE.findall(body)))
    headings = HEADING_RE.findall(body)
    tasks = TASK_RE.findall(body)

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()

    return ParsedNote(
        note_id=digest[:16],
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
