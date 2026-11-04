"""Obsidian canvas parsing (JSON Canvas 1.0, https://jsoncanvas.org/spec/1.0/).

A canvas is JSON with ``nodes`` and ``edges``. It maps onto the parsed-note
shape as follows:

- text nodes become sections, titled by the labels of the groups that
  contain them, so each one is chunked and searchable;
- file nodes become links from the canvas to those files, stored
  vault-absolute (``/Projects/Alpha.md``) so they resolve to that exact path;
- an edge between two file nodes becomes a link between those two files,
  following the arrowheads (``toEnd`` defaults to ``arrow``, ``fromEnd`` to
  ``none``).

Link nodes (URLs) and edge labels are not indexed yet.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from second_brain.chunker import trim_section
from second_brain.models import ParsedNote
from second_brain.parser import extract_section_fields

ROOT_HEADING = "root"


@dataclass(frozen=True, slots=True)
class _Box:
    x: float
    y: float
    width: float
    height: float

    def contains(self, other: _Box) -> bool:
        return (
            self.x <= other.x
            and self.y <= other.y
            and other.x + other.width <= self.x + self.width
            and other.y + other.height <= self.y + self.height
        )

    @property
    def area(self) -> float:
        return self.width * self.height


def parse_canvas(path: Path, vault_root: Path | None = None) -> ParsedNote:
    """Parse a ``.canvas`` file into the parsed-note shape the indexer consumes.

    Raises ``ValueError`` for invalid JSON or a top level that is not an
    object, so the sync reports the file and retries it next time. Nodes and
    edges missing required fields are skipped instead: one bad node should not
    hide the rest of the canvas.
    """

    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid canvas JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("invalid canvas: top level must be a JSON object")

    nodes = [node for node in _list(data, "nodes") if _valid_node(node)]
    groups = [
        (box, label)
        for node in nodes
        if node["type"] == "group"
        and isinstance(label := node.get("label"), str)
        and label.strip()
        and (box := _box(node)) is not None
    ]

    sections: list[tuple[str, str]] = []
    # Reading order (top to bottom, left to right) rather than z-order, so the
    # stored body reads the way the canvas looks.
    text_nodes = [node for node in nodes if node["type"] == "text"]
    for node in sorted(text_nodes, key=_reading_order):
        body = node.get("text")
        if isinstance(body, str) and body.strip():
            # Trim like markdown sections: a full strip() would drop the first
            # line's indent and break an indented code block.
            sections.append((_heading_for(node, groups), trim_section(body.splitlines())))

    file_by_node: dict[str, str] = {}
    for node in nodes:
        file = node.get("file") if node["type"] == "file" else None
        if isinstance(file, str) and file.strip():
            # A file card names one exact vault path. The leading "/" makes
            # the graph resolve it exactly, not by name proximity like a
            # wikilink.
            file_by_node[node["id"]] = "/" + file.strip().lstrip("/")

    link_pairs: list[tuple[str, str]] = []
    for edge in _list(data, "edges"):
        if not isinstance(edge, dict):
            continue
        source = file_by_node.get(edge.get("fromNode"))
        target = file_by_node.get(edge.get("toNode"))
        if source is None or target is None or source == target:
            continue
        to_arrow = edge.get("toEnd", "arrow") != "none"
        from_arrow = edge.get("fromEnd", "none") == "arrow"
        # A plain line with no arrowhead still connects the two files; read
        # it in drawing order rather than dropping it.
        if to_arrow or not from_arrow:
            link_pairs.append((source, target))
        if from_arrow:
            link_pairs.append((target, source))

    body = "\n\n".join(section_text for _, section_text in sections)
    fields = extract_section_fields([section_text for _, section_text in sections], {})
    file_links = sorted(set(file_by_node.values()))

    rel_path = str(path if vault_root is None else path.relative_to(vault_root))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    identity = hashlib.sha256(f"{rel_path}\0{digest}".encode("utf-8")).hexdigest()

    return ParsedNote(
        note_id=identity[:16],
        path=rel_path,
        # The full file name, so the graph's title key is ``board.canvas``:
        # Obsidian resolves ``[[Board]]`` to ``Board.md``, and a canvas only
        # through ``[[Board.canvas]]``.
        title=path.name,
        body=body,
        frontmatter={},
        tags=fields.tags,
        links=sorted(set(fields.links) | set(file_links)),
        headings=fields.headings,
        tasks=fields.tasks,
        code_mentions=fields.code_mentions,
        mtime=path.stat().st_mtime,
        content_hash=digest,
        kind="canvas",
        sections=sections,
        file_links=file_links,
        link_pairs=sorted(set(link_pairs)),
    )


def _list(data: dict[str, Any], key: str) -> list[Any]:
    value = data.get(key)
    return value if isinstance(value, list) else []


def _valid_node(node: Any) -> bool:
    return (
        isinstance(node, dict)
        and isinstance(node.get("id"), str)
        and isinstance(node.get("type"), str)
    )


def _box(node: dict[str, Any]) -> _Box | None:
    values = [node.get(key) for key in ("x", "y", "width", "height")]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
        return None
    return _Box(*values)


def _reading_order(node: dict[str, Any]) -> tuple[float, float]:
    box = _box(node)
    return (box.y, box.x) if box is not None else (float("inf"), float("inf"))


def _heading_for(node: dict[str, Any], groups: list[tuple[_Box, str]]) -> str:
    """Join the labels of every group containing ``node``, outermost first."""

    box = _box(node)
    if box is None:
        return ROOT_HEADING
    containing = [(group, label) for group, label in groups if group.contains(box)]
    if not containing:
        return ROOT_HEADING
    containing.sort(key=lambda item: -item[0].area)
    return " > ".join(label.strip() for _, label in containing)
