"""Canvas loader: parsing, chunking, redaction, and indexing into search and the graph."""

import json
from pathlib import Path

import pytest

from second_brain.canvas import parse_canvas
from second_brain.chunker import chunk_note
from second_brain.config import RagConfig
from second_brain.redaction import Redactor
from second_brain.service import RagService


def _node(node_id: str, node_type: str, x=0, y=0, width=100, height=100, **extra) -> dict:
    return {"id": node_id, "type": node_type, "x": x, "y": y, "width": width, "height": height, **extra}


def _write_canvas(path: Path, nodes: list, edges: list | None = None) -> Path:
    path.write_text(json.dumps({"nodes": nodes, "edges": edges or []}), encoding="utf-8")
    return path


def test_text_nodes_become_sections_titled_by_containing_groups(tmp_path: Path) -> None:
    canvas = _write_canvas(
        tmp_path / "Board.canvas",
        [
            _node("outer", "group", x=0, y=0, width=1000, height=1000, label="Plan"),
            _node("inner", "group", x=10, y=10, width=500, height=500, label="Risks"),
            _node("t-lower", "text", x=20, y=300, text="Second idea #risk"),
            _node("t-upper", "text", x=20, y=20, text="First idea links [[Note]]"),
            _node("t-outer", "text", x=600, y=600, text="Outside the inner group"),
            _node("t-free", "text", x=2000, y=100, text="Loose note"),
            _node("t-blank", "text", x=0, y=0, text="   "),
        ],
    )

    parsed = parse_canvas(canvas, tmp_path)

    assert parsed.kind == "canvas"
    assert parsed.title == "Board.canvas"
    assert parsed.path == "Board.canvas"
    # Reading order: top to bottom, then left to right; blank text dropped.
    assert parsed.sections == [
        ("Plan > Risks", "First idea links [[Note]]"),
        ("root", "Loose note"),
        ("Plan > Risks", "Second idea #risk"),
        ("Plan", "Outside the inner group"),
    ]
    assert parsed.tags == ["risk"]
    assert parsed.links == ["Note"]
    assert parsed.frontmatter == {}


def test_file_nodes_become_links(tmp_path: Path) -> None:
    canvas = _write_canvas(
        tmp_path / "Board.canvas",
        [
            _node("f1", "file", file="Projects/Alpha.md"),
            _node("f2", "file", file="Projects/Alpha.md", subpath="#Goals"),
            _node("f3", "file", file="img/diagram.png"),
            _node("u1", "link", url="https://example.com"),
        ],
    )

    parsed = parse_canvas(canvas, tmp_path)

    assert parsed.file_links == ["/Projects/Alpha.md", "/img/diagram.png"]
    assert parsed.links == ["/Projects/Alpha.md", "/img/diagram.png"]
    assert parsed.sections == []


@pytest.mark.parametrize(
    ("ends", "expected"),
    [
        ({}, [("/a.md", "/b.md")]),  # spec defaults: toEnd=arrow, fromEnd=none
        ({"fromEnd": "arrow", "toEnd": "none"}, [("/b.md", "/a.md")]),
        ({"fromEnd": "arrow", "toEnd": "arrow"}, [("/a.md", "/b.md"), ("/b.md", "/a.md")]),
        ({"fromEnd": "none", "toEnd": "none"}, [("/a.md", "/b.md")]),
    ],
)
def test_edges_between_file_nodes_follow_arrowheads(
    tmp_path: Path, ends: dict, expected: list
) -> None:
    canvas = _write_canvas(
        tmp_path / "Board.canvas",
        [_node("na", "file", file="a.md"), _node("nb", "file", file="b.md")],
        [{"id": "e1", "fromNode": "na", "toNode": "nb", **ends}],
    )

    assert parse_canvas(canvas, tmp_path).link_pairs == expected


def test_edges_touching_non_file_nodes_or_one_file_are_ignored(tmp_path: Path) -> None:
    canvas = _write_canvas(
        tmp_path / "Board.canvas",
        [
            _node("na", "file", file="a.md"),
            _node("na2", "file", file="a.md"),
            _node("t", "text", text="hello"),
        ],
        [
            {"id": "e1", "fromNode": "na", "toNode": "t"},
            {"id": "e2", "fromNode": "na", "toNode": "na2"},  # same file twice
            {"id": "e3", "fromNode": "na", "toNode": "missing"},
            "not an edge",
        ],
    )

    assert parse_canvas(canvas, tmp_path).link_pairs == []


def test_malformed_json_raises_value_error(tmp_path: Path) -> None:
    canvas = tmp_path / "Broken.canvas"
    canvas.write_text('{"nodes": [', encoding="utf-8")

    with pytest.raises(ValueError, match="invalid canvas JSON"):
        parse_canvas(canvas, tmp_path)


def test_non_object_top_level_raises_value_error(tmp_path: Path) -> None:
    canvas = tmp_path / "List.canvas"
    canvas.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="top level"):
        parse_canvas(canvas, tmp_path)


def test_invalid_nodes_are_skipped_and_empty_canvas_parses(tmp_path: Path) -> None:
    canvas = tmp_path / "Odd.canvas"
    canvas.write_text(
        json.dumps(
            {
                "nodes": [
                    "junk",
                    {"type": "text", "text": "no id"},
                    {"id": "t", "type": "text", "text": "no geometry"},
                    {"id": "g", "type": "group", "label": "Bad box", "x": "0"},
                ],
                "edges": {"not": "a list"},
            }
        ),
        encoding="utf-8",
    )
    empty = tmp_path / "Empty.canvas"
    empty.write_text("{}", encoding="utf-8")

    assert parse_canvas(canvas, tmp_path).sections == [("root", "no geometry")]
    assert parse_canvas(empty, tmp_path).sections == []


def test_chunks_carry_canvas_kind_and_group_heading(tmp_path: Path) -> None:
    canvas = _write_canvas(
        tmp_path / "Board.canvas",
        [
            _node("g", "group", x=0, y=0, width=500, height=500, label="Ideas"),
            _node("t", "text", x=10, y=10, text="# Heading inside\nbody text"),
        ],
    )

    chunks = chunk_note(parse_canvas(canvas, tmp_path), chunk_size=50, chunk_overlap=0)

    assert len(chunks) == 1
    assert chunks[0].metadata["kind"] == "canvas"
    # A markdown heading inside a text node stays text: the section comes
    # from the canvas structure, not the body.
    assert chunks[0].heading_path == "Ideas"
    assert chunks[0].text == "# Heading inside\nbody text"


def test_redaction_covers_sections_and_keeps_file_links(tmp_path: Path) -> None:
    canvas = _write_canvas(
        tmp_path / "Board.canvas",
        [
            _node("t", "text", text="call 555-0199 now"),
            _node("f", "file", file="people/555-0199.md"),
            _node("f2", "file", file="b.md"),
        ],
        [{"id": "e", "fromNode": "f", "toNode": "f2"}],
    )

    redacted = Redactor([r"555-\d{4}"]).note(parse_canvas(canvas, tmp_path))

    assert redacted.sections == [("root", "call [REDACTED] now")]
    assert redacted.file_links == ["/b.md", "/people/[REDACTED].md"]
    assert redacted.links == ["/b.md", "/people/[REDACTED].md"]
    assert redacted.link_pairs == [("/people/[REDACTED].md", "/b.md")]


# -- end to end through RagService -------------------------------------------


class _VocabEmbedder:
    _VOCAB = ["alpha", "beta", "gamma", "roadmap"]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0 + text.lower().count(w) for w in self._VOCAB] for text in texts]

    def health(self) -> bool:
        return True


def _service(tmp_path: Path, include_extensions: list[str]) -> RagService:
    vault = tmp_path / "vault"
    vault.mkdir()
    config = RagConfig(
        vault_path=vault,
        qdrant_path=tmp_path / "qdrant",
        fts_path=tmp_path / "fts.sqlite",
        sync_state_path=tmp_path / "sync_state.sqlite",
        include_extensions=include_extensions,
    )
    service = RagService(config, use_in_memory_vector=True)
    service.embedder = _VocabEmbedder()
    service.indexer.embedder = _VocabEmbedder()
    return service


def _vault_with_canvas(vault: Path) -> None:
    (vault / "Alpha.md").write_text("# Alpha\nalpha notes", encoding="utf-8")
    (vault / "Beta.md").write_text("# Beta\nbeta notes", encoding="utf-8")
    _write_canvas(
        vault / "Board.canvas",
        [
            _node("t", "text", x=0, y=0, text="roadmap for the quarter"),
            _node("a", "file", x=200, y=0, file="Alpha.md"),
            _node("b", "file", x=400, y=0, file="Beta.md"),
        ],
        [{"id": "e", "fromNode": "a", "toNode": "b"}],
    )


def test_canvas_is_ignored_unless_its_extension_is_included(tmp_path: Path) -> None:
    service = _service(tmp_path, [".md"])
    _vault_with_canvas(service.config.vault_path)

    service.sync(mode="full")

    assert sorted(service.sync_state.tracked_paths()) == ["Alpha.md", "Beta.md"]
    assert service.graph_store.note_meta_for("Board.canvas") is None


def test_canvas_text_is_searchable_and_filterable_by_kind(tmp_path: Path) -> None:
    service = _service(tmp_path, [".md", ".canvas"])
    _vault_with_canvas(service.config.vault_path)

    result = service.sync(mode="full")
    assert result["errors"] == []

    hits = service.search("roadmap quarter", filters={"kind": "canvas"})["hits"]
    assert [hit["metadata"]["path"] for hit in hits] == ["Board.canvas"]
    assert hits[0]["metadata"]["kind"] == "canvas"

    markdown = service.search("notes", filters={"kind": ["markdown"]})["hits"]
    assert {hit["metadata"]["path"] for hit in markdown} == {"Alpha.md", "Beta.md"}


def test_canvas_file_nodes_and_edges_become_graph_edges(tmp_path: Path) -> None:
    service = _service(tmp_path, [".md", ".canvas"])
    _vault_with_canvas(service.config.vault_path)

    service.sync(mode="full")

    to_alpha = service.graph_store.edge_between("Board.canvas", "Alpha.md")
    assert to_alpha is not None and to_alpha.link > 0
    # The canvas edge Alpha -> Beta is link evidence between those two notes.
    pair = service.graph_store.edge_between("Alpha.md", "Beta.md")
    assert pair is not None
    assert pair.link_src_to_dst is True
    assert pair.link_dst_to_src is False


def test_file_cards_resolve_to_their_exact_path_not_a_nearer_namesake(tmp_path: Path) -> None:
    service = _service(tmp_path, [".md", ".canvas"])
    vault = service.config.vault_path
    (vault / "Projects").mkdir()
    for path in ("Alpha.md", "Projects/Alpha.md", "Beta.md"):
        (vault / path).write_text(f"# {path}\nalpha beta", encoding="utf-8")
    _write_canvas(
        vault / "Projects" / "Board.canvas",
        [
            _node("a", "file", x=0, y=0, file="Alpha.md"),
            _node("b", "file", x=200, y=0, file="Beta.md"),
        ],
        [{"id": "e", "fromNode": "a", "toNode": "b"}],
    )

    service.sync(mode="full")

    root = service.graph_store.edge_between("Alpha.md", "Beta.md")
    assert root is not None and root.link_src_to_dst is True
    nested = service.graph_store.edge_between("Projects/Alpha.md", "Beta.md")
    assert nested is None or not (nested.link_src_to_dst or nested.link_dst_to_src)
    board = service.graph_store.edge_between("Alpha.md", "Projects/Board.canvas")
    assert board is not None and board.link > 0
    namesake = service.graph_store.edge_between("Projects/Alpha.md", "Projects/Board.canvas")
    assert namesake is None or namesake.link == 0


def test_removing_a_canvas_edge_incrementally_drops_its_link(tmp_path: Path) -> None:
    service = _service(tmp_path, [".md", ".canvas"])
    vault = service.config.vault_path
    _vault_with_canvas(vault)
    service.sync(mode="full")

    _write_canvas(
        vault / "Board.canvas",
        [
            _node("t", "text", x=0, y=0, text="roadmap for the quarter"),
            _node("a", "file", x=200, y=0, file="Alpha.md"),
            _node("b", "file", x=400, y=0, file="Beta.md"),
        ],
    )
    service.sync(mode="incremental")

    pair = service.graph_store.edge_between("Alpha.md", "Beta.md")
    # Still co-mentioned by the canvas, but no longer linked.
    assert pair is None or (pair.link_src_to_dst, pair.link_dst_to_src) == (False, False)


def test_malformed_canvas_is_reported_and_markdown_still_indexes(tmp_path: Path) -> None:
    service = _service(tmp_path, [".md", ".canvas"])
    vault = service.config.vault_path
    (vault / "Alpha.md").write_text("# Alpha\nalpha notes", encoding="utf-8")
    (vault / "Broken.canvas").write_text("{not json", encoding="utf-8")

    result = service.sync(mode="full")

    assert len(result["errors"]) == 1
    assert "Broken.canvas" in result["errors"][0]
    assert set(service.sync_state.tracked_paths()) == {"Alpha.md"}


def test_file_sync_of_excluded_extension_removes_it(tmp_path: Path) -> None:
    service = _service(tmp_path, [".md", ".canvas"])
    _vault_with_canvas(service.config.vault_path)
    service.sync(mode="full")
    assert "Board.canvas" in service.sync_state.tracked_paths()

    service.config.include_extensions = [".md"]
    result = service.sync(mode="file", file_path="Board.canvas")

    assert result["deleted"] == 1
    assert "Board.canvas" not in service.sync_state.tracked_paths()


def test_graph_store_adds_link_pairs_column_to_older_databases(tmp_path: Path) -> None:
    import sqlite3

    from second_brain.graph import GraphStore

    db = tmp_path / "fts.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE note_meta (path TEXT PRIMARY KEY, title TEXT NOT NULL, "
            "title_key TEXT NOT NULL, tags_json TEXT NOT NULL, links_json TEXT NOT NULL, "
            "aliases_json TEXT NOT NULL DEFAULT '[]', centroid BLOB, dim INTEGER, "
            "updated_at REAL)"
        )
        conn.execute(
            "INSERT INTO note_meta(path, title, title_key, tags_json, links_json) "
            "VALUES('old.md', 'old', 'old', '[]', '[]')"
        )
    store = GraphStore(db)

    store.initialize()
    store.upsert_note_meta("b.canvas", "b.canvas", [], ["x.md"], None, [], [("x.md", "y.md")])

    assert store.note_meta_for("old.md")["link_pairs"] == []
    assert store.note_meta_for("b.canvas")["link_pairs"] == [("x.md", "y.md")]


def test_an_unclosed_fence_in_one_card_does_not_hide_the_next_cards_links(
    tmp_path: Path,
) -> None:
    canvas = _write_canvas(
        tmp_path / "Board.canvas",
        [
            _node("code", "text", x=0, y=0, text="```python\nprint(1)"),
            _node("prose", "text", x=0, y=200, text="see [[Alpha]] #important"),
        ],
    )

    parsed = parse_canvas(canvas, tmp_path)
    # Redaction re-extracts fields; it must keep the cards apart too.
    redacted = Redactor([r"nomatch-\d+"]).note(parsed)

    for note in (parsed, redacted):
        assert note.links == ["Alpha"]
        assert note.tags == ["important"]
