"""Split source files into symbol outlines with tree-sitter.

Each function, class, method or type becomes one ``CodeSymbol``. Its ``text``
is the symbol's *why*: the leading comment, the signature and the docstring,
copied from the source as written. Bodies are left out because grep, LSP tools
and the file itself answer *what* the code does better than a RAG copy that
goes stale. A symbol with no comment or docstring gets the first lines of its
body instead, so its embedding still has something to say.

tree-sitter is an optional dependency (``pip install 'second-brain-mcp[code]'``);
nothing here imports it until a code file is parsed.
"""

from __future__ import annotations

import bisect
import hashlib
import importlib.util
import inspect
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from second_brain.models import ParsedNote

CODE_KIND = "code"
# Body lines used in place of documentation for an undocumented symbol.
UNDOCUMENTED_BODY_LINES = 12
# Type and interface declarations are documentation themselves, up to a point.
MAX_DECLARATION_LINES = 40

# language name -> (grammar module, attribute returning the grammar, extensions)
_GRAMMARS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "python": ("tree_sitter_python", "language", (".py", ".pyi")),
    "go": ("tree_sitter_go", "language", (".go",)),
    "typescript": ("tree_sitter_typescript", "language_typescript", (".ts", ".mts", ".cts")),
    "tsx": ("tree_sitter_typescript", "language_tsx", (".tsx",)),
}
# Config names a language; TypeScript covers both of its grammars.
SUPPORTED_LANGUAGES: dict[str, tuple[str, ...]] = {
    "python": ("python",),
    "go": ("go",),
    "typescript": ("typescript", "tsx"),
}
_GRAMMAR_BY_EXTENSION = {
    extension: grammar for grammar, (_, _, extensions) in _GRAMMARS.items() for extension in extensions
}


@dataclass(slots=True)
class CodeSymbol:
    """One function, class, method or type declaration, outlined."""

    name: str
    # Dotted path inside the file, such as ``GraphBuilder.rebuild_full``.
    qualname: str
    parent: str | None
    # function | method | class | type | interface | module
    symbol_kind: str
    start_line: int  # 1-based, inclusive
    end_line: int
    signature: str
    # Docstring and/or leading comment, with comment markers removed.
    doc: str
    # What is chunked, embedded and keyword-indexed.
    text: str
    # ``doc`` as written in the source, so redaction can rebuild it from the
    # redacted originals (see ``symbol_doc``). The leading comments are kept
    # as one span, from the first comment through the last.
    comment_source: str | None = None
    docstring_source: str | None = None


def missing_code_dependencies(languages: list[str]) -> list[str]:
    """Return the importable modules ``languages`` need that are not installed."""

    modules = {"tree_sitter"} | {
        _GRAMMARS[grammar][0] for language in languages for grammar in SUPPORTED_LANGUAGES[language]
    }
    return sorted(m for m in modules if importlib.util.find_spec(m) is None)


def code_extensions(languages: list[str]) -> frozenset[str]:
    """File extensions handled by the configured ``languages``."""

    return frozenset(
        extension
        for language in languages
        for grammar in SUPPORTED_LANGUAGES[language]
        for extension in _GRAMMARS[grammar][2]
    )


def language_for(path: str | Path) -> str | None:
    """Return the grammar name for a file's extension, or None."""

    return _GRAMMAR_BY_EXTENSION.get(Path(path).suffix)


def load_code_file(path: Path, rel_path: str, repo: str, stored_path: str) -> ParsedNote:
    """Parse one source file into a ``ParsedNote`` whose ``symbols`` are its outlines.

    The body, links and tags stay empty: markdown extraction over code would
    read ``#include`` as a tag and ``[[x]]`` as a wikilink.
    """

    text = path.read_text(encoding="utf-8")
    grammar = language_for(path)
    if grammar is None:
        raise ValueError(f"no code grammar for {path.suffix or 'files without an extension'}")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    identity = hashlib.sha256(f"{stored_path}\0{digest}".encode("utf-8")).hexdigest()
    language = "typescript" if grammar == "tsx" else grammar
    return ParsedNote(
        note_id=identity[:16],
        path=stored_path,
        title=f"{repo}: {rel_path}",
        body="",
        frontmatter={},
        tags=[],
        links=[],
        headings=[],
        tasks=[],
        mtime=path.stat().st_mtime,
        content_hash=digest,
        kind=CODE_KIND,
        metadata={"repo": repo, "language": language, "file": rel_path},
        symbols=extract_symbols(text, grammar),
        source=text,
    )


def extract_symbols(
    source: str,
    grammar: str,
    redactions: Sequence[tuple[int, int]] = (),
    token: str = "",
) -> list[CodeSymbol]:
    """Outline ``source``: the module docstring, then each symbol in file order.

    ``redactions`` are sorted, disjoint ``(start, end)`` character spans
    matched against the whole file. The tree is parsed from the original
    source, so the outlines and line numbers are unchanged, but every piece
    of text read from a span is replaced by ``token``. Parsing the redacted
    text instead would break on a token such as ``[REDACTED]``, which is not
    valid syntax in most places it lands.
    """

    data = source.encode("utf-8")
    tree = _parser(grammar).parse(data)
    extractor = _EXTRACTORS[grammar]
    symbols: list[CodeSymbol] = []
    extractor(_Source(data, _byte_spans(source, redactions), token), tree.root_node, symbols)
    return symbols


def _byte_spans(source: str, spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Convert sorted character spans to byte spans of the UTF-8 encoding."""

    byte_spans = []
    char, byte = 0, 0
    for start, end in spans:
        byte += len(source[char:start].encode("utf-8"))
        start_byte = byte
        byte += len(source[start:end].encode("utf-8"))
        char = end
        byte_spans.append((start_byte, byte))
    return byte_spans


@cache
def _parser(grammar: str):
    import importlib

    from tree_sitter import Language, Parser

    module_name, attribute, _ = _GRAMMARS[grammar]
    module = importlib.import_module(module_name)
    return Parser(Language(getattr(module, attribute)()))


class _Source:
    """The file's bytes, with helpers that read node text and source rows.

    Every read goes through ``text``, which replaces each part of a redacted
    span with ``token``, so no extracted field can carry a matched string.
    """

    def __init__(self, data: bytes, redactions: Sequence[tuple[int, int]] = (), token: str = "") -> None:
        self.data = data
        self.redactions = redactions
        self.token = token
        # Byte offset where each row starts. tree-sitter counts rows at "\n"
        # only; str.splitlines() also breaks on form feeds and U+2028, and
        # character offsets drift from byte offsets after any multi-byte
        # character, so rows are located in bytes.
        self.row_starts = [0]
        offset = data.find(b"\n")
        while offset != -1:
            self.row_starts.append(offset + 1)
            offset = data.find(b"\n", offset + 1)

    def text(self, start_byte: int, end_byte: int) -> str:
        parts = []
        cursor = start_byte
        for span_start, span_end in self.redactions:
            if span_end <= cursor:
                continue
            if span_start >= end_byte:
                break
            parts.append(self.data[cursor:span_start].decode("utf-8", "replace"))
            parts.append(self.token)
            cursor = min(span_end, end_byte)
        parts.append(self.data[cursor:end_byte].decode("utf-8", "replace"))
        return "".join(parts)

    def node_text(self, node) -> str:
        return self.text(node.start_byte, node.end_byte)

    def row_of(self, byte: int) -> int:
        return bisect.bisect_right(self.row_starts, byte) - 1

    def row_end(self, row: int) -> int:
        """Byte offset of the newline ending ``row``, or the end of the file."""

        if row + 1 < len(self.row_starts):
            return self.row_starts[row + 1] - 1
        return len(self.data)

    def outline(self, start_byte: int, end_byte: int) -> str:
        """Source between two offsets, starting at the row's start when only
        indentation comes before ``start_byte``.

        Slicing by offset rather than whole rows keeps out a sibling that shares
        a row, such as ``function a() {} function b() {}``.
        """

        row_start = self.row_starts[self.row_of(start_byte)]
        if not self.data[row_start:start_byte].strip():
            start_byte = row_start
        return self.text(start_byte, end_byte).strip("\n")


def _leading_comments(node) -> list:
    """Comment nodes directly above ``node``, with no blank line between them."""

    comments = []
    row = node.start_point[0]
    sibling = node.prev_named_sibling
    while sibling is not None and sibling.type == "comment" and sibling.end_point[0] == row - 1:
        # A trailing comment on the previous statement's line is not a header.
        before = sibling.prev_named_sibling
        if before is not None and before.end_point[0] == sibling.start_point[0]:
            break
        comments.append(sibling)
        row = sibling.start_point[0]
        sibling = sibling.prev_named_sibling
    return list(reversed(comments))


def _clean_comment(text: str) -> str:
    """Strip ``#``, ``//`` and ``/* */`` markers, keeping the words."""

    lines = []
    for line in text.strip().splitlines():
        line = line.strip()
        for marker in ("/**", "/*", "*/", "//", "#"):
            if line.startswith(marker):
                line = line[len(marker) :]
        if line.endswith("*/"):
            line = line[:-2]
        line = line.lstrip("*").strip() if line.startswith("*") else line.strip()
        lines.append(line)
    return "\n".join(lines).strip()


def _clean_docstring(literal: str) -> str:
    """Turn a Python string literal into its docstring text."""

    body = literal.lstrip("rRbBuUfF")
    for quote in ('"""', "'''", '"', "'"):
        if body.startswith(quote) and body.endswith(quote) and len(body) >= 2 * len(quote):
            body = body[len(quote) : -len(quote)]
            break
    return inspect.cleandoc(body)


def symbol_doc(comment_source: str | None, docstring_source: str | None) -> str:
    """Build a symbol's ``doc`` from its leading comments and docstring as written.

    Redaction calls this on the redacted sources: a pattern that matches source
    syntax, such as ``// SECRET:.*``, no longer matches once the markers are
    cleaned off, so redacting ``doc`` alone would let the secret through.
    """

    parts = [] if comment_source is None else [_clean_comment(comment_source)]
    if docstring_source is not None:
        parts.append(_clean_docstring(docstring_source))
    return "\n\n".join(part for part in parts if part)


def _symbol(
    src: _Source,
    node,
    *,
    name: str,
    parent: str | None,
    symbol_kind: str,
    signature_end: int,
    docstring_node=None,
    header_node=None,
) -> CodeSymbol:
    """Build a symbol whose text runs from its first leading comment to its outline's end.

    ``header_node`` is the node comments sit above (an ``export`` statement or
    a decorated definition), when it is not ``node`` itself.
    """

    header = header_node or node
    comments = _leading_comments(header)
    signature = src.text(header.start_byte, signature_end).rstrip()
    # One span, not one string per comment: consecutive ``//`` lines are a
    # node each, and a redaction pattern may span several of them.
    comment_source = src.text(comments[0].start_byte, comments[-1].end_byte) if comments else None
    docstring_source = None if docstring_node is None else src.node_text(docstring_node)
    doc = symbol_doc(comment_source, docstring_source)
    start_byte = comments[0].start_byte if comments else header.start_byte
    last_row = src.row_of(signature_end)
    if docstring_node is not None:
        last_row = docstring_node.end_point[0]
    if not doc:
        last_row = min(node.end_point[0], last_row + UNDOCUMENTED_BODY_LINES)
    return CodeSymbol(
        name=name,
        qualname=f"{parent}.{name}" if parent else name,
        parent=parent,
        symbol_kind=symbol_kind,
        start_line=header.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        signature=signature,
        doc=doc,
        text=src.outline(start_byte, min(node.end_byte, src.row_end(last_row))),
        comment_source=comment_source,
        docstring_source=docstring_source,
    )


# -- Python ------------------------------------------------------------------------


def _python_docstring(block):
    if block is None:
        return None
    # A shebang, encoding line or license header comes before the docstring
    # without stopping it from being one.
    first = next((child for child in block.named_children if child.type != "comment"), None)
    if first is not None and first.type == "expression_statement" and first.named_child_count == 1:
        literal = first.named_children[0]
        if literal.type == "string":
            return literal
    return None


def _extract_python(src: _Source, root, symbols: list[CodeSymbol]) -> None:
    module_doc = _python_docstring(root)
    if module_doc is not None:
        symbols.append(
            CodeSymbol(
                name="(module)",
                qualname="(module)",
                parent=None,
                symbol_kind="module",
                start_line=1,
                end_line=module_doc.end_point[0] + 1,
                signature="",
                doc=symbol_doc(None, src.node_text(module_doc)),
                text=src.node_text(module_doc),
                docstring_source=src.node_text(module_doc),
            )
        )
    _python_block(src, root, None, symbols)


# Statements whose blocks hold module- or class-level definitions: platform
# variants under ``if``, ``except ImportError`` fallbacks, and the like.
_PYTHON_CONTROL_FLOW = frozenset(
    {
        "if_statement",
        "elif_clause",
        "else_clause",
        "try_statement",
        "except_clause",
        "except_group_clause",
        "finally_clause",
        "with_statement",
        "block",
    }
)


def _python_block(src: _Source, block, parent: str | None, symbols: list[CodeSymbol]) -> None:
    for child in block.named_children:
        if child.type in _PYTHON_CONTROL_FLOW:
            _python_block(src, child, parent, symbols)
            continue
        header, node = child, child
        if child.type == "decorated_definition":
            node = child.child_by_field_name("definition")
        if node is None or node.type not in ("function_definition", "class_definition"):
            continue
        body = node.child_by_field_name("body")
        name = src.node_text(node.child_by_field_name("name"))
        is_class = node.type == "class_definition"
        if is_class:
            symbol_kind = "class"
        else:
            symbol_kind = "method" if parent is not None else "function"
        symbol = _symbol(
            src,
            node,
            name=name,
            parent=parent,
            symbol_kind=symbol_kind,
            signature_end=body.start_byte if body is not None else node.end_byte,
            docstring_node=_python_docstring(body),
            header_node=header,
        )
        symbols.append(symbol)
        # Methods belong to classes; functions nested in functions are details.
        if is_class and body is not None:
            _python_block(src, body, symbol.qualname, symbols)


# -- Go ----------------------------------------------------------------------------


def _go_receiver_type(src: _Source, method) -> str | None:
    receiver = method.child_by_field_name("receiver")
    if receiver is None:
        return None
    for node in _walk(receiver):
        if node.type == "type_identifier":
            return src.node_text(node)
    return None


def _extract_go(src: _Source, root, symbols: list[CodeSymbol]) -> None:
    for child in root.named_children:
        if child.type in ("function_declaration", "method_declaration"):
            body = child.child_by_field_name("body")
            is_method = child.type == "method_declaration"
            symbols.append(
                _symbol(
                    src,
                    child,
                    name=src.node_text(child.child_by_field_name("name")),
                    parent=_go_receiver_type(src, child) if is_method else None,
                    symbol_kind="method" if is_method else "function",
                    signature_end=body.start_byte if body is not None else child.end_byte,
                )
            )
        elif child.type == "type_declaration":
            # In a ``type ( ... )`` group each spec has its own lines and
            # comments; a lone spec's comment sits above the ``type`` keyword.
            grouped = any(token.type == "(" for token in child.children)
            for spec in child.named_children:
                if spec.type not in ("type_spec", "type_alias"):
                    continue
                header = spec if grouped else child
                type_node = spec.child_by_field_name("type")
                is_interface = type_node is not None and type_node.type == "interface_type"
                symbols.append(
                    _symbol(
                        src,
                        spec,
                        name=src.node_text(spec.child_by_field_name("name")),
                        parent=None,
                        symbol_kind="interface" if is_interface else "type",
                        signature_end=_capped_end(src, header),
                        header_node=header,
                    )
                )


# -- TypeScript ------------------------------------------------------------------

# ``function_signature`` is a bodyless ``declare function`` or overload.
_TS_FUNCTIONS = ("function_declaration", "generator_function_declaration", "function_signature")
_TS_METHODS = ("method_definition", "method_signature", "abstract_method_signature")
_TS_CLASSES = ("class_declaration", "abstract_class_declaration")
_TS_TYPES = {
    "interface_declaration": "interface",
    "type_alias_declaration": "type",
    "enum_declaration": "type",
}


def _extract_typescript(src: _Source, root, symbols: list[CodeSymbol]) -> None:
    for child in root.named_children:
        header, node = child, child
        if child.type == "export_statement":
            node = child.child_by_field_name("declaration")
            if node is None:
                continue
        if node.type == "ambient_declaration":
            # ``declare class``/``declare function``: one declaration inside.
            # ``declare module`` and ``declare global`` blocks are skipped.
            node = node.named_children[0] if node.named_child_count else None
            if node is None:
                continue
        _typescript_declaration(src, node, header, symbols)


def _typescript_declaration(src: _Source, node, header, symbols: list[CodeSymbol]) -> None:
    name_node = node.child_by_field_name("name")
    if node.type in _TS_FUNCTIONS or node.type in _TS_CLASSES:
        body = node.child_by_field_name("body")
        is_class = node.type in _TS_CLASSES
        symbol = _symbol(
            src,
            node,
            name=src.node_text(name_node) if name_node else "default",
            parent=None,
            symbol_kind="class" if is_class else "function",
            signature_end=body.start_byte if body is not None else node.end_byte,
            header_node=header,
        )
        symbols.append(symbol)
        if is_class and body is not None:
            for member in body.named_children:
                if member.type in _TS_METHODS:
                    member_body = member.child_by_field_name("body")
                    symbols.append(
                        _symbol(
                            src,
                            member,
                            name=src.node_text(member.child_by_field_name("name")),
                            parent=symbol.qualname,
                            symbol_kind="method",
                            signature_end=(
                                member_body.start_byte if member_body is not None else member.end_byte
                            ),
                        )
                    )
    elif node.type in _TS_TYPES and name_node is not None:
        symbols.append(
            _symbol(
                src,
                node,
                name=src.node_text(name_node),
                parent=None,
                symbol_kind=_TS_TYPES[node.type],
                signature_end=_capped_end(src, node),
                header_node=header,
            )
        )
    elif node.type == "lexical_declaration":
        # ``const handler = (req) => ...`` is how much TypeScript declares functions.
        for index, declarator in enumerate(node.named_children):
            value = declarator.child_by_field_name("value")
            if declarator.type != "variable_declarator" or value is None:
                continue
            if value.type not in ("arrow_function", "function_expression", "function"):
                continue
            body = value.child_by_field_name("body")
            symbols.append(
                _symbol(
                    src,
                    declarator,
                    name=src.node_text(declarator.child_by_field_name("name")),
                    parent=None,
                    symbol_kind="function",
                    signature_end=body.start_byte if body is not None else value.end_byte,
                    # The statement's comment and ``export const`` belong to its
                    # first declarator; ``, second = ...`` starts at its name.
                    header_node=header if index == 0 else declarator,
                )
            )


def _capped_end(src: _Source, node) -> int:
    """End byte of ``node``, or of its first ``MAX_DECLARATION_LINES`` lines."""

    last_row = node.start_point[0] + MAX_DECLARATION_LINES - 1
    if node.end_point[0] <= last_row:
        return node.end_byte
    return src.row_end(last_row)


def _walk(node):
    yield node
    for child in node.named_children:
        yield from _walk(child)


_EXTRACTORS = {
    "python": _extract_python,
    "go": _extract_go,
    "typescript": _extract_typescript,
    "tsx": _extract_typescript,
}
