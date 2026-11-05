"""Typed graph edges between notes and code-source files.

The vault graph (``links`` edges) scores pairs of vault notes from four blended
signals. This module adds edges that are facts rather than scores:

- ``mentions_code``: a note (or a repo doc) names a code-source file, symbol or
  repo, in inline code (``GraphBuilder``, ``second_brain/graph.py``), a GitHub
  URL, or a path written in prose. A vault note can mention any repo; a repo
  doc only its own.
- ``imports``: a source file imports another file of the same repo.

Matching favors precision: a name is linked only when exactly one file
defines it, so a mention of ``load`` that five loaders define links nothing,
and a bare lowercase word such as ``sync`` never matches a symbol, because in
prose it is far more often the word than the function. Qualify it
(``Indexer.sync``) to link it.

Two rules keep links stable and believable across syncs:

- A name that was linked stays linked to the same file while that file still
  matches it, even if another file starts matching too (``prefer``). The
  note was written about the first one.
- In a vault note, a one-word bare name (``Task``, ``base.py``) links only if
  the note also names that repo some other way: its name, a URL, a path with
  a folder, or a longer name. One-word names are what other projects, and
  other languages, use too.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from second_brain.models import Edge
from second_brain.sources import split_code_source_path

LINKS = "links"
MENTIONS_CODE = "mentions_code"
IMPORTS = "imports"
EDGE_TYPES = (LINKS, MENTIONS_CODE, IMPORTS)

# A mention or import is a one-way link; a mutual one (two files importing
# each other) scores like a mutual wikilink.
_ONE_WAY = 0.7
_MUTUAL = 1.0
# Evidence listed per edge; a note that mentions a file fifty times needs few.
MAX_REFS_PER_EDGE = 10

_GITHUB_RE = re.compile(
    r"^https?://github\.com/[\w.-]+/(?P<repo>[\w.-]+?)(?:\.git)?"
    r"(?:/(?:blob|tree)/[^/]+/(?P<path>[^#?]+))?/?(?:[#?].*)?$"
)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_CALL_SUFFIX_RE = re.compile(r"\([^()]*\)$")
_LINE_SUFFIX_RE = re.compile(r"(?::\d+){1,2}$|#L\d+(?:-L\d+)?$")
_PATH_EXTENSIONS = (".py", ".pyi", ".go", ".ts", ".tsx", ".mts", ".cts", ".md")
# File names that every project has. Alone, without a folder, they say which
# file but not whose: `README.md` in a note is rarely this repo's README.
_GENERIC_BASENAMES = frozenset(
    {
        "__init__.py", "__main__.py", "main.py", "setup.py", "conftest.py", "utils.py",
        "main.go", "doc.go", "index.ts", "index.tsx", "types.ts", "utils.ts",
    }
)
_SENTENCE_PUNCTUATION = ".,;:!?"
# Words of a name: ``GraphBuilder`` -> Graph, Builder; ``HTTPServer`` -> HTTP,
# Server; ``compute_backoff`` (split at ``_`` first) -> compute, backoff.
_NAME_WORD_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
# Emitted extension -> sources TypeScript checks for it, implementation first,
# then the declaration file a package may ship alone.
_EMITTED_TO_SOURCE = {".mjs": (".mts", ".d.mts"), ".cjs": (".cts", ".d.cts"), ".jsx": (".tsx",)}
_TS_RESOLUTION = (
    ".ts", ".tsx", ".mts", ".cts", ".d.ts", "/index.ts", "/index.tsx", "/index.d.ts",
)


def looks_like_symbol(name: str) -> bool:
    """Whether a bare name is specific enough to match a symbol on its own.

    ``chunk_note`` and ``GraphBuilder`` are; ``sync`` and ``path`` are not.
    """

    return len(name) >= 3 and ("_" in name.strip("_") or any(ch.isupper() for ch in name))


def is_one_word_name(mention: str) -> bool:
    """Whether a mention is a bare one-word name: ``Task``, ``base.py``, ``util.go:12``.

    Paths with a folder, dotted names, URLs and multi-word names
    (``GraphBuilder``, ``compute_backoff``, ``stack-ledger.py``) are not.
    """

    text = _LINE_SUFFIX_RE.sub("", _CALL_SUFFIX_RE.sub("", mention.strip().strip("'\"")))
    if "/" in text or "://" in text:
        return False
    stem, dot, extension = text.rpartition(".")
    if dot:
        if f".{extension.lower()}" not in _PATH_EXTENSIONS or "." in stem:
            return False
        text = stem
    return sum(len(_NAME_WORD_RE.findall(part)) for part in re.split(r"[-_]+", text)) <= 1


@dataclass(frozen=True, slots=True)
class MentionTarget:
    """What one mention in one note resolved to: a file, and a symbol in it if named."""

    path: str
    symbol: str | None


@dataclass(slots=True)
class _File:
    path: str  # stored path, repo:<name>/<rel>
    repo: str
    rel: str  # path in the repo, exact case
    folded: str  # ``rel`` lowercased, for mentions typed in another case


@dataclass(slots=True)
class _CodeIndex:
    """Lookups over every code-source file in a note population."""

    # Keyed by lowercased basename; matching then prefers the exact case.
    files_by_basename: dict[str, list[_File]] = field(default_factory=dict)
    files_by_rel: dict[tuple[str, str], _File] = field(default_factory=dict)
    # A case-sensitive filesystem can hold pkg/Helper.py and pkg/helper.py,
    # so a lowercased path may name several files.
    files_by_folded: dict[tuple[str, str], list[_File]] = field(default_factory=dict)
    # Non-test Go files by (repo, folder): a Go import names a package folder.
    # The repo root is folder "".
    go_packages: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    # Module paths a repo's own imports reveal: ``github.com/acme/widget``
    # when some file imports ``github.com/acme/widget/pkg`` and pkg/ exists.
    go_modules: dict[str, set[str]] = field(default_factory=dict)
    by_qualname: dict[str, set[str]] = field(default_factory=dict)
    by_name: dict[str, set[str]] = field(default_factory=dict)
    symbols_by_file: dict[str, list[str]] = field(default_factory=dict)
    readmes: dict[str, str] = field(default_factory=dict)
    # Built once: every mention checks it, so a per-call scan of all files
    # would make resolution O(files x mentions).
    repos: set[str] = field(default_factory=set)
    repos_by_folded: dict[str, str] = field(default_factory=dict)

    @classmethod
    def build(cls, metas: Iterable[dict]) -> _CodeIndex:
        index = cls()
        for meta in metas:
            split = split_code_source_path(meta["path"])
            if split is None:
                continue
            repo, rel = split
            file = _File(meta["path"], repo, rel, rel.lower())
            index.files_by_rel[(repo, rel)] = file
            index.files_by_folded.setdefault((repo, file.folded), []).append(file)
            index.files_by_basename.setdefault(posixpath.basename(file.folded), []).append(file)
            if file.rel.endswith(".go") and not file.rel.endswith("_test.go"):
                directory = posixpath.dirname(file.rel)
                index.go_packages.setdefault((repo, directory), []).append(meta["path"])
            if "/" not in rel and rel.lower().startswith("readme") and rel.lower().endswith(".md"):
                # README.md wins over README.de.md as the repo's own node.
                current = index.readmes.get(repo)
                if current is None or len(meta["path"]) < len(current):
                    index.readmes[repo] = meta["path"]
            index.symbols_by_file[meta["path"]] = meta.get("code", {}).get("symbols", [])
            for qualname in meta.get("code", {}).get("symbols", []):
                index.by_qualname.setdefault(qualname, set()).add(meta["path"])
                index.by_name.setdefault(qualname.rsplit(".", 1)[-1], set()).add(meta["path"])
        index.repos = {repo for repo, _ in index.files_by_rel}
        index.repos_by_folded = {repo.lower(): repo for repo in index.repos}
        index._infer_go_modules(metas)
        return index

    def _infer_go_modules(self, metas: Iterable[dict]) -> None:
        """Learn each repo's Go module path from imports of its own subpackages.

        go.mod is not indexed, so this is how ``import "github.com/acme/widget"``
        can be recognized as the repo's root package rather than a stranger's.
        """

        for meta in metas:
            split = split_code_source_path(meta["path"])
            if split is None or not split[1].endswith(".go"):
                continue
            for spec in meta.get("code", {}).get("imports", []):
                parts = spec.split("/")
                if len(parts) < 2 or "." not in parts[0]:
                    continue
                for start in range(1, len(parts)):
                    if (split[0], "/".join(parts[start:])) in self.go_packages:
                        self.go_modules.setdefault(split[0], set()).add("/".join(parts[:start]))
                        break

    def _lookup(self, repo: str, rel: str) -> _File | None:
        """The file at ``rel``: exact case first, else the only case-insensitive match."""

        exact = self.files_by_rel.get((repo, rel))
        if exact is not None:
            return exact
        folded = self.files_by_folded.get((repo, rel.lower()), [])
        return folded[0] if len(folded) == 1 else None

    # -- mentions ------------------------------------------------------------------

    def resolve_mention(self, mention: str, scope: str | None, directory: str | None = None) -> list[str]:
        """Return the code-source file ``mention`` names, or [] if none or ambiguous.

        ``scope`` limits a repo doc's mentions to its own repo, and
        ``directory`` is the doc's folder in it, for ``./`` and ``../`` paths.
        """

        return [target.path for target in self.resolve_mention_targets(mention, scope, directory)]

    def resolve_mention_targets(
        self,
        mention: str,
        scope: str | None,
        directory: str | None = None,
        prefer: str | None = None,
    ) -> list[MentionTarget]:
        """Like ``resolve_mention``, also saying which symbol the mention named.

        The symbol comes from the step that resolved the mention, so
        ``pkg.compute_backoff`` resolved as the module ``pkg/compute_backoff.py``
        names the whole file even if that file defines ``compute_backoff``.
        ``prefer`` is the file the mention resolved to last time: when the
        mention now matches several files and that is one of them, it wins.
        """

        paths, qualname, terminal = self._resolve(mention, scope, directory, prefer)
        return [MentionTarget(path, self._symbol_in(path, qualname, terminal)) for path in paths]

    def _symbol_in(self, path: str, qualname: str | None, terminal: str | None) -> str | None:
        if qualname is not None:
            return qualname
        if terminal is None:
            return None
        named = [q for q in self.symbols_by_file.get(path, []) if q.rsplit(".", 1)[-1] == terminal]
        # Two symbols of that name in the file (``A.load``, ``B.load``): the
        # mention names the file, not one of them.
        return named[0] if len(named) == 1 else None

    def _resolve(
        self, mention: str, scope: str | None, directory: str | None, prefer: str | None = None
    ) -> tuple[list[str], str | None, str | None]:
        """Resolve a mention to files, plus the qualname or terminal name that matched.

        Returns ``(paths, qualname, terminal)``: ``qualname`` when a qualified
        symbol matched, ``terminal`` when a bare or module-qualified name
        matched by its last segment, neither when a path, URL, repo or module did.
        """

        # A URL ending a sentence keeps its period in mentions recorded
        # before the parser trimmed it.
        text = mention.strip().strip("'\"")
        if text.startswith(("http://", "https://")):
            text = text.rstrip(_SENTENCE_PUNCTUATION)
        github = _GITHUB_RE.match(text)
        if github:
            repo = github.group("repo")
            if repo not in self.repos or (scope is not None and repo != scope):
                return [], None, None
            if github.group("path"):
                # The URL names the repo, so a root README.md or main.py is
                # not the ambiguous bare name _resolve_path refuses.
                path = github.group("path").strip("/")
                exact = self._lookup(repo, path)
                if exact is not None:
                    return [exact.path], None, None
                return self._resolve_path(path, repo, prefer), None, None
            return ([self.readmes[repo]] if repo in self.readmes else []), None, None
        if text in self.repos:
            if scope is not None and text != scope:
                return [], None, None
            return ([self.readmes[text]] if text in self.readmes else []), None, None

        text = _LINE_SUFFIX_RE.sub("", _CALL_SUFFIX_RE.sub("", text))
        if scope is not None and directory is not None and text.startswith(("./", "../")):
            # A doc's ``./helper.py`` is its sibling, not the repo root's file.
            text = posixpath.normpath(posixpath.join(directory, text))
            if text == ".." or text.startswith("../"):
                return [], None, None
            # The doc pins the file, so ``../README.md`` is not the ambiguous
            # bare name ``_resolve_path`` refuses.
            exact = self._lookup(scope, text)
            if exact is not None:
                return [exact.path], None, None
        if "/" in text or text.lower().endswith(_PATH_EXTENSIONS):
            return self._resolve_path(text, scope, prefer), None, None
        if not _IDENTIFIER_RE.match(text):
            return [], None, None
        if "." in text:
            return self._resolve_dotted(text, scope, prefer)
        if not looks_like_symbol(text):
            return [], None, None
        return self._unique(self.by_name.get(text, set()), scope, prefer), None, text

    def _resolve_dotted(
        self, text: str, scope: str | None, prefer: str | None = None
    ) -> tuple[list[str], str | None, str | None]:
        """``Class.method``, ``module.Class`` or a module path such as ``pkg.mod``."""

        parts = text.split(".")
        # Drop leading segments until a qualified symbol matches, so
        # ``graph.GraphBuilder.rebuild_full`` finds ``GraphBuilder.rebuild_full``.
        for start in range(len(parts) - 1):
            qualname = ".".join(parts[start:])
            matches = self.by_qualname.get(qualname)
            if matches:
                return self._unique(matches, scope, prefer), qualname, None
        module = "/".join(parts)
        for candidate in (f"{module}.py", f"{module}/__init__.py"):
            found = self._resolve_path(candidate, scope, prefer)
            if found:
                return found, None, None
        # ``graph.GraphBuilder``: a module-qualified top-level symbol, held to
        # the same rules as a bare name, so ``self.sync`` or ``os.path`` still
        # match nothing.
        terminal = parts[-1]
        if looks_like_symbol(terminal):
            return self._unique(self.by_name.get(terminal, set()), scope, prefer), None, terminal
        return [], None, None

    def _resolve_path(self, raw: str, scope: str | None, prefer: str | None = None) -> list[str]:
        """Match a written path to one file: exact path, then unique suffix.

        Each step tries the exact case before ignoring case, and the first
        step that matches anything decides: two matches there mean the mention
        is ambiguous, not that a looser step should pick one.
        """

        path = raw.strip().replace("\\", "/")
        while path.startswith(("./", "/")):
            path = path[2:] if path.startswith("./") else path[1:]
        if not path:
            return []
        folded = path.lower()
        if "/" not in folded and (folded.endswith(".md") or folded in _GENERIC_BASENAMES):
            return []
        first, _, rest = path.partition("/")
        # ``second-brain/README.md`` names the repo as its first segment.
        named_repo = self.repos_by_folded.get(first.lower())
        if named_repo is not None and rest and (scope is None or named_repo == scope):
            exact = self._lookup(named_repo, rest)
            if exact is not None:
                return [exact.path]
        candidates = [
            file
            for file in self.files_by_basename.get(posixpath.basename(folded), [])
            if scope is None or file.repo == scope
        ]
        for matches in (
            lambda f: f.rel == path,
            lambda f: f.folded == folded,
            lambda f: f.rel.endswith("/" + path),
            lambda f: f.folded.endswith("/" + folded),
        ):
            chosen = [file.path for file in candidates if matches(file)]
            if chosen:
                return self._unique(set(chosen), None, prefer)
        return []

    @staticmethod
    def _unique(paths: set[str], scope: str | None, prefer: str | None = None) -> list[str]:
        """The one path in ``paths``, or ``prefer`` when it is among several."""

        if scope is not None:
            paths = {p for p in paths if split_code_source_path(p)[0] == scope}
        if len(paths) == 1:
            return sorted(paths)
        return [prefer] if prefer is not None and prefer in paths else []

    # -- imports -------------------------------------------------------------------

    def resolve_import(self, spec: str, importer: str) -> list[str]:
        """Return the files of the importer's repo that ``spec`` imports."""

        repo, rel = split_code_source_path(importer)
        directory = posixpath.dirname(rel)
        suffix = posixpath.splitext(rel)[1]
        if suffix in (".py", ".pyi"):
            targets = self._python_import(spec, repo, directory, stub=suffix == ".pyi")
        elif suffix == ".go":
            targets = self._go_import(spec, repo)
        else:
            targets = self._typescript_import(spec, repo, directory)
        return [target for target in targets if target != importer]

    def _python_import(self, spec: str, repo: str, directory: str, stub: bool) -> list[str]:
        """Resolve a module to its file, preferring stubs only when a stub imports it."""

        dots = len(spec) - len(spec.lstrip("."))
        module = spec[dots:].replace(".", "/")
        suffixes = (".pyi", ".py") if stub else (".py", ".pyi")
        # A package wins over a same-named module, as in Python's import
        # system and in type checkers: pkg/helper/__init__.py before pkg/helper.py.
        forms = [f"{{}}/__init__{suffix}" for suffix in suffixes] + [f"{{}}{suffix}" for suffix in suffixes]
        if dots:
            base = directory
            for _ in range(dots - 1):
                base = posixpath.dirname(base)
            stem = posixpath.join(base, module) if module else base
            for form in forms:
                file = self._lookup(repo, form.format(stem).lstrip("/"))
                if file is not None:
                    return [file.path]
            return []
        candidates = [form.format(module) for form in forms]
        for candidate in candidates:
            file = self._lookup(repo, candidate)
            if file is not None:
                return [file.path]
        # A src/ layout puts ``pkg/mod.py`` under a folder. Only once no
        # candidate exists at the root: a suffix match on the package form
        # would otherwise pick tests/fixtures/pkg/helper/__init__.py over the
        # real pkg/helper.py. A one-segment module (``json``) must match
        # exactly, or a repo's utils/json.py would claim every ``import json``.
        if "/" in module:
            for candidate in candidates:
                found = self._resolve_path(candidate, repo)
                if found:
                    return found
        return []

    def _typescript_import(self, spec: str, repo: str, directory: str) -> list[str]:
        if not spec.startswith("."):
            return []
        stem = posixpath.normpath(posixpath.join(directory, spec))
        if stem.startswith(".."):
            return []
        # ESM sources import the emitted name: ``./x.js`` for ``x.ts``,
        # ``./x.mjs`` for ``x.mts``, ``./x.cjs`` for ``x.cts``.
        candidates: tuple[str, ...] = (stem,)
        base, extension = posixpath.splitext(stem)
        if extension == ".js":
            candidates = (base, *(base + ext for ext in _TS_RESOLUTION))
        elif extension in _EMITTED_TO_SOURCE:
            candidates = tuple(base + source for source in _EMITTED_TO_SOURCE[extension])
        else:
            candidates = (stem, *(stem + ext for ext in _TS_RESOLUTION))
        for candidate in candidates:
            # The repo root normalizes to ".", so "./index.ts" must become "index.ts".
            file = self._lookup(repo, posixpath.normpath(candidate))
            if file is not None:
                return [file.path]
        return []

    def _go_import(self, spec: str, repo: str) -> list[str]:
        """Match a module import path's trailing folders to a package folder.

        Standard-library paths (``errors``, ``net/http``) have no dot in their
        first segment and are skipped, so a repo's own ``errors/`` folder is
        not claimed by every ``import "errors"``. go.mod is not indexed, so a
        third-party path that ends in the same folders as a repo package
        (``github.com/x/y/internal/log`` vs ``internal/log``) can still match.
        """

        # Go import paths are case-sensitive, like the folders they name.
        parts = spec.split("/")
        if len(parts) < 2 or "." not in parts[0]:
            return []
        # Longest folder suffix first: ``.../pkg/foo`` prefers pkg/foo over foo.
        for start in range(1, len(parts)):
            files = self.go_packages.get((repo, "/".join(parts[start:])))
            if files:
                return sorted(files)
        # The module's root package: a module path learned from the repo's
        # own imports, or a path ending in the repo's name (``/v2`` aside).
        name_parts = parts[:-1] if re.fullmatch(r"v\d+", parts[-1]) else parts
        if spec in self.go_modules.get(repo, set()) or name_parts[-1].lower() == repo.lower():
            return sorted(self.go_packages.get((repo, ""), []))
        return []


def code_edges(metas: list[dict], link_weight: float) -> list[Edge]:
    """Compute every ``mentions_code`` and ``imports`` edge in a note population.

    Recomputed whole on every sync: it is a dictionary lookup per mention, and
    a changed symbol in one file can re-target mentions in notes that did not
    change.
    """

    return typed_graph(metas, link_weight)[0]


def typed_graph(
    metas: list[dict],
    link_weight: float,
    previous: dict[tuple[str, str], str] | None = None,
) -> tuple[list[Edge], dict[tuple[str, str], MentionTarget | None]]:
    """Typed edges plus what each ``(note, mention)`` resolved to, for drift.

    ``previous`` maps a ``(note, mention)`` to the file it resolved to at the
    last sync, so a name that becomes ambiguous keeps its file. Unresolved
    mentions are absent from the map: most never named code at all. A
    one-word name in a vault note that matched code in a repo the note names
    no other way maps to None: it matched, but is not taken to mean that code.
    """

    previous = previous or {}
    index = _CodeIndex.build(metas)
    resolved: dict[tuple[str, str], MentionTarget | None] = {}
    if not index.files_by_rel:
        return [], resolved
    directed: dict[tuple[str, str], tuple[str, set[str]]] = {}

    def add(source: str, target: str, edge_type: str, ref: str) -> None:
        _, refs = directed.setdefault((source, target), (edge_type, set()))
        refs.add(ref)

    for meta in metas:
        code = meta.get("code", {})
        split = split_code_source_path(meta["path"])
        scope, directory = (split[0], posixpath.dirname(split[1])) if split else (None, None)
        found: dict[str, MentionTarget] = {}
        for mention in code.get("mentions", []):
            prefer = previous.get((meta["path"], mention))
            for target in index.resolve_mention_targets(mention, scope, directory, prefer):
                if target.path != meta["path"]:
                    found[mention] = target
        # A repo doc's scope already says which repo it means; a vault note
        # says so by naming the repo in some way more specific than one word.
        context = {
            split_code_source_path(target.path)[0]
            for mention, target in found.items()
            if scope is not None
            or not is_one_word_name(mention)
            or mention.strip().strip("'\"") in index.repos
        }
        for mention, target in found.items():
            key = (meta["path"], mention)
            if split_code_source_path(target.path)[0] not in context:
                resolved[key] = None
                continue
            add(meta["path"], target.path, MENTIONS_CODE, mention)
            resolved[key] = target
        if split is not None:
            for spec in code.get("imports", []):
                for target in index.resolve_import(spec, meta["path"]):
                    add(meta["path"], target, IMPORTS, spec)

    edges: list[Edge] = []
    seen: set[tuple[str, str]] = set()
    for source, target in sorted(directed):
        src, dst = (source, target) if source < target else (target, source)
        if (src, dst) in seen:
            continue
        seen.add((src, dst))
        forward = directed.get((src, dst))
        backward = directed.get((dst, src))
        edge_type = (forward or backward)[0]
        refs = set().union(*(entry[1] for entry in (forward, backward) if entry))
        link = _MUTUAL if forward and backward else _ONE_WAY
        edges.append(
            Edge(
                src=src,
                dst=dst,
                semantic=0.0,
                link=link,
                tag=0.0,
                comention=0.0,
                comention_count=0,
                link_src_to_dst=forward is not None,
                link_dst_to_src=backward is not None,
                shared_tags=[],
                composite=link_weight * link,
                type=edge_type,
                code_refs=sorted(refs)[:MAX_REFS_PER_EDGE],
            )
        )
    return edges, resolved
