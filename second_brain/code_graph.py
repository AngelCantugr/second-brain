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
_EMITTED_TO_SOURCE = {".mjs": ".mts", ".cjs": ".cts", ".jsx": ".tsx"}
_TS_RESOLUTION = (".ts", ".tsx", ".mts", ".cts", ".d.ts", "/index.ts", "/index.tsx")


def looks_like_symbol(name: str) -> bool:
    """Whether a bare name is specific enough to match a symbol on its own.

    ``chunk_note`` and ``GraphBuilder`` are; ``sync`` and ``path`` are not.
    """

    return len(name) >= 3 and ("_" in name.strip("_") or any(ch.isupper() for ch in name))


@dataclass(slots=True)
class _File:
    path: str  # stored path, repo:<name>/<rel>
    repo: str
    rel: str  # lowercased path in the repo


@dataclass(slots=True)
class _CodeIndex:
    """Lookups over every code-source file in a note population."""

    files_by_basename: dict[str, list[_File]] = field(default_factory=dict)
    files_by_rel: dict[tuple[str, str], _File] = field(default_factory=dict)
    # Non-test Go files by (repo, folder): a Go import names a package folder.
    # The repo root is folder "".
    go_packages: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    # Module paths a repo's own imports reveal: ``github.com/acme/widget``
    # when some file imports ``github.com/acme/widget/pkg`` and pkg/ exists.
    go_modules: dict[str, set[str]] = field(default_factory=dict)
    by_qualname: dict[str, set[str]] = field(default_factory=dict)
    by_name: dict[str, set[str]] = field(default_factory=dict)
    readmes: dict[str, str] = field(default_factory=dict)

    @classmethod
    def build(cls, metas: Iterable[dict]) -> _CodeIndex:
        index = cls()
        for meta in metas:
            split = split_code_source_path(meta["path"])
            if split is None:
                continue
            repo, rel = split
            file = _File(meta["path"], repo, rel.lower())
            index.files_by_rel[(repo, file.rel)] = file
            index.files_by_basename.setdefault(posixpath.basename(file.rel), []).append(file)
            if file.rel.endswith(".go") and not file.rel.endswith("_test.go"):
                directory = posixpath.dirname(file.rel)
                index.go_packages.setdefault((repo, directory), []).append(meta["path"])
            if "/" not in rel and rel.lower().startswith("readme") and rel.lower().endswith(".md"):
                # README.md wins over README.de.md as the repo's own node.
                current = index.readmes.get(repo)
                if current is None or len(meta["path"]) < len(current):
                    index.readmes[repo] = meta["path"]
            for qualname in meta.get("code", {}).get("symbols", []):
                index.by_qualname.setdefault(qualname, set()).add(meta["path"])
                index.by_name.setdefault(qualname.rsplit(".", 1)[-1], set()).add(meta["path"])
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
                parts = spec.lower().split("/")
                if len(parts) < 2 or "." not in parts[0]:
                    continue
                for start in range(1, len(parts)):
                    if (split[0], "/".join(parts[start:])) in self.go_packages:
                        self.go_modules.setdefault(split[0], set()).add("/".join(parts[:start]))
                        break

    @property
    def repos(self) -> set[str]:
        return {repo for repo, _ in self.files_by_rel}

    # -- mentions ------------------------------------------------------------------

    def resolve_mention(self, mention: str, scope: str | None) -> list[str]:
        """Return the code-source file ``mention`` names, or [] if none or ambiguous.

        ``scope`` limits a repo doc's mentions to its own repo.
        """

        text = mention.strip().strip("'\"")
        github = _GITHUB_RE.match(text)
        if github:
            repo = github.group("repo")
            if repo not in self.repos or (scope is not None and repo != scope):
                return []
            if github.group("path"):
                # The URL names the repo, so a root README.md or main.py is
                # not the ambiguous bare name _resolve_path refuses.
                path = github.group("path").strip("/").lower()
                exact = self.files_by_rel.get((repo, path))
                return [exact.path] if exact is not None else self._resolve_path(path, repo)
            return [self.readmes[repo]] if repo in self.readmes else []
        if text in self.repos:
            if scope is not None and text != scope:
                return []
            return [self.readmes[text]] if text in self.readmes else []

        text = _LINE_SUFFIX_RE.sub("", _CALL_SUFFIX_RE.sub("", text))
        if "/" in text or text.lower().endswith(_PATH_EXTENSIONS):
            return self._resolve_path(text, scope)
        if not _IDENTIFIER_RE.match(text):
            return []
        if "." in text:
            return self._resolve_dotted(text, scope)
        if not looks_like_symbol(text):
            return []
        return self._unique(self.by_name.get(text, set()), scope)

    def _resolve_dotted(self, text: str, scope: str | None) -> list[str]:
        """``Class.method``, ``module.Class`` or a module path such as ``pkg.mod``."""

        parts = text.split(".")
        # Drop leading segments until a qualified symbol matches, so
        # ``graph.GraphBuilder.rebuild_full`` finds ``GraphBuilder.rebuild_full``.
        for start in range(len(parts) - 1):
            matches = self.by_qualname.get(".".join(parts[start:]))
            if matches:
                return self._unique(matches, scope)
        module = "/".join(parts)
        for candidate in (f"{module}.py", f"{module}/__init__.py"):
            found = self._resolve_path(candidate, scope)
            if found:
                return found
        # ``graph.GraphBuilder``: a module-qualified top-level symbol, held to
        # the same rules as a bare name, so ``self.sync`` or ``os.path`` still
        # match nothing.
        terminal = parts[-1]
        if looks_like_symbol(terminal):
            return self._unique(self.by_name.get(terminal, set()), scope)
        return []

    def _resolve_path(self, raw: str, scope: str | None) -> list[str]:
        """Match a written path to one file by exact path, then by unique suffix."""

        path = raw.strip().replace("\\", "/").lower()
        while path.startswith(("./", "/")):
            path = path[2:] if path.startswith("./") else path[1:]
        if not path:
            return []
        if "/" not in path and (path.endswith(".md") or path in _GENERIC_BASENAMES):
            return []
        first, _, rest = path.partition("/")
        # ``second-brain/README.md`` names the repo as its first segment.
        named_repo = next((repo for repo in self.repos if repo.lower() == first), None)
        if named_repo is not None and rest and (scope is None or named_repo == scope):
            exact = self.files_by_rel.get((named_repo, rest))
            if exact is not None:
                return [exact.path]
        candidates = [
            file
            for file in self.files_by_basename.get(posixpath.basename(path), [])
            if (scope is None or file.repo == scope)
            and (file.rel == path or file.rel.endswith("/" + path))
        ]
        exact = [file for file in candidates if file.rel == path]
        chosen = exact if exact else candidates
        return [chosen[0].path] if len(chosen) == 1 else []

    @staticmethod
    def _unique(paths: set[str], scope: str | None) -> list[str]:
        if scope is not None:
            paths = {p for p in paths if split_code_source_path(p)[0] == scope}
        return sorted(paths) if len(paths) == 1 else []

    # -- imports -------------------------------------------------------------------

    def resolve_import(self, spec: str, importer: str) -> list[str]:
        """Return the files of the importer's repo that ``spec`` imports."""

        repo, rel = split_code_source_path(importer)
        directory = posixpath.dirname(rel)
        suffix = posixpath.splitext(rel)[1]
        if suffix in (".py", ".pyi"):
            targets = self._python_import(spec, repo, directory)
        elif suffix == ".go":
            targets = self._go_import(spec, repo)
        else:
            targets = self._typescript_import(spec, repo, directory)
        return [target for target in targets if target != importer]

    def _python_import(self, spec: str, repo: str, directory: str) -> list[str]:
        dots = len(spec) - len(spec.lstrip("."))
        module = spec[dots:].replace(".", "/")
        if dots:
            base = directory
            for _ in range(dots - 1):
                base = posixpath.dirname(base)
            stem = posixpath.join(base, module) if module else base
            for candidate in (f"{stem}.py", f"{stem}/__init__.py"):
                file = self.files_by_rel.get((repo, candidate.lower().lstrip("/")))
                if file is not None:
                    return [file.path]
            return []
        for candidate in (f"{module}.py", f"{module}/__init__.py"):
            file = self.files_by_rel.get((repo, candidate.lower()))
            if file is not None:
                return [file.path]
            # A src/ layout puts ``pkg/mod.py`` under a folder. A one-segment
            # module (``json``) must match exactly, or a repo's utils/json.py
            # would claim every ``import json``.
            if "/" in module:
                found = self._resolve_path(candidate, repo)
                if found:
                    return found
        return []

    def _typescript_import(self, spec: str, repo: str, directory: str) -> list[str]:
        if not spec.startswith("."):
            return []
        stem = posixpath.normpath(posixpath.join(directory, spec)).lower()
        if stem.startswith(".."):
            return []
        # ESM sources import the emitted name: ``./x.js`` for ``x.ts``,
        # ``./x.mjs`` for ``x.mts``, ``./x.cjs`` for ``x.cts``.
        candidates: tuple[str, ...] = (stem,)
        base, extension = posixpath.splitext(stem)
        if extension == ".js":
            candidates = (base, *(base + ext for ext in _TS_RESOLUTION))
        elif extension in _EMITTED_TO_SOURCE:
            candidates = (base + _EMITTED_TO_SOURCE[extension],)
        else:
            candidates = (stem, *(stem + ext for ext in _TS_RESOLUTION))
        for candidate in candidates:
            file = self.files_by_rel.get((repo, candidate))
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

        parts = spec.lower().split("/")
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
        if spec.lower() in self.go_modules.get(repo, set()) or name_parts[-1] == repo.lower():
            return sorted(self.go_packages.get((repo, ""), []))
        return []


def code_edges(metas: list[dict], link_weight: float) -> list[Edge]:
    """Compute every ``mentions_code`` and ``imports`` edge in a note population.

    Recomputed whole on every sync: it is a dictionary lookup per mention, and
    a changed symbol in one file can re-target mentions in notes that did not
    change.
    """

    index = _CodeIndex.build(metas)
    if not index.files_by_rel:
        return []
    directed: dict[tuple[str, str], tuple[str, set[str]]] = {}

    def add(source: str, target: str, edge_type: str, ref: str) -> None:
        _, refs = directed.setdefault((source, target), (edge_type, set()))
        refs.add(ref)

    for meta in metas:
        code = meta.get("code", {})
        split = split_code_source_path(meta["path"])
        scope = split[0] if split else None
        for mention in code.get("mentions", []):
            for target in index.resolve_mention(mention, scope):
                if target != meta["path"]:
                    add(meta["path"], target, MENTIONS_CODE, mention)
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
    return edges
