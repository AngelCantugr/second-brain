"""Query routing: decide whether a query is about a code source.

A query that names something distinctive in a code source (the repo name, a
directory or file name, or a symbol) is a question about that repo, so its
repo hits keep their rank. Any other query is treated as a question about the
vault, and ``code_source_weight`` scales its repo hits down.

"Distinctive" is measured against the vault: a name that appears in more than
``MAX_VAULT_NOTES`` vault notes (``plugins``, ``scripts``, ``ledger``) is an
ordinary word there and would route vault questions to the repo. Vault notes
are counted with the same matching a query gets, so a name typed with spaces
("career desk") counts as an appearance of ``career-desk``.
"""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from collections.abc import Collection, Iterable

from second_brain.sources import split_code_source_path

# A name in more vault notes than this is not evidence of repo intent.
MAX_VAULT_NOTES = 2
# Shorter names ("src", "run", "app") collide with ordinary words.
MIN_TERM_LENGTH = 4
# Longest hyphen or underscore name matched when it is written with spaces.
MAX_TERM_WORDS = 4

# Words keep inner hyphens and underscores, so "career-desk" and
# "parse_evidence" are single words, while "ledger.py" splits at the dot.
# Leading and trailing underscores stay too, for "_private" and "__init__".
_WORD_RE = re.compile(r"_*[a-z0-9][a-z0-9_-]*")
_SEPARATOR_RE = re.compile(r"[-_]")


def _words(text: str) -> list[str]:
    return [word.rstrip("-") for word in _WORD_RE.findall(text.lower())]


def _forms(word: str) -> tuple[str, ...]:
    # Markdown italics wrap a word in underscores: "_career-desk_".
    bare = word.strip("_")
    return (word, bare) if bare != word else (word,)


def _first_words(names: Iterable[str]) -> frozenset[str]:
    """First word of each multi-word name, to skip windows that cannot match."""

    return frozenset(
        _SEPARATOR_RE.split(name, maxsplit=1)[0]
        for name in names
        if _SEPARATOR_RE.search(name)
    )


def _named(
    words: list[str], names: Collection[str], first_words: Collection[str]
) -> list[str]:
    """Return the ``names`` that ``words`` contains, in order of appearance.

    A hyphen or underscore name also matches its words written with spaces.
    Windows run over whole words, so "career-desk-v2" never yields
    "career-desk".
    """

    found: list[str] = []
    for word in words:
        for form in _forms(word):
            if form in names and form not in found:
                found.append(form)
    for start, first in enumerate(words):
        if first not in first_words:
            continue
        for size in range(2, MAX_TERM_WORDS + 1):
            window = words[start : start + size]
            if len(window) < size:
                break
            for joined in ("-".join(window), "_".join(window)):
                if joined in names and joined not in found:
                    found.append(joined)
    return found


def _name_terms(metadata: dict) -> set[str]:
    names = {str(metadata["repo"])}
    split = split_code_source_path(str(metadata.get("path", "")))
    if split is not None:
        for part in split[1].split("/"):
            names.add(part.rsplit(".", 1)[0] if "." in part else part)
    if metadata.get("symbol_name"):
        names.add(str(metadata["symbol_name"]))
    return {name.lower() for name in names}


def _is_term(name: str) -> bool:
    return (
        len(name) >= MIN_TERM_LENGTH
        and _WORD_RE.fullmatch(name) is not None
        and not name.endswith("-")
        and re.search(r"[a-z]", name) is not None
    )


def build_repo_terms(chunks: Iterable[tuple[dict, str]]) -> frozenset[str]:
    """Return the code-source names that signal repo intent.

    ``chunks`` yields each indexed chunk's metadata and text. Code-source
    chunks (those with a ``repo``) contribute names; vault chunks contribute
    the note counts that rule common names out.
    """

    candidates: set[str] = set()
    # A set per note: every chunk of a note repeats its title and headings.
    vault_texts: dict[str, set[str]] = {}
    for metadata, text in chunks:
        if metadata.get("repo"):
            candidates.update(name for name in _name_terms(metadata) if _is_term(name))
        else:
            texts = vault_texts.setdefault(str(metadata.get("path", "")), set())
            texts.add(text)
            # The chunker moves headings out of chunk text into heading_path.
            for field in ("note_title", "heading_path"):
                if metadata.get(field):
                    texts.add(str(metadata[field]))

    first_words = _first_words(candidates)
    notes_naming: Counter[str] = Counter()
    for texts in vault_texts.values():
        named: set[str] = set()
        for text in texts:
            named.update(_named(_words(text), candidates, first_words))
        notes_naming.update(named)
    return frozenset(
        name for name in candidates if notes_naming[name] <= MAX_VAULT_NOTES
    )


def repo_intent_terms(query: str, terms: frozenset[str]) -> list[str]:
    """Return the repo terms ``query`` names, in query order.

    A hyphen or underscore name also matches when written with spaces
    ("career desk" for ``career-desk``), since people type names that way.
    """

    return _named(_words(query), terms, _first_words_of_terms(terms))


# Search reuses one term set until the next sync; frozensets cache their hash.
_first_words_of_terms = lru_cache(maxsize=4)(_first_words)
