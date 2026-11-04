"""Query routing: decide whether a query is about a code source.

A query that names something distinctive in a code source (the repo name, a
directory or file name, or a symbol) is a question about that repo, so its
repo hits keep their rank. Any other query is treated as a question about the
vault, and ``code_source_weight`` scales its repo hits down.

"Distinctive" is measured against the vault: a name that appears in more than
``MAX_VAULT_NOTES`` vault notes (``plugins``, ``scripts``, ``ledger``) is an
ordinary word there and would route vault questions to the repo.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

from second_brain.sources import split_code_source_path

# A name in more vault notes than this is not evidence of repo intent.
MAX_VAULT_NOTES = 2
# Shorter names ("src", "run", "app") collide with ordinary words.
MIN_TERM_LENGTH = 4
# Longest hyphen or underscore name matched when the query spells it with spaces.
MAX_TERM_WORDS = 4

# Words keep inner hyphens and underscores, so "career-desk" and
# "parse_evidence" are single words, while "ledger.py" splits at the dot.
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")


def _words(text: str) -> list[str]:
    return [word.rstrip("_-") for word in _WORD_RE.findall(text.lower())]


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
        and not name.endswith(("-", "_"))
        and re.search(r"[a-z]", name) is not None
    )


def build_repo_terms(chunks: Iterable[tuple[dict, str]]) -> frozenset[str]:
    """Return the code-source names that signal repo intent.

    ``chunks`` yields each indexed chunk's metadata and text. Code-source
    chunks (those with a ``repo``) contribute names; vault chunks contribute
    the word counts that rule common names out.
    """

    candidates: set[str] = set()
    vault_words: dict[str, set[str]] = {}
    for metadata, text in chunks:
        if metadata.get("repo"):
            candidates.update(_name_terms(metadata))
        else:
            path = str(metadata.get("path", ""))
            vault_words.setdefault(path, set()).update(_words(text))

    notes_per_word: Counter[str] = Counter()
    for words in vault_words.values():
        notes_per_word.update(words)
    return frozenset(
        name
        for name in candidates
        if _is_term(name) and notes_per_word[name] <= MAX_VAULT_NOTES
    )


def repo_intent_terms(query: str, terms: frozenset[str]) -> list[str]:
    """Return the repo terms ``query`` names, in query order.

    A hyphen or underscore name also matches when written with spaces
    ("career desk" for ``career-desk``), since people type names that way.
    """

    words = _words(query)
    found: list[str] = []
    for word in words:
        if word in terms and word not in found:
            found.append(word)

    # Windows run over whole words, so "career-desk-v2" never yields "career-desk".
    for size in range(2, MAX_TERM_WORDS + 1):
        for start in range(len(words) - size + 1):
            window = words[start : start + size]
            for joined in ("-".join(window), "_".join(window)):
                if joined in terms and joined not in found:
                    found.append(joined)
    return found
