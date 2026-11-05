"""What part a document plays next to the code it mentions.

Docs in a repo (and notes in a vault) relate to code in different ways, so the
same signal means different things for each:

- ``reference``: living docs (README, a skill's SKILL.md, an architecture
  page) that should describe the code as it is now. Code changing under one
  means the doc may be out of date.
- ``record``: intent at a point in time (a design, a plan, a spec, an ADR).
  It was written against the code of its day and is not updated as the code
  moves on, so code changing under it is expected. What is worth knowing is
  how far the code has moved from it: a record whose code is mostly changed or
  gone was superseded, or was built differently from what it says.
- ``history``: a log of past changes (a changelog). It names code that has
  since changed by definition, so drift ignores it.

A role is decided from the path by globs, checked in the order ``reference``,
``history``, ``record``; anything no glob matches is ``reference``. Roles are
computed when asked, not stored at index time, so changing the globs needs no
re-sync.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from second_brain.scanner import path_is_excluded

if TYPE_CHECKING:
    from second_brain.config import RagConfig

REFERENCE = "reference"
RECORD = "record"
HISTORY = "history"
DOC_ROLES = (REFERENCE, RECORD, HISTORY)
# Explicit reference globs win, so a README inside a specs folder stays living.
_PRECEDENCE = (REFERENCE, HISTORY, RECORD)

# A date in a file or folder name: 2026-11-04.
_DATE = "*[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]*"

# Defaults for vault notes and every code source's docs. Slash-free globs match
# a file name at any depth; ``specs/**`` matches inside a ``specs`` folder at
# any depth. A dated journal note that mentions code is a record too.
DEFAULT_DOC_ROLES: dict[str, tuple[str, ...]] = {
    REFERENCE: ("README*", "CONTRIBUTING*", "ARCHITECTURE*", "SKILL.md"),
    HISTORY: ("CHANGELOG*", "HISTORY*"),
    RECORD: (
        "design.md", "*-design.md", "plan.md", "plan-*.md", "*-plan.md",
        "specs/**", "spec/**", "plans/**", "adr/**", "adrs/**", "decisions/**", "rfcs/**",
        _DATE, f"{_DATE}/**",
    ),
}


def validated_doc_roles(
    value: object, setting: str, defaults: Mapping[str, tuple[str, ...]]
) -> dict[str, tuple[str, ...]]:
    """Parse a ``doc_roles`` table: each role given replaces that role's defaults."""

    if not isinstance(value, dict):
        raise ValueError(f"{setting} must be a table of role = [globs]")
    unknown = sorted(set(value) - set(DOC_ROLES))
    if unknown:
        raise ValueError(f"{setting}: unknown roles {unknown}; roles: {list(DOC_ROLES)}")
    roles = {role: tuple(defaults.get(role, ())) for role in DOC_ROLES}
    for role, globs in value.items():
        if not isinstance(globs, list) or any(not isinstance(item, str) for item in globs):
            raise ValueError(f"{setting}.{role} must be an array of strings")
        roles[role] = tuple(globs)
    return roles


def role_for(relative_path: str, roles: Mapping[str, tuple[str, ...]]) -> str:
    """The role of a path (vault-relative, or relative to its repo) under ``roles``."""

    for role in _PRECEDENCE:
        if roles.get(role) and path_is_excluded(relative_path, list(roles[role])):
            return role
    return REFERENCE


def doc_role(path: str, config: RagConfig) -> str:
    """The role of an indexed note or repo doc, from the globs that cover its path."""

    # Imported here: sources imports config, which imports this module.
    from second_brain.sources import split_code_source_path

    split = split_code_source_path(path)
    if split is None:
        return role_for(path, config.doc_roles)
    source = next((s for s in config.code_sources if s.name == split[0]), None)
    roles = source.doc_roles if source is not None else DEFAULT_DOC_ROLES
    return role_for(split[1], roles)
