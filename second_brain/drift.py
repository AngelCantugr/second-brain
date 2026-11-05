"""Drift between notes and the code they mention.

A ``mentions_code`` edge says a note names some code; it does not say whether
the two still agree. This module keeps, per ``(note, mention)``, the code
fingerprint the note was last written against, so a later sync can tell:

- ``stale``: the code changed and the note has not changed since. The note
  may still be right; nothing here reads either text.
- ``dangling``: a mention that used to resolve no longer does, because the
  file or symbol is gone. A name that also starts matching other files keeps
  its old file while that file still matches (``typed_graph``). The edge
  itself disappears with the next sync; this row is what remembers it.
- ``ok``: neither of the above.

"Written against" is approximated by the note's content hash: whenever the
note changes, every mention in it takes the code's current fingerprint. An
unrelated edit to the note therefore clears its stale mentions too; the
alternative, asking the user to mark each mention reviewed, was judged too
much ceremony for a personal vault.

Whether a stale note actually disagrees with its code takes reading both,
which the server leaves to the calling agent (``rag.drift``).
"""

from __future__ import annotations

import difflib
import posixpath
from collections.abc import Collection, Iterable
from dataclasses import dataclass

from second_brain.code_graph import _GITHUB_RE, MentionTarget
from second_brain.sources import split_code_source_path

OK = "ok"
STALE = "stale"
DANGLING = "dangling"
STATUSES = (OK, STALE, DANGLING)

# What a mention names, from most to least specific. A repo mention resolves
# to the root README, which changes in nearly every pull request, so on its own
# it says little about drift.
SYMBOL = "symbol"
FILE = "file"
REPO = "repo"
LEVELS = (SYMBOL, FILE, REPO)

# Why a dangling mention stopped resolving.
FILE_REMOVED = "file_removed"
SYMBOL_REMOVED = "symbol_removed"
UNRESOLVED = "unresolved"  # still there, but the mention no longer resolves to it


@dataclass(slots=True)
class MentionState:
    """One mention in one note and the code it pointed at, across syncs."""

    src: str
    mention: str
    target: str
    # Qualname when the mention names a symbol; None when it names the file.
    symbol: str | None
    # The note's content hash the last time this row was computed.
    src_hash: str
    # The code's fingerprint when the note last changed (what it was written
    # against): ``<hash>@<path>::<symbol>``, so a move changes it too.
    acked_hash: str | None
    # The code's fingerprint now; None once the mention is dangling.
    current_hash: str | None
    # The target file's mtime at the sync that first saw ``current_hash``;
    # None while it is the fingerprint first recorded (no change seen yet).
    changed_at: float | None
    # When the mention stopped resolving.
    lost_at: float | None
    status: str
    reason: str | None = None


def mention_level(state: MentionState) -> str:
    """Whether a mention names a symbol, a file, or a whole repo."""

    if state.symbol is not None:
        return SYMBOL
    text = state.mention.strip().strip("'\"").rstrip(".,;:!?")
    github = _GITHUB_RE.match(text)
    if github is not None:
        return FILE if github.group("path") else REPO
    split = split_code_source_path(state.target)
    return REPO if split is not None and text.lower() == split[0].lower() else FILE


def fingerprint(target: MentionTarget, meta_by_path: dict[str, dict]) -> str | None:
    """The target symbol's body hash, or the file's content hash, tagged with
    which file and symbol it belongs to; None if unknown.

    The tag makes a move count: ``compute_backoff`` moved unchanged from
    ``a.py`` to ``b.py`` keeps its body hash, but the note was written about
    ``a.py``, whose imports and neighbors it may describe.

    Rows written before hashes were recorded have none, which reads as
    "unknown" rather than "changed" until their file is indexed again.
    """

    meta = meta_by_path.get(target.path)
    if meta is None:
        return None
    code = meta.get("code", {})
    if target.symbol is not None:
        digest = code.get("symbol_hashes", {}).get(target.symbol)
    else:
        digest = code.get("hash")
    return None if digest is None else f"{digest}@{target.path}::{target.symbol or ''}"


def _dangling_reason(previous: MentionState, meta_by_path: dict[str, dict]) -> str:
    meta = meta_by_path.get(previous.target)
    if meta is None:
        return FILE_REMOVED
    if previous.symbol is not None and previous.symbol not in meta.get("code", {}).get("symbols", []):
        return SYMBOL_REMOVED
    return UNRESOLVED


def mention_states(
    metas: Iterable[dict],
    resolved: dict[tuple[str, str], MentionTarget | None],
    previous: dict[tuple[str, str], MentionState],
    now: float,
) -> list[MentionState]:
    """Compute every mention's drift state from this sync's resolution and the last one.

    Notes without a recorded content hash (indexed before drift tracking)
    are skipped until they are indexed again. A mention the note no longer
    contains, or a note that is gone, drops its row, and so does one that
    ``resolved`` maps to None: it still matches code, but is no longer taken
    to mean it (see ``typed_graph``), so it neither drifts nor dangles.
    """

    metas = list(metas)
    meta_by_path = {meta["path"]: meta for meta in metas}
    states: list[MentionState] = []
    for meta in metas:
        code = meta.get("code", {})
        src_hash = code.get("hash")
        if src_hash is None:
            continue
        for mention in dict.fromkeys(code.get("mentions", [])):
            key = (meta["path"], mention)
            if key in resolved and resolved[key] is None:
                continue
            prev = previous.get(key)
            target = resolved.get(key)
            if target is None:
                if prev is None:
                    continue  # never named code, as far as we know
                # Keep what it used to point at, so the agent can look for
                # the rename; the first sync that lost it dates the loss.
                states.append(
                    MentionState(
                        src=meta["path"],
                        mention=mention,
                        target=prev.target,
                        symbol=prev.symbol,
                        src_hash=src_hash,
                        acked_hash=prev.acked_hash,
                        current_hash=None,
                        changed_at=prev.changed_at,
                        lost_at=prev.lost_at if prev.status == DANGLING else now,
                        status=DANGLING,
                        reason=_dangling_reason(prev, meta_by_path),
                    )
                )
                continue
            current = fingerprint(target, meta_by_path)
            note_changed = prev is None or prev.src_hash != src_hash or prev.acked_hash is None
            acked = current if note_changed else prev.acked_hash
            if prev is not None and prev.current_hash is not None and prev.current_hash != current:
                changed_at = meta_by_path.get(target.path, {}).get("code", {}).get("mtime")
            else:
                changed_at = None if prev is None else prev.changed_at
            # A retarget (the name now resolves to another file or symbol)
            # with an unchanged note is drift too: the fingerprints differ.
            stale = acked is not None and current is not None and acked != current
            states.append(
                MentionState(
                    src=meta["path"],
                    mention=mention,
                    target=target.path,
                    symbol=target.symbol,
                    src_hash=src_hash,
                    acked_hash=acked,
                    current_hash=current,
                    changed_at=changed_at,
                    lost_at=None,
                    status=STALE if stale else OK,
                )
            )
    return states


def rename_candidates(state: MentionState, metas: Iterable[dict], limit: int = 3) -> list[str]:
    """Where a dangling mention's code may have gone, for the agent to check.

    A removed symbol suggests close names in the same file, then the same
    name elsewhere in the repo (a move); a removed file suggests files of the
    same name in the repo; a name that became ambiguous lists the files that
    now define it. Symbols are written ``<path>::<qualname>``.
    """

    repo_prefix = state.target.split("/", 1)[0] + "/"
    metas = [meta for meta in metas if meta["path"].startswith(repo_prefix)]
    if state.reason == FILE_REMOVED:
        basename = posixpath.basename(state.target)
        return sorted(m["path"] for m in metas if posixpath.basename(m["path"]) == basename)[:limit]
    if state.symbol is None:
        return []
    terminal = state.symbol.rsplit(".", 1)[-1]
    in_file = next((m for m in metas if m["path"] == state.target), None)
    found: list[str] = []
    if in_file is not None and state.reason == SYMBOL_REMOVED:
        symbols = in_file.get("code", {}).get("symbols", [])
        by_terminal: dict[str, list[str]] = {}
        for qualname in symbols:
            by_terminal.setdefault(qualname.rsplit(".", 1)[-1], []).append(qualname)
        for name in difflib.get_close_matches(terminal, list(by_terminal), n=limit, cutoff=0.6):
            found += [f"{state.target}::{q}" for q in by_terminal[name]]
    for meta in sorted(metas, key=lambda m: m["path"]):
        for qualname in meta.get("code", {}).get("symbols", []):
            if qualname.rsplit(".", 1)[-1] == terminal and meta["path"] != state.target:
                found.append(f"{meta['path']}::{qualname}")
    return list(dict.fromkeys(found))[:limit]


# Mentions listed per record summary: enough to see what went missing.
MAX_GONE_PER_RECORD = 5
# Unmoved mentions assumed for every record when ranking, so a record naming
# one thing cannot outrank one naming fifteen on share alone.
RANK_PRIOR = 2


def record_summaries(states: Iterable[MentionState], levels: Collection[str]) -> list[dict]:
    """How far the code has moved from each record (a design, plan or spec).

    A record is not updated as code moves on, so its stale pairs one by one
    say little. Per record, the share of the code it names that has changed
    or gone since it was written does: a record whose code is mostly moved was
    superseded, or was built differently from what it says. Only records with
    any movement are returned, most moved first: by share, damped for
    records that name only a few things (one of one moved ranks below 14 of
    15).
    ``changed_at`` is the latest code change or loss among its mentions.
    """

    by_src: dict[str, dict] = {}
    for state in states:
        if mention_level(state) not in levels:
            continue
        entry = by_src.setdefault(
            state.src,
            {"path": state.src, "tracked": 0, "changed": 0, "gone": 0, "changed_at": None, "gone_mentions": []},
        )
        entry["tracked"] += 1
        if state.status == STALE:
            entry["changed"] += 1
        elif state.status == DANGLING:
            entry["gone"] += 1
            entry["gone_mentions"].append(state.mention)
        when = state.lost_at if state.status == DANGLING else state.changed_at
        if state.status != OK and when is not None:
            entry["changed_at"] = max(entry["changed_at"] or when, when)
    moved = [entry for entry in by_src.values() if entry["changed"] or entry["gone"]]
    for entry in moved:
        entry["moved_share"] = round((entry["changed"] + entry["gone"]) / entry["tracked"], 3)
        entry["gone_mentions"] = sorted(entry["gone_mentions"])[:MAX_GONE_PER_RECORD]
    moved.sort(
        key=lambda e: (-(e["changed"] + e["gone"]) / (e["tracked"] + RANK_PRIOR), e["path"])
    )
    return moved
