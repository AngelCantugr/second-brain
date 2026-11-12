"""Install the packaged `second-brain` skill into each harness's skills directory.

The skill ships as package data and is copied (never symlinked: a link into a
checkout breaks under a `pipx` install). Each installed copy carries a
`.install.json` marker holding a SHA-256 of its files, which is how a later run
tells "unmodified copy of an older release" (safe to replace) from "edited by
hand" or "someone else's folder" (refuse unless forced).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.resources
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

from second_brain.parser import parse_frontmatter

TARGETS = ("claude", "agents")
MARKER = ".install.json"
SKILL_NAME = "second-brain"
LEGACY_SKILL = "second-brain-research"


@dataclass(frozen=True)
class HarnessPaths:
    """Where each harness keeps its files, resolved against an injectable home and environment."""

    home: Path
    environ: Mapping[str, str]

    @classmethod
    def from_env(
        cls, home: Path | None = None, environ: Mapping[str, str] | None = None
    ) -> HarnessPaths:
        return cls(
            Path.home() if home is None else home,
            os.environ if environ is None else environ,
        )

    @property
    def claude_home(self) -> Path:
        return self.home / ".claude"

    @property
    def codex_home(self) -> Path:
        return self._env_dir("CODEX_HOME", ".codex")

    @property
    def kimi_home(self) -> Path:
        return self._env_dir("KIMI_CODE_HOME", ".kimi-code")

    @property
    def backup_root(self) -> Path:
        return self.home / ".local" / "state" / "second-brain" / "skill-backups"

    def skills_dir(self, target: str) -> Path:
        if target == "claude":
            return self.claude_home / "skills"
        if target == "agents":
            return self.home / ".agents" / "skills"
        raise ValueError(f"unknown target: {target}")

    def target_present(self, target: str) -> bool:
        if target == "claude":
            return self.claude_home.exists()
        if target == "agents":
            return self.codex_home.exists() or self.kimi_home.exists()
        raise ValueError(f"unknown target: {target}")

    def _env_dir(self, variable: str, default_name: str) -> Path:
        # An empty value counts as unset, matching how shells export "VAR=".
        value = self.environ.get(variable, "")
        return Path(value) if value else self.home / default_name


@dataclass
class Result:
    kind: str  # "target" | "legacy" | "reference" | "mcp"
    name: str
    status: str
    message: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "detail": self.detail,
        }


def packaged_skill() -> Traversable:
    return importlib.resources.files("second_brain") / "skills" / SKILL_NAME


def tree_files(root: Traversable | Path) -> dict[str, bytes]:
    """Map relative path -> bytes for every file under `root`, excluding the top-level marker."""
    files: dict[str, bytes] = {}

    def walk(node: Traversable, prefix: str) -> None:
        for child in node.iterdir():
            relative = f"{prefix}{child.name}"
            if isinstance(child, Path) and child.is_symlink():
                # Never follow links inside an installed copy: a loop would recurse forever,
                # and a link is never part of the packaged tree. Hashing it as a distinct
                # entry makes the copy read as modified, so it is refused rather than read through.
                files[relative] = b"\0symlink\0" + os.readlink(child).encode()
            elif child.is_dir():
                walk(child, f"{relative}/")
            elif relative != MARKER:
                files[relative] = child.read_bytes()

    walk(root, "")
    return files


def tree_hash(files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(files):
        digest.update(relative.encode() + b"\0" + files[relative] + b"\0")
    return digest.hexdigest()


def backup(src: Path, paths: HarnessPaths, now: datetime, target: str) -> Path:
    """Move `src` under the state dir, never into a `skills/` dir.

    Harnesses load every folder in `skills/`, so a backup left there would
    load as a second, competing skill.
    """
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    candidate, suffix = stamp, 0
    while (paths.backup_root / candidate / target / src.name).exists():
        suffix += 1
        candidate = f"{stamp}-{suffix}"
    destination = paths.backup_root / candidate / target / src.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(destination))  # moves a symlink as a link
    return destination


def read_marker(skill_dir: Path) -> dict[str, Any] | None:
    """Return the marker only if it parses and carries a string `hash`; otherwise None."""
    try:
        marker = json.loads((skill_dir / MARKER).read_text("utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(marker, dict) and isinstance(marker.get("hash"), str):
        return marker
    return None


def _package_version() -> str:
    return importlib.metadata.version("second-brain-mcp")


def install_skill(
    paths: HarnessPaths,
    targets: Sequence[str],
    *,
    force: bool,
    dry_run: bool,
    now: datetime,
) -> list[Result]:
    # One target's failure must never stop the others.
    results = []
    for target in targets:
        try:
            results.append(_install_target(paths, target, force, dry_run, now))
        except OSError as exc:
            results.append(Result("target", target, "error", str(exc)))
    return results


def _install_target(
    paths: HarnessPaths, target: str, force: bool, dry_run: bool, now: datetime
) -> Result:
    if not paths.target_present(target):
        return Result("target", target, "not_installed", "harness home not found")

    skills = paths.skills_dir(target)
    destination = skills / SKILL_NAME
    packaged = tree_files(packaged_skill())
    packaged_hash = tree_hash(packaged)
    detail: dict[str, Any] = {"path": str(destination)}
    if dry_run:
        detail["dry_run"] = True

    replacing = os.path.lexists(destination)
    needs_backup = False
    if replacing:
        marker = read_marker(destination) if _is_real_dir(destination) else None
        on_disk = tree_hash(tree_files(destination)) if marker else None
        if marker is None or marker["hash"] != on_disk:
            if not force:
                reason = (
                    "no valid install marker"
                    if marker is None
                    else "modified since install"
                )
                return Result(
                    "target",
                    target,
                    "refused",
                    f"{reason}; rerun with --force to back up and replace",
                    detail,
                )
            needs_backup = True
        elif marker["hash"] == packaged_hash:
            return Result("target", target, "skipped", "already current", detail)

    status = "updated" if replacing else "ok"
    if dry_run:
        return Result("target", target, status, "dry run: no changes written", detail)

    skills.mkdir(parents=True, exist_ok=True)
    try:
        _swap_in(
            skills,
            destination,
            packaged,
            packaged_hash,
            paths,
            target,
            now,
            needs_backup,
            detail,
        )
    except OSError as exc:
        # detail may carry "backup" so the user can still find their edits.
        return Result("target", target, "error", str(exc), detail)
    return Result("target", target, status, "", detail)


def _is_real_dir(path: Path) -> bool:
    # A symlinked dir is not ours to read through or delete; treat it as unmanaged.
    return path.is_dir() and not path.is_symlink()


def _swap_in(
    skills: Path,
    destination: Path,
    packaged: dict[str, bytes],
    packaged_hash: str,
    paths: HarnessPaths,
    target: str,
    now: datetime,
    needs_backup: bool,
    detail: dict[str, Any],
) -> None:
    # Staging lives in skills/'s parent: same filesystem (so os.replace is atomic) but
    # outside skills/, so an interrupted install never leaves a loadable half-written folder.
    staging = skills.parent / f".second-brain-install-{os.getpid()}"
    displaced = skills.parent / f".second-brain-old-{os.getpid()}"
    try:
        if staging.exists():
            shutil.rmtree(staging)
        for relative, content in packaged.items():
            file_path = staging / relative
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(content)
        (staging / MARKER).write_text(
            json.dumps({"version": _package_version(), "hash": packaged_hash}, indent=2)
            + "\n",
            "utf-8",
        )

        if os.path.lexists(destination):
            if needs_backup:
                detail["backup"] = str(backup(destination, paths, now, target))
            else:
                # Unmodified managed copy: park it aside so a failed replace can be undone.
                os.replace(destination, displaced)
        try:
            os.replace(staging, destination)
        except OSError:
            if displaced.exists():
                os.replace(displaced, destination)
            elif "backup" in detail:
                # The user's edited copy was moved out; put it back, and if that fails too,
                # leave detail["backup"] so the error report says where their edits are.
                shutil.move(detail["backup"], str(destination))
                del detail["backup"]
            raise
        if displaced.exists():
            shutil.rmtree(displaced)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def uninstall_skill(
    paths: HarnessPaths, targets: Sequence[str], *, dry_run: bool
) -> list[Result]:
    results = []
    for target in targets:
        try:
            results.append(_uninstall_target(paths, target, dry_run))
        except OSError as exc:
            results.append(Result("target", target, "error", str(exc)))
    return results


def _uninstall_target(paths: HarnessPaths, target: str, dry_run: bool) -> Result:
    if not paths.target_present(target):
        return Result("target", target, "not_installed", "harness home not found")

    destination = paths.skills_dir(target) / SKILL_NAME
    detail: dict[str, Any] = {"path": str(destination)}
    if dry_run:
        detail["dry_run"] = True

    if not os.path.lexists(destination):
        return Result("target", target, "skipped", "nothing installed", detail)
    # Only remove what this installer wrote; an unmarked folder is somebody's own skill.
    marker = read_marker(destination) if _is_real_dir(destination) else None
    if marker is None:
        return Result(
            "target", target, "refused", "no valid install marker; not removed", detail
        )
    if marker["hash"] != tree_hash(tree_files(destination)):
        # Uninstall has no backup step, so deleting an edited copy would destroy the edits.
        return Result(
            "target",
            target,
            "refused",
            "edited since install; remove it by hand if you want it gone",
            detail,
        )
    if not dry_run:
        shutil.rmtree(destination)
    return Result(
        "target",
        target,
        "ok",
        "removed" if not dry_run else "dry run: would remove",
        detail,
    )


def retire_legacy(
    paths: HarnessPaths,
    *,
    dry_run: bool,
    now: datetime,
    targets: Sequence[str] = TARGETS,
) -> list[Result]:
    """Back up the superseded `second-brain-research` skill out of each skills dir.

    `targets` lets the caller pass only harnesses where the new skill installed, so a
    failed install never leaves a harness with neither skill. Targets with no legacy
    entry produce no result.
    """
    results = []
    for target in targets:
        if not paths.target_present(target):
            continue
        entry = paths.skills_dir(target) / LEGACY_SKILL
        if not os.path.lexists(entry):
            continue
        try:
            results.append(_retire_entry(paths, target, entry, dry_run, now))
        except OSError as exc:
            results.append(Result("legacy", target, "error", str(exc)))
    return results


def _retire_entry(
    paths: HarnessPaths, target: str, entry: Path, dry_run: bool, now: datetime
) -> Result:
    detail: dict[str, Any] = {"path": str(entry)}
    if dry_run:
        detail["dry_run"] = True
    # Read through a symlink to identify the skill, but only ever move `entry` itself:
    # backup() moves a link as a link, so a symlinked skill's real directory is untouched.
    try:
        text = (entry / "SKILL.md").read_text("utf-8", errors="replace")
    except OSError:
        return Result(
            "legacy", target, "skipped", "no readable SKILL.md; left in place", detail
        )
    if parse_frontmatter(text)[0].get("name") != LEGACY_SKILL:
        return Result(
            "legacy",
            target,
            "skipped",
            f"frontmatter name is not {LEGACY_SKILL}; left in place",
            detail,
        )
    if not dry_run:
        detail["backup"] = str(backup(entry, paths, now, target))
    return Result(
        "legacy",
        target,
        "retired",
        "moved to backup" if not dry_run else "dry run: would move to backup",
        detail,
    )


def scan_stale_references(paths: HarnessPaths) -> list[Result]:
    """Report lines in user-owned instruction files that still name the retired skill.

    Read-only by design: these files are the user's hand-written config, so we point at
    the lines and never edit them.
    """
    candidates = (
        paths.claude_home / "CLAUDE.md",
        paths.codex_home / "AGENTS.md",
        paths.kimi_home / "AGENTS.md",
    )
    results = []
    for path in candidates:
        try:
            text = path.read_text("utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if LEGACY_SKILL in line:
                results.append(
                    Result("reference", f"{path}:{lineno}", "found", line.strip())
                )
    return results


def skill_status(paths: HarnessPaths) -> list[Result]:
    """Read-only report per target: ok, absent, modified, outdated or not_installed."""
    packaged_hash = tree_hash(tree_files(packaged_skill()))
    results = []
    for target in TARGETS:
        try:
            results.append(_status_target(paths, target, packaged_hash))
        except OSError as exc:
            results.append(Result("target", target, "error", str(exc)))
    return results


def _status_target(paths: HarnessPaths, target: str, packaged_hash: str) -> Result:
    if not paths.target_present(target):
        return Result("target", target, "not_installed", "harness home not found")
    destination = paths.skills_dir(target) / SKILL_NAME
    detail: dict[str, Any] = {"path": str(destination)}
    if not os.path.lexists(destination):
        return Result("target", target, "absent", "not installed", detail)

    marker = read_marker(destination) if _is_real_dir(destination) else None
    on_disk = tree_hash(tree_files(destination)) if _is_real_dir(destination) else None
    # A folder we did not write (or whose marker is unreadable) counts as modified: it is
    # not a managed copy, and install would refuse it just the same.
    modified = marker is None or marker["hash"] != on_disk
    update_available = (on_disk if on_disk is not None else "") != packaged_hash
    version = marker.get("version") if marker else None
    detail.update(
        installed_version=version if isinstance(version, str) else None,
        modified=modified,
        update_available=update_available,
    )
    if modified:
        return Result(
            "target", target, "modified", "differs from its install marker", detail
        )
    if update_available:
        return Result(
            "target", target, "outdated", "a newer packaged skill exists", detail
        )
    return Result("target", target, "ok", "", detail)


def exit_code(action: str, results: Sequence[Result]) -> int:
    """1 only for a failed install/uninstall of a target or legacy skill; MCP and reference are warnings."""
    if action == "status":
        return 0
    failed = any(
        r.kind in ("target", "legacy") and r.status in ("error", "refused")
        for r in results
    )
    return 1 if failed else 0


def run_skills(
    action: str,
    *,
    targets: Sequence[str],
    force: bool,
    dry_run: bool,
    config_path: Path,
    paths: HarnessPaths,
    now: datetime,
    runner: Any = None,
    which: Any = shutil.which,
) -> tuple[list[Result], int]:
    # Imported here because harness_mcp imports Result/HarnessPaths from this module.
    from second_brain.harness_mcp import check_mcp

    if action == "install":
        results = install_skill(paths, targets, force=force, dry_run=dry_run, now=now)
        # Retire the old skill only where the new one is in place (or would be), so a
        # failed install never leaves a harness with neither.
        covered = [
            r.name
            for r in results
            if r.kind == "target" and r.status in ("ok", "updated", "skipped")
        ]
        if covered:
            results += retire_legacy(paths, dry_run=dry_run, now=now, targets=covered)
        results += scan_stale_references(paths)
        results += check_mcp(paths, config_path, runner=runner, which=which)
    elif action == "uninstall":
        results = uninstall_skill(paths, targets, dry_run=dry_run)
    elif action == "status":
        results = skill_status(paths) + check_mcp(
            paths, config_path, runner=runner, which=which
        )
    else:
        raise ValueError(f"unknown action: {action}")
    return results, exit_code(action, results)
