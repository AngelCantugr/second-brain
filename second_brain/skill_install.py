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

TARGETS = ("claude", "agents")
MARKER = ".install.json"
SKILL_NAME = "second-brain"


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
