"""Installer core: copies the packaged skill into harness skill dirs with a content-hash marker."""

import importlib.metadata
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from second_brain.skill_install import (
    MARKER,
    TARGETS,
    HarnessPaths,
    Result,
    backup,
    install_skill,
    packaged_skill,
    tree_files,
    tree_hash,
    uninstall_skill,
)

NOW = datetime(2026, 11, 12, 8, 0, tzinfo=UTC)


@pytest.fixture
def paths(tmp_path):
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".codex").mkdir()
    return HarnessPaths(tmp_path, {})


def _install(paths, **kwargs):
    options = {"force": False, "dry_run": False, "now": NOW} | kwargs
    return install_skill(paths, TARGETS, **options)


def _snapshot(root: Path) -> dict[str, bytes | None]:
    return {
        str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None)
        for p in sorted(root.rglob("*"))
    }


def test_env_paths_default_and_empty_means_unset(tmp_path):
    p = HarnessPaths(tmp_path, {"CODEX_HOME": "", "KIMI_CODE_HOME": "/k"})
    assert p.codex_home == tmp_path / ".codex" and p.kimi_home == Path("/k")


def test_from_env_defaults_to_given_home(tmp_path):
    p = HarnessPaths.from_env(tmp_path, {})
    assert p.claude_home == tmp_path / ".claude"
    assert p.kimi_home == tmp_path / ".kimi-code"
    assert p.backup_root == tmp_path / ".local/state/second-brain/skill-backups"
    assert p.skills_dir("agents") == tmp_path / ".agents/skills"


def test_result_to_dict_shape():
    r = Result("target", "claude", "ok", "done", {"path": "/x"})
    assert r.to_dict() == {
        "kind": "target",
        "name": "claude",
        "status": "ok",
        "message": "done",
        "detail": {"path": "/x"},
    }


def test_fresh_install_writes_both_targets_with_marker(paths):
    res = _install(paths)
    assert [r.status for r in res] == ["ok", "ok"]
    for t in TARGETS:
        d = paths.skills_dir(t) / "second-brain"
        marker = json.loads((d / MARKER).read_text())
        assert (
            marker["hash"]
            == tree_hash(tree_files(packaged_skill()))
            == tree_hash(tree_files(d))
        )
        assert marker["version"] == importlib.metadata.version("second-brain-mcp")


def test_reinstall_current_copy_is_skipped(paths):
    _install(paths)
    assert [r.status for r in _install(paths)] == ["skipped", "skipped"]


def test_update_replaces_tree_and_drops_removed_files(paths):
    _install(paths)
    d = paths.skills_dir("claude") / "second-brain"
    (d / "references" / "old.md").write_text("from an older release")
    (d / MARKER).write_text(
        json.dumps({"version": "0", "hash": tree_hash(tree_files(d))})
    )

    res = _install(paths)

    assert [r.status for r in res] == ["updated", "skipped"]
    assert not (d / "references" / "old.md").exists()
    assert tree_files(d) == tree_files(packaged_skill())


def test_hand_edited_copy_is_refused_without_force(paths):
    _install(paths)
    skill_md = paths.skills_dir("claude") / "second-brain" / "SKILL.md"
    skill_md.write_text("my edits")

    res = _install(paths)

    assert [r.status for r in res] == ["refused", "skipped"]
    assert skill_md.read_text() == "my edits"


def test_force_backs_up_then_replaces(paths):
    _install(paths)
    skill_md = paths.skills_dir("claude") / "second-brain" / "SKILL.md"
    skill_md.write_text("my edits")

    res = _install(paths, force=True)

    assert res[0].status == "updated"
    backed_up = (
        paths.backup_root / "20261112T080000Z" / "claude" / "second-brain" / "SKILL.md"
    )
    assert backed_up.read_text() == "my edits"
    assert tree_files(skill_md.parent) == tree_files(packaged_skill())
    for t in TARGETS:
        names = [p.name for p in paths.skills_dir(t).glob("second-brain*")]
        assert names == ["second-brain"]


def test_corrupt_marker_is_refused(paths):
    _install(paths)
    marker = paths.skills_dir("claude") / "second-brain" / MARKER
    marker.write_text("not json")
    assert _install(paths)[0].status == "refused"
    assert _install(paths, force=True)[0].status == "updated"


def test_marker_without_hash_is_refused(paths):
    _install(paths)
    (paths.skills_dir("claude") / "second-brain" / MARKER).write_text(
        '{"version": "1"}'
    )
    assert _install(paths)[0].status == "refused"


def test_unmarked_existing_dir_is_refused(paths):
    d = paths.skills_dir("claude") / "second-brain"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("someone else's skill")
    assert _install(paths)[0].status == "refused"
    assert (d / "SKILL.md").read_text() == "someone else's skill"


def test_dry_run_writes_nothing(paths):
    before = _snapshot(paths.home)
    res = _install(paths, dry_run=True)
    assert _snapshot(paths.home) == before
    assert [r.status for r in res] == ["ok", "ok"]
    assert all(r.detail["dry_run"] is True for r in res)


def test_dry_run_reports_update_and_refusal_without_touching_disk(paths):
    _install(paths)
    (paths.skills_dir("claude") / "second-brain" / "SKILL.md").write_text("edited")
    before = _snapshot(paths.home)
    res = _install(paths, dry_run=True)
    assert [r.status for r in res] == ["refused", "skipped"]
    assert _snapshot(paths.home) == before


def test_agents_target_not_installed_without_codex_or_kimi(tmp_path):
    (tmp_path / ".claude").mkdir()
    res = install_skill(
        HarnessPaths(tmp_path, {}), TARGETS, force=False, dry_run=False, now=NOW
    )
    assert [r.status for r in res] == ["ok", "not_installed"]
    assert not (tmp_path / ".agents").exists()


def test_claude_target_not_installed_without_claude_home(tmp_path):
    (tmp_path / ".kimi-code").mkdir()
    res = install_skill(
        HarnessPaths(tmp_path, {}), TARGETS, force=False, dry_run=False, now=NOW
    )
    assert [r.status for r in res] == ["not_installed", "ok"]
    assert not (tmp_path / ".claude").exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_unwritable_target_reports_error_and_other_installs(paths):
    skills = paths.skills_dir("claude")
    skills.mkdir()
    skills.chmod(0o500)
    try:
        res = _install(paths)
    finally:
        skills.chmod(0o700)

    assert [r.status for r in res] == ["error", "ok"]
    assert res[0].message
    leftovers = [
        p.name
        for p in paths.claude_home.iterdir()
        if p.name.startswith(".second-brain")
    ]
    assert leftovers == []


def test_uninstall_removes_only_marked_dirs(paths):
    _install(paths)
    agents_copy = paths.skills_dir("agents") / "second-brain"
    (agents_copy / MARKER).unlink()

    res = uninstall_skill(paths, TARGETS, dry_run=False)

    assert [r.status for r in res] == ["ok", "refused"]
    assert not (paths.skills_dir("claude") / "second-brain").exists()
    assert agents_copy.exists()


def test_uninstall_statuses_for_absent_and_missing_harness(tmp_path):
    (tmp_path / ".claude").mkdir()
    res = uninstall_skill(HarnessPaths(tmp_path, {}), TARGETS, dry_run=False)
    assert [r.status for r in res] == ["skipped", "not_installed"]


def test_uninstall_dry_run_keeps_files(paths):
    _install(paths)
    before = _snapshot(paths.home)
    res = uninstall_skill(paths, TARGETS, dry_run=True)
    assert [r.status for r in res] == ["ok", "ok"]
    assert all(r.detail["dry_run"] is True for r in res)
    assert _snapshot(paths.home) == before


def test_backup_name_collision_gets_suffix(paths):
    first = paths.skills_dir("claude") / "second-brain"
    first.mkdir(parents=True)
    (first / "a").write_text("1")
    one = backup(first, paths, NOW, "claude")
    first.mkdir()
    (first / "a").write_text("2")
    two = backup(first, paths, NOW, "claude")

    assert one != two
    assert (one / "a").read_text() == "1" and (two / "a").read_text() == "2"
    assert one.parts[-3] == "20261112T080000Z"
    assert two.parts[-3] == "20261112T080000Z-1"
