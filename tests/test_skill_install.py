"""Installer core: copies the packaged skill into harness skill dirs with a content-hash marker."""

import importlib.metadata
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from second_brain.skill_install import (
    LEGACY_SKILL,
    MARKER,
    TARGETS,
    HarnessPaths,
    Result,
    backup,
    exit_code,
    install_skill,
    packaged_skill,
    retire_legacy,
    scan_stale_references,
    skill_status,
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


def test_uninstall_refuses_hand_edited_copy(paths):
    _install(paths)
    skill_md = paths.skills_dir("claude") / "second-brain" / "SKILL.md"
    skill_md.write_text("my edits")

    res = uninstall_skill(paths, TARGETS, dry_run=False)

    assert [r.status for r in res] == ["refused", "ok"]
    assert skill_md.read_text() == "my edits"


def test_symlink_loop_in_installed_copy_is_refused_and_other_target_installs(paths):
    _install(paths)
    d = paths.skills_dir("claude") / "second-brain"
    (d / "loop").symlink_to(d)
    shutil.rmtree(paths.skills_dir("agents") / "second-brain")

    res = _install(paths)

    assert [r.status for r in res] == ["refused", "ok"]
    assert (d / "loop").is_symlink()
    assert uninstall_skill(paths, ["claude"], dry_run=False)[0].status == "refused"


def _fail_final_swap(monkeypatch):
    real_replace = os.replace

    def replace(src, dst):
        if Path(src).name.startswith(".second-brain-install"):
            raise OSError("disk exploded")
        real_replace(src, dst)

    monkeypatch.setattr("second_brain.skill_install.os.replace", replace)


def test_failed_forced_install_restores_edited_copy(paths, monkeypatch):
    _install(paths)
    skill_md = paths.skills_dir("claude") / "second-brain" / "SKILL.md"
    skill_md.write_text("my edits")
    _fail_final_swap(monkeypatch)

    res = _install(paths, force=True)

    assert res[0].status == "error" and "disk exploded" in res[0].message
    assert "backup" not in res[0].detail
    assert skill_md.read_text() == "my edits"


def test_failed_forced_install_reports_backup_when_restore_fails(paths, monkeypatch):
    _install(paths)
    skill_md = paths.skills_dir("claude") / "second-brain" / "SKILL.md"
    skill_md.write_text("my edits")
    _fail_final_swap(monkeypatch)
    real_move = shutil.move
    calls = []

    def move(src, dst):
        calls.append(src)
        if len(calls) > 1:
            raise OSError("restore failed")
        return real_move(src, dst)

    monkeypatch.setattr("second_brain.skill_install.shutil.move", move)

    res = _install(paths, force=True)

    assert res[0].status == "error"
    assert (Path(res[0].detail["backup"]) / "SKILL.md").read_text() == "my edits"


def _legacy(paths, target, name=LEGACY_SKILL):
    skill = paths.skills_dir(target) / LEGACY_SKILL
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(f"---\nname: {name}\n---\nbody\n")
    return skill


def test_retire_moves_matching_legacy_from_both_dirs(paths):
    for target in TARGETS:
        _legacy(paths, target)
    res = retire_legacy(paths, dry_run=False, now=NOW)
    assert [r.status for r in res] == ["retired", "retired"]
    assert not any((paths.skills_dir(t) / LEGACY_SKILL).exists() for t in TARGETS)
    backed_up = paths.backup_root / "20261112T080000Z" / "agents" / LEGACY_SKILL
    assert (backed_up / "SKILL.md").is_file()
    assert res[1].detail["backup"] == str(backed_up)


def test_retire_without_legacy_or_harness_yields_no_results(tmp_path):
    (tmp_path / ".claude").mkdir()
    assert retire_legacy(HarnessPaths(tmp_path, {}), dry_run=False, now=NOW) == []
    # agents harness absent: a stray legacy dir there is not ours to touch
    stray = tmp_path / ".agents" / "skills" / LEGACY_SKILL
    stray.mkdir(parents=True)
    (stray / "SKILL.md").write_text(f"---\nname: {LEGACY_SKILL}\n---\n")
    assert retire_legacy(HarnessPaths(tmp_path, {}), dry_run=False, now=NOW) == []
    assert stray.exists()


def test_retire_skips_dir_with_other_frontmatter_name(paths):
    skill = _legacy(paths, "claude", name="something-else")
    [r] = retire_legacy(paths, dry_run=False, now=NOW)
    assert r.status == "skipped" and skill.is_dir()


def test_retire_skips_unreadable_or_dangling_entries(paths):
    bare = paths.skills_dir("claude") / LEGACY_SKILL
    bare.mkdir(parents=True)  # no SKILL.md
    agents = paths.skills_dir("agents")
    agents.mkdir(parents=True)
    (agents / LEGACY_SKILL).symlink_to(paths.home / "nowhere")
    res = retire_legacy(paths, dry_run=False, now=NOW)
    assert [r.status for r in res] == ["skipped", "skipped"]
    assert bare.is_dir() and (agents / LEGACY_SKILL).is_symlink()


def test_retire_moves_symlink_not_its_target(paths, tmp_path):
    real = tmp_path / "elsewhere" / "real-skill"
    real.mkdir(parents=True)
    (real / "SKILL.md").write_text(f"---\nname: {LEGACY_SKILL}\n---\n")
    agents = paths.skills_dir("agents")
    agents.mkdir(parents=True)
    (agents / LEGACY_SKILL).symlink_to(real)
    [r] = retire_legacy(paths, dry_run=False, now=NOW)
    assert r.status == "retired"
    assert os.path.lexists(agents / LEGACY_SKILL) is False
    assert Path(r.detail["backup"]).is_symlink()
    assert (real / "SKILL.md").is_file()


def test_retire_never_touches_other_symlinked_skills(paths, tmp_path):
    other = tmp_path / "elsewhere" / "superpowers"
    other.mkdir(parents=True)
    agents = paths.skills_dir("agents")
    agents.mkdir(parents=True)
    (agents / "superpowers").symlink_to(other)
    assert retire_legacy(paths, dry_run=False, now=NOW) == []
    assert (agents / "superpowers").is_symlink() and other.is_dir()


def test_retire_dry_run_moves_nothing(paths):
    skill = _legacy(paths, "claude")
    [r] = retire_legacy(paths, dry_run=True, now=NOW)
    assert r.status == "retired" and r.detail["dry_run"] is True
    assert (skill / "SKILL.md").is_file() and not paths.backup_root.exists()


def test_retire_only_looks_in_requested_targets(paths):
    for target in TARGETS:
        _legacy(paths, target)
    [r] = retire_legacy(paths, dry_run=False, now=NOW, targets=("claude",))
    assert r.name == "claude" and r.status == "retired"
    assert (paths.skills_dir("agents") / LEGACY_SKILL).is_dir()


def test_retire_reports_os_error_per_entry(paths, monkeypatch):
    _legacy(paths, "claude")
    _legacy(paths, "agents")
    real_move = shutil.move

    def flaky(src, dst):
        if "/.agents/" in str(src):
            raise OSError("disk says no")
        return real_move(src, dst)

    monkeypatch.setattr(shutil, "move", flaky)
    res = retire_legacy(paths, dry_run=False, now=NOW)
    assert [r.status for r in res] == ["retired", "error"]


def test_stale_reference_scan_reports_path_and_line(paths):
    (paths.claude_home / "CLAUDE.md").write_text(
        "a\nuse the `second-brain-research` skill\n"
    )
    [r] = scan_stale_references(paths)
    assert (r.kind, r.status, r.name) == (
        "reference",
        "found",
        f"{paths.claude_home / 'CLAUDE.md'}:2",
    )
    assert r.message == "use the `second-brain-research` skill"


def test_stale_reference_scan_checks_codex_and_kimi_and_skips_missing(paths):
    (paths.codex_home / "AGENTS.md").write_text("second-brain-research\n")
    assert [r.name for r in scan_stale_references(paths)] == [
        f"{paths.codex_home / 'AGENTS.md'}:1"
    ]


def test_stale_reference_scan_never_writes(paths):
    target = paths.claude_home / "CLAUDE.md"
    target.write_text("second-brain-research\n")
    before = (target.read_bytes(), target.stat().st_mtime_ns)
    scan_stale_references(paths)
    assert (target.read_bytes(), target.stat().st_mtime_ns) == before


# --- status, exit codes, CLI -------------------------------------------------


def _statuses(results):
    return {r.name: r.status for r in results if r.kind == "target"}


def test_status_reports_ok_modified_outdated_absent(paths):
    assert _statuses(skill_status(paths)) == {"claude": "absent", "agents": "absent"}

    _install(paths)
    results = skill_status(paths)
    assert _statuses(results) == {"claude": "ok", "agents": "ok"}
    assert results[0].detail["modified"] is False
    assert results[0].detail["update_available"] is False
    assert results[0].detail["installed_version"]

    claude_skill = paths.skills_dir("claude") / "second-brain"
    (claude_skill / "SKILL.md").write_text("edited", "utf-8")
    modified = {r.name: r for r in skill_status(paths)}["claude"]
    assert modified.status == "modified"
    assert modified.detail["modified"] is True

    # A different valid tree with a matching marker is an older release, not an edit.
    agents_skill = paths.skills_dir("agents") / "second-brain"
    (agents_skill / "SKILL.md").write_text("older release", "utf-8")
    old_hash = tree_hash(tree_files(agents_skill))
    (agents_skill / MARKER).write_text(
        json.dumps({"version": "0.0.1", "hash": old_hash}), "utf-8"
    )
    outdated = {r.name: r for r in skill_status(paths)}["agents"]
    assert outdated.status == "outdated"
    assert outdated.detail["update_available"] is True
    assert outdated.detail["installed_version"] == "0.0.1"


def test_status_unmarked_or_corrupt_marker_is_modified(paths):
    unmarked = paths.skills_dir("claude") / "second-brain"
    unmarked.mkdir(parents=True)
    (unmarked / "SKILL.md").write_text("mine", "utf-8")
    result = {r.name: r for r in skill_status(paths)}["claude"]
    assert result.status == "modified"
    assert result.detail["installed_version"] is None
    assert result.detail["modified"] is True
    assert result.detail["update_available"] is True

    (unmarked / MARKER).write_text("{not json", "utf-8")
    assert {r.name: r for r in skill_status(paths)}["claude"].status == "modified"


def test_status_not_installed_without_harness(tmp_path):
    results = skill_status(HarnessPaths(tmp_path, {}))
    assert _statuses(results) == {"claude": "not_installed", "agents": "not_installed"}


def test_status_writes_nothing(paths, tmp_path):
    _install(paths)
    (paths.skills_dir("claude") / "second-brain" / "SKILL.md").write_text("x", "utf-8")
    before = _snapshot(tmp_path)
    skill_status(paths)
    assert _snapshot(tmp_path) == before


def test_exit_codes():
    assert (
        exit_code(
            "install",
            [Result("target", "claude", "ok"), Result("mcp", "kimi", "missing")],
        )
        == 0
    )
    assert exit_code("install", [Result("target", "agents", "refused")]) == 1
    assert exit_code("uninstall", [Result("target", "claude", "error")]) == 1
    assert exit_code("status", [Result("target", "claude", "error")]) == 0
    assert (
        exit_code(
            "install",
            [Result("mcp", "codex", "error"), Result("reference", "x:1", "found")],
        )
        == 0
    )
    assert exit_code("install", [Result("legacy", "agents", "error")]) == 1


def _run_cli(monkeypatch, capsys, *argv):
    from second_brain import cli

    monkeypatch.setattr(sys, "argv", ["second-brain", *argv])
    with pytest.raises(SystemExit) as raised:
        cli.main()
    return raised.value.code, json.loads(capsys.readouterr().out)


def _isolate_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.delenv("KIMI_CODE_HOME", raising=False)
    monkeypatch.setenv("PATH", "")


def test_cli_install_end_to_end(tmp_path, monkeypatch, capsys):
    _isolate_home(monkeypatch, tmp_path)
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".codex").mkdir()
    legacy = tmp_path / ".claude" / "skills" / LEGACY_SKILL
    legacy.mkdir(parents=True)
    (legacy / "SKILL.md").write_text(f"---\nname: {LEGACY_SKILL}\n---\nold", "utf-8")
    cfg = tmp_path / "rag_config.toml"
    cfg.write_text("", "utf-8")

    code, out = _run_cli(monkeypatch, capsys, "--config", str(cfg), "skills", "install")

    assert code == 0
    assert out["action"] == "install"
    by_key = {(r["kind"], r["name"]): r for r in out["results"]}
    assert by_key[("target", "claude")]["status"] == "ok"
    assert by_key[("target", "agents")]["status"] == "ok"
    assert by_key[("legacy", "claude")]["status"] == "retired"
    assert by_key[("mcp", "codex")]["status"] == "missing"
    assert str(cfg.resolve()) in by_key[("mcp", "codex")]["message"] + json.dumps(
        by_key[("mcp", "codex")]["detail"]
    )
    backup_root = tmp_path / ".local/state/second-brain/skill-backups"
    assert any(backup_root.rglob(LEGACY_SKILL))
    assert not legacy.exists()


def test_cli_install_keeps_legacy_where_install_failed(tmp_path, monkeypatch, capsys):
    _isolate_home(monkeypatch, tmp_path)
    (tmp_path / ".claude").mkdir()
    blocker = tmp_path / ".claude" / "skills" / "second-brain"
    blocker.mkdir(parents=True)  # unmarked: install is refused
    legacy = tmp_path / ".claude" / "skills" / LEGACY_SKILL
    legacy.mkdir()
    (legacy / "SKILL.md").write_text(f"---\nname: {LEGACY_SKILL}\n---\n", "utf-8")
    cfg = tmp_path / "rag_config.toml"
    cfg.write_text("", "utf-8")

    code, out = _run_cli(monkeypatch, capsys, "--config", str(cfg), "skills", "install")

    assert code == 1
    assert legacy.exists()
    assert not [r for r in out["results"] if r["kind"] == "legacy"]


def test_cli_skills_does_not_load_config(tmp_path, monkeypatch, capsys):
    _isolate_home(monkeypatch, tmp_path)
    code, out = _run_cli(
        monkeypatch,
        capsys,
        "--config",
        str(tmp_path / "missing.toml"),
        "skills",
        "status",
    )
    assert code == 0
    assert out["action"] == "status"
