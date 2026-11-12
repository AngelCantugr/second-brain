import json
import shlex
import subprocess
import tomllib

import pytest

from second_brain.harness_mcp import check_mcp
from second_brain.skill_install import HarnessPaths


@pytest.fixture
def paths(tmp_path):
    for name in (".claude", ".codex", ".kimi-code"):
        (tmp_path / name).mkdir()
    return HarnessPaths(tmp_path, {})


@pytest.fixture
def cfg(tmp_path):
    path = tmp_path / "rag_config.toml"
    path.write_text("")
    return path


def fake(code):
    calls = []

    def runner(argv):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, code, "", "")

    runner.calls = calls
    return runner


def by_name(results):
    assert all(r.kind == "mcp" for r in results)
    return {r.name: r for r in results}


def test_results_are_ordered_claude_codex_kimi(paths, cfg):
    results = check_mcp(paths, cfg, runner=fake(0), which=lambda _: "/bin/claude")
    assert [r.name for r in results] == ["claude", "codex", "kimi"]


def test_codex_present(paths, cfg):
    (paths.codex_home / "config.toml").write_text(
        '[mcp_servers.second-brain]\ncommand = "second-brain-mcp"\n'
    )
    results = by_name(check_mcp(paths, cfg, runner=fake(1), which=lambda _: None))
    assert results["codex"].status == "present"


def test_codex_missing_snippet_is_valid_toml(paths, cfg):
    (paths.codex_home / "config.toml").write_text(
        '[mcp_servers.other]\ncommand = "x"\n'
    )
    r = by_name(check_mcp(paths, cfg, runner=fake(1), which=lambda _: None))["codex"]
    assert r.status == "missing"
    assert tomllib.loads(r.detail["fix"])["mcp_servers"]["second-brain"]["args"] == [
        "--config",
        str(cfg.resolve()),
    ]
    assert r.detail["config"] == str(paths.codex_home / "config.toml")


def test_codex_missing_file_is_missing(paths, cfg):
    r = by_name(check_mcp(paths, cfg, runner=fake(1), which=lambda _: None))["codex"]
    assert r.status == "missing"


def test_kimi_reads_kimi_code_home_env(tmp_path, cfg):
    kimi = tmp_path / "elsewhere"
    kimi.mkdir()
    (kimi / "mcp.json").write_text(json.dumps({"mcpServers": {"second-brain": {}}}))
    paths = HarnessPaths(tmp_path, {"KIMI_CODE_HOME": str(kimi)})
    r = by_name(check_mcp(paths, cfg, runner=fake(1), which=lambda _: None))["kimi"]
    assert r.status == "present"


def test_kimi_missing_snippet_is_valid_json(paths, cfg):
    (paths.kimi_home / "mcp.json").write_text(json.dumps({"mcpServers": {}}))
    r = by_name(check_mcp(paths, cfg, runner=fake(1), which=lambda _: None))["kimi"]
    assert r.status == "missing"
    assert "/mcp-config" in r.message
    server = json.loads(r.detail["fix"])["mcpServers"]["second-brain"]
    assert server == {
        "command": "second-brain-mcp",
        "args": ["--config", str(cfg.resolve())],
    }


@pytest.mark.parametrize("body", ["[]", '{"mcpServers": []}'])
def test_kimi_malformed_shape_is_error_and_unchanged(paths, cfg, body):
    target = paths.kimi_home / "mcp.json"
    target.write_text(body)
    r = by_name(check_mcp(paths, cfg, runner=fake(1), which=lambda _: None))["kimi"]
    assert r.status == "error"
    assert target.read_text() == body


def test_malformed_config_reported_and_unchanged(paths, cfg):
    target = paths.codex_home / "config.toml"
    target.write_bytes(b"[[[")
    r = by_name(check_mcp(paths, cfg, runner=fake(1), which=lambda _: None))["codex"]
    assert r.status == "error"
    assert r.message
    assert target.read_bytes() == b"[[["


def test_undecodable_config_is_error(paths, cfg):
    (paths.kimi_home / "mcp.json").write_bytes(b"\xff\xfe\x00")
    r = by_name(check_mcp(paths, cfg, runner=fake(1), which=lambda _: None))["kimi"]
    assert r.status == "error"


def test_harness_home_absent_is_not_installed(tmp_path, cfg):
    paths = HarnessPaths(tmp_path, {})
    runner = fake(0)
    results = by_name(
        check_mcp(paths, cfg, runner=runner, which=lambda _: "/bin/claude")
    )
    assert {r.status for r in results.values()} == {"not_installed"}
    assert runner.calls == []


def test_claude_present_and_missing_from_exit_code(paths, cfg):
    ok = fake(0)
    r = by_name(check_mcp(paths, cfg, runner=ok, which=lambda _: "/bin/claude"))[
        "claude"
    ]
    assert r.status == "present"
    assert ok.calls == [["claude", "mcp", "get", "second-brain"]]
    r = by_name(check_mcp(paths, cfg, runner=fake(1), which=lambda _: "/bin/claude"))[
        "claude"
    ]
    assert r.status == "missing"
    assert r.detail["fix"].startswith("claude mcp add --scope user second-brain -- ")


def test_claude_not_on_path_is_unknown(paths, cfg):
    runner = fake(0)
    r = by_name(check_mcp(paths, cfg, runner=runner, which=lambda _: None))["claude"]
    assert r.status == "unknown"
    assert runner.calls == []


@pytest.mark.parametrize(
    "exc", [subprocess.TimeoutExpired(["claude"], 20), OSError("exec format error")]
)
def test_claude_runner_failure_is_unknown(paths, cfg, exc):
    def runner(argv):
        raise exc

    r = by_name(check_mcp(paths, cfg, runner=runner, which=lambda _: "/bin/claude"))[
        "claude"
    ]
    assert r.status == "unknown"
    assert r.message


def test_missing_config_file_warns_without_snippet(paths, tmp_path):
    absent = tmp_path / "nope.toml"
    results = by_name(
        check_mcp(paths, absent, runner=fake(1), which=lambda _: "/bin/claude")
    )
    for name in ("claude", "codex", "kimi"):
        assert results[name].status == "missing"
        assert "fix" not in results[name].detail
        assert str(absent) in results[name].message


def test_missing_config_file_does_not_affect_present(paths, tmp_path):
    (paths.codex_home / "config.toml").write_text("[mcp_servers.second-brain]\n")
    r = by_name(
        check_mcp(paths, tmp_path / "nope.toml", runner=fake(0), which=lambda _: "/b")
    )
    assert r["codex"].status == "present"
    assert r["claude"].status == "present"


def test_snippets_survive_spaces_in_config_path(paths, tmp_path):
    cfg = tmp_path / "Application Support" / "rag config.toml"
    cfg.parent.mkdir()
    cfg.write_text("")
    results = by_name(
        check_mcp(paths, cfg, runner=fake(1), which=lambda _: "/bin/claude")
    )
    expected = str(cfg.resolve())
    codex = tomllib.loads(results["codex"].detail["fix"])
    assert codex["mcp_servers"]["second-brain"]["args"] == ["--config", expected]
    kimi = json.loads(results["kimi"].detail["fix"])
    assert kimi["mcpServers"]["second-brain"]["args"] == ["--config", expected]
    assert shlex.split(results["claude"].detail["fix"])[-1] == expected
