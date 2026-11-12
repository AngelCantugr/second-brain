"""Read-only check: does each harness have the `second-brain` MCP server registered?

This module never writes or repairs a harness config. A user's `config.toml` /
`mcp.json` is theirs, and a half-valid file is better reported than rewritten, so
the fix is always a snippet for the user to apply.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import tomllib
from collections.abc import Callable
from pathlib import Path

from second_brain.skill_install import HarnessPaths, Result

Runner = Callable[[list[str]], subprocess.CompletedProcess[str]]
SERVER = "second-brain"
COMMAND = "second-brain-mcp"
_CLAUDE_TIMEOUT_SECONDS = 20


def _default_runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
    # Argument list, no shell: nothing here is interpolated into a command line.
    return subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        timeout=_CLAUDE_TIMEOUT_SECONDS,
    )


def check_mcp(
    paths: HarnessPaths,
    config_path: Path,
    *,
    runner: Runner | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> list[Result]:
    cfg = str(config_path.resolve())
    config_exists = config_path.exists()
    run = runner or _default_runner
    return [
        _check_claude(paths, cfg, config_exists, run, which),
        _check_codex(paths, cfg, config_exists),
        _check_kimi(paths, cfg, config_exists),
    ]


def _missing(
    name: str,
    cfg: str,
    config_exists: bool,
    message: str,
    snippet: str,
    where: Path | None,
) -> Result:
    detail: dict[str, str] = {}
    if where is not None:
        detail["config"] = str(where)
    if not config_exists:
        # A snippet pointing at a nonexistent file would register a server that fails to start.
        return Result(
            "mcp",
            name,
            "missing",
            f"{message}; the config file {cfg} does not exist, so create it first",
            detail,
        )
    detail["fix"] = snippet
    return Result("mcp", name, "missing", message, detail)


def _not_installed(name: str) -> Result:
    return Result("mcp", name, "not_installed", "harness not found")


def _check_claude(
    paths: HarnessPaths,
    cfg: str,
    config_exists: bool,
    run: Runner,
    which: Callable[[str], str | None],
) -> Result:
    if not paths.claude_home.exists():
        return _not_installed("claude")
    if which("claude") is None:
        return Result(
            "mcp", "claude", "unknown", "claude CLI not on PATH; cannot check"
        )
    try:
        completed = run(["claude", "mcp", "get", SERVER])
    except (subprocess.TimeoutExpired, OSError) as exc:
        return Result("mcp", "claude", "unknown", f"claude mcp get failed: {exc}")
    if completed.returncode == 0:
        return Result("mcp", "claude", "present", "registered")
    snippet = (
        f"claude mcp add --scope user {SERVER} -- {COMMAND} --config {shlex.quote(cfg)}"
    )
    return _missing(
        "claude", cfg, config_exists, "not registered; run the command", snippet, None
    )


def _check_codex(paths: HarnessPaths, cfg: str, config_exists: bool) -> Result:
    if not paths.codex_home.exists():
        return _not_installed("codex")
    target = paths.codex_home / "config.toml"
    try:
        data = (
            tomllib.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
        )
    except (tomllib.TOMLDecodeError, UnicodeDecodeError, OSError) as exc:
        return Result(
            "mcp",
            "codex",
            "error",
            f"cannot read {target}: {exc}",
            {"config": str(target)},
        )
    if SERVER in _as_dict(data.get("mcp_servers")):
        return Result("mcp", "codex", "present", "registered", {"config": str(target)})
    # A JSON string is a valid TOML basic string, so json.dumps quotes paths safely.
    snippet = (
        f"[mcp_servers.{SERVER}]\n"
        f'command = "{COMMAND}"\n'
        f'args = ["--config", {json.dumps(cfg)}]\n'
    )
    return _missing(
        "codex", cfg, config_exists, f"not registered; add to {target}", snippet, target
    )


def _check_kimi(paths: HarnessPaths, cfg: str, config_exists: bool) -> Result:
    if not paths.kimi_home.exists():
        return _not_installed("kimi")
    target = paths.kimi_home / "mcp.json"
    detail = {"config": str(target)}
    try:
        data = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        return Result("mcp", "kimi", "error", f"cannot read {target}: {exc}", detail)
    if not isinstance(data, dict):
        return Result(
            "mcp", "kimi", "error", f"{target}: top level is not a JSON object", detail
        )
    servers = data.get("mcpServers", {})
    if not isinstance(servers, dict):
        return Result(
            "mcp", "kimi", "error", f"{target}: mcpServers is not an object", detail
        )
    if SERVER in servers:
        return Result("mcp", "kimi", "present", "registered", detail)
    snippet = json.dumps(
        {"mcpServers": {SERVER: {"command": COMMAND, "args": ["--config", cfg]}}},
        indent=2,
    )
    return _missing(
        "kimi",
        cfg,
        config_exists,
        f"not registered; merge into {target} or run /mcp-config in Kimi",
        snippet,
        target,
    )


def _as_dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}
