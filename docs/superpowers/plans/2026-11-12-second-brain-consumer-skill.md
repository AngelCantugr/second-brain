# Second-brain Consumer Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship one portable `second-brain` skill that tells Claude Code, Codex and Kimi agents when and how to use the second-brain MCP tools, plus a `second-brain skills install|status|uninstall` command that installs it, retires `second-brain-research`, and checks each harness's MCP wiring without editing it.

**Architecture:** The skill is package data under `second_brain/skills/second-brain/`, read through `importlib.resources`. `second_brain/skill_install.py` copies it to two target directories, tracks each copy with a content-hash marker, backs up anything it replaces outside the skill directories, and retires the legacy skill. `second_brain/harness_mcp.py` reads each harness's MCP config and builds fix snippets. Both take a `HarnessPaths` object instead of reading `~` directly, so tests run against `tmp_path`.

**Tech Stack:** Python ≥ 3.11 stdlib (`importlib.resources`, `hashlib`, `shutil`, `tomllib`, `json`, `subprocess`, `shlex`), existing `second_brain.parser.parse_frontmatter`, argparse CLI, pytest.

**Spec:** `docs/superpowers/specs/2026-11-12-second-brain-consumer-skill-design.md`

## Global Constraints

- Skill name `second-brain`; frontmatter `description` ≤ 1024 characters; `SKILL.md` ≤ 120 lines.
- Install targets: `claude` → `<home>/.claude/skills/second-brain/`; `agents` → `<home>/.agents/skills/second-brain/`. `--target` default `all`.
- Marker file `.install.json` (keys `version`, `hash`), excluded from the hash. Hash is SHA-256.
- Backups: `<home>/.local/state/second-brain/skill-backups/<UTC timestamp>/<target>/<skill name>/`. Never inside a `skills/` directory.
- Env: `CODEX_HOME` (default `<home>/.codex`), `KIMI_CODE_HOME` (default `<home>/.kimi-code`); an empty value counts as unset.
- Never edit harness MCP configs, `CLAUDE.md` or `AGENTS.md`. Never rewrite a malformed config.
- Only external binary: `claude mcp get second-brain`, argument list, no shell.
- Fix snippets use `second-brain-mcp --config <absolute --config path>`; when that file does not exist, warn instead of printing a snippet.
- Exit codes: `install`/`uninstall` → 1 if any result is `error` or `refused`, else 0 (a missing MCP server does not change it); `status` → always 0.
- The `skills` command is dispatched before `load_config`, like `init`.
- The skill contains no capture instructions.

## Review Focus

1. **A target directory that cannot be written** (this repo's own Claude sandbox denies writes to `~/.claude/skills`): that target reports `error` with the OS message and the other target still installs. Test in Task 2.
2. **`second-brain-research` installed as a symlink** (dotfile managers, and `~/.agents/skills/superpowers` is already a symlink): retiring moves the link itself and leaves its target untouched; unrelated symlinked skills are never touched. Test in Task 3.
3. **An update whose packaged skill dropped a file**: the installed copy matches the new tree exactly; the old file does not linger. Test in Task 2.
4. **A corrupt or hand-written `.install.json`**: treated as no marker, so the target is `refused` rather than overwritten or crashing. Test in Task 2.
5. **A `--config` path containing spaces**: every printed snippet still parses (TOML string, JSON string, shell-quoted command). Test in Task 4.

---

## File map

| File | Responsibility |
|---|---|
| `second_brain/skills/second-brain/SKILL.md` | The skill body (create) |
| `second_brain/skills/second-brain/references/{retrieval,graph,maintenance}.md` | On-demand detail (create) |
| `pyproject.toml` | `[tool.setuptools.package-data]` (modify) |
| `second_brain/skill_install.py` | Paths, results, hashing, install/uninstall/status, backup, legacy retire, stale-ref scan, `run_skills` (create) |
| `second_brain/harness_mcp.py` | Per-harness MCP check and fix snippets (create) |
| `second_brain/cli.py` | `skills` subcommand wiring (modify, around lines 84–118) |
| `tests/test_skill_content.py` | Drift guard on the skill text (create) |
| `tests/test_skill_install.py` | Installer behavior (create) |
| `tests/test_harness_mcp.py` | MCP checks (create) |
| `README.md`, `CHANGELOG.md` | Docs (modify) |

---

### Task 1: Skill content, packaging, drift guard

**Files:**
- Create: `second_brain/skills/second-brain/SKILL.md`, `second_brain/skills/second-brain/references/retrieval.md`, `.../graph.md`, `.../maintenance.md`
- Modify: `pyproject.toml` (after `[tool.setuptools.packages.find]`)
- Test: `tests/test_skill_content.py`

**Interfaces:**
- Consumes: nothing. The test reads the files with `importlib.resources.files("second_brain") / "skills" / "second-brain"`, which also proves the package can reach them.
- Produces: `references/retrieval.md` contains a `## Filters` heading followed by a markdown table whose first column is the filter key in backticks.

- [ ] **Step 1: Write the failing tests** in `tests/test_skill_content.py`

Helpers in the test module:
- `SKILL = importlib.resources.files("second_brain") / "skills" / "second-brain"`
- `registered_tools() -> set[str]`: `ast.parse` `second_brain/mcp_server.py`, collect the `name=` keyword string of every `mcp.tool(...)` decorator call.
- `tool_docstring(name: str) -> str`: the docstring of the function decorated with `mcp.tool(name=name)`.
- `skill_texts() -> dict[str, str]`: every `*.md` under `SKILL`, keyed by relative path.

```python
def test_registered_tools_parsed():
    assert len(registered_tools()) == 10 and "rag.query" in registered_tools()

def test_frontmatter_name_and_description():
    fm, _ = parse_frontmatter((SKILL / "SKILL.md").read_text("utf-8"))
    assert fm["name"] == "second-brain" == SKILL.name
    assert 0 < len(fm["description"]) <= 1024

def test_skill_md_line_budget():
    assert len((SKILL / "SKILL.md").read_text("utf-8").splitlines()) <= 120

def test_reference_links_resolve():
    links = re.findall(r"\]\((references/[^)#]+)\)", (SKILL / "SKILL.md").read_text("utf-8"))
    assert {"references/retrieval.md", "references/graph.md", "references/maintenance.md"} <= set(links)
    for link in links:
        assert (SKILL / link).is_file(), link

def test_every_rag_name_is_a_registered_tool():
    mentioned = {m for t in skill_texts().values() for m in re.findall(r"\brag\.[a-z_]+", t)}
    assert mentioned and mentioned <= registered_tools(), mentioned - registered_tools()

def test_filter_keys_are_documented_by_rag_query():
    keys = filter_table_keys(skill_texts()["references/retrieval.md"])  # rows under "## Filters"
    assert {"tags", "path_prefix", "doc_role", "kind", "repo"} <= keys
    doc = tool_docstring("rag.query")
    assert all(f"`{k}`" in doc for k in keys), [k for k in keys if f"`{k}`" not in doc]

def test_no_capture_instructions():
    assert all("rag.capture" not in t for t in skill_texts().values())
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_skill_content.py -v`
Expected: `test_registered_tools_parsed` passes; the others fail because the skill files do not exist.

- [ ] **Step 3: Write the skill files and package data**

- `pyproject.toml`: add
  ```toml
  [tool.setuptools.package-data]
  second_brain = ["skills/**/*"]
  ```
- `SKILL.md`: frontmatter verbatim from the spec's "Frontmatter" section; body sections 1–6 in the spec's order ("SKILL.md body, in order"), with relative links to the three reference files. The `sb-assistant` line uses exactly `/bin/zsh -lic 'sb-assistant ask --json "$1"' -- "$question"` and lists exit codes 0–5 as the spec gives them.
- `references/retrieval.md`: `rag.query` vs `rag.search`; `## Filters` table (key, shape, example) covering every key the `rag.query` docstring lists (`tags`, `path_prefix`, `date_range`, `modified_since`, `frontmatter_contains`, `kind`, `repo`, `doc_role`, `exclude_status`); `min_score`, `recency_boost`, `graph_boost`, `verbose`, `top_k`.
- `references/graph.md`: `rag.related` (`edge_types`, `signals`/`evidence`), `rag.connections`, `rag.note_context` (no note text), `rag.map` (clusters, orphans, bridges, ghosts; quadratic cost).
- `references/maintenance.md`: `rag.status` fields that signal staleness (`stale_files`, `untracked_files`, `missing_files`, `watcher_running`), `rag.sync` modes and when full is needed, `rag.health`, `rag.drift` (`roles`, `records`, judging a pair: note right / note outdated / code wrong).

Source every fact from the docstrings in `second_brain/mcp_server.py`; invent no parameter.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_skill_content.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml second_brain/skills tests/test_skill_content.py
git commit -m "feat(skill): add the second-brain consumer skill and a drift guard"
```

---

### Task 2: Installer core — paths, markers, install, uninstall

**Files:**
- Create: `second_brain/skill_install.py`
- Test: `tests/test_skill_install.py`

**Interfaces:**
- Produces:
  ```python
  TARGETS = ("claude", "agents")
  MARKER = ".install.json"

  @dataclass(frozen=True)
  class HarnessPaths:
      home: Path
      environ: Mapping[str, str]
      @classmethod
      def from_env(cls, home: Path | None = None, environ: Mapping[str, str] | None = None) -> "HarnessPaths"
      claude_home: Path            # home/.claude
      codex_home: Path             # $CODEX_HOME or home/.codex (empty = unset)
      kimi_home: Path              # $KIMI_CODE_HOME or home/.kimi-code (empty = unset)
      backup_root: Path            # home/.local/state/second-brain/skill-backups
      def skills_dir(self, target: str) -> Path      # claude_home/skills | home/.agents/skills
      def target_present(self, target: str) -> bool  # claude: claude_home exists; agents: codex_home or kimi_home exists

  @dataclass
  class Result:
      kind: str      # "target" | "legacy" | "reference" | "mcp"
      name: str      # target name, harness name, or file path
      status: str
      message: str = ""
      detail: dict[str, Any] = field(default_factory=dict)
      def to_dict(self) -> dict[str, Any]

  def packaged_skill() -> Traversable
  def tree_files(root: Traversable | Path) -> dict[str, bytes]   # relpath -> bytes, MARKER excluded
  def tree_hash(files: dict[str, bytes]) -> str                  # sha256 hex
  def backup(src: Path, paths: HarnessPaths, now: datetime, target: str) -> Path
  def install_skill(paths, targets: Sequence[str], *, force: bool, dry_run: bool, now: datetime) -> list[Result]
  def uninstall_skill(paths, targets: Sequence[str], *, dry_run: bool) -> list[Result]
  ```
- Target statuses: `ok` (fresh install), `updated`, `skipped` (already current), `not_installed`, `refused`, `error`. Dry-run returns the same statuses with `detail["dry_run"] = True` and writes nothing.

- [ ] **Step 1: Write the failing tests**

Fixture `paths(tmp_path)`: creates `home/.claude/` and `home/.codex/`, returns `HarnessPaths(home, {})`. Fixed `NOW = datetime(2026, 11, 12, 8, 0, tzinfo=UTC)`.

```python
def test_env_paths_default_and_empty_means_unset(tmp_path):
    p = HarnessPaths(tmp_path, {"CODEX_HOME": "", "KIMI_CODE_HOME": "/k"})
    assert p.codex_home == tmp_path / ".codex" and p.kimi_home == Path("/k")

def test_fresh_install_writes_both_targets_with_marker(paths):
    res = install_skill(paths, TARGETS, force=False, dry_run=False, now=NOW)
    assert [r.status for r in res] == ["ok", "ok"]
    for t in TARGETS:
        d = paths.skills_dir(t) / "second-brain"
        marker = json.loads((d / MARKER).read_text())
        assert marker["hash"] == tree_hash(tree_files(packaged_skill())) == tree_hash(tree_files(d))
        assert marker["version"] == importlib.metadata.version("second-brain-mcp")

def test_reinstall_current_copy_is_skipped(paths): ...        # second run -> ["skipped", "skipped"]

def test_update_replaces_tree_and_drops_removed_files(paths):  # Review Focus 3
    # install, then add <d>/references/old.md and rewrite marker hash to match the edited tree
    # (simulates an older packaged version that had old.md); reinstall
    # -> status "updated", old.md gone, tree equals packaged tree

def test_hand_edited_copy_is_refused_without_force(paths):     # edit SKILL.md after install -> "refused", file untouched

def test_force_backs_up_then_replaces(paths):
    # edited copy + force=True -> "updated"; backup at
    # backup_root/20261112T080000Z/claude/second-brain/SKILL.md holds the edited text;
    # nothing named second-brain* except the new copy under any skills/ dir

def test_corrupt_marker_is_refused(paths):                     # Review Focus 4: marker "not json" -> "refused"

def test_unmarked_existing_dir_is_refused(paths): ...

def test_dry_run_writes_nothing(paths):                        # snapshot of tmp tree before == after; statuses "ok", detail dry_run

def test_agents_target_not_installed_without_codex_or_kimi(tmp_path):
    # only .claude exists -> agents "not_installed", <home>/.agents not created

def test_unwritable_target_reports_error_and_other_installs(paths, monkeypatch):  # Review Focus 1
    # chmod claude skills dir 0o500 (skip on root) -> claude "error" with OS message, agents "ok"

def test_uninstall_removes_only_marked_dirs(paths):
    # install; create unmarked agents copy -> uninstall: claude "ok" (dir gone), agents "refused"

def test_backup_name_collision_gets_suffix(paths): ...         # two backups at same NOW -> distinct dirs
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_skill_install.py -v`
Expected: collection error, `ModuleNotFoundError: second_brain.skill_install`.

- [ ] **Step 3: Implement `second_brain/skill_install.py`**

- `packaged_skill()`: `importlib.resources.files("second_brain") / "skills" / "second-brain"`.
- `tree_hash`: SHA-256 over sorted entries, each `relpath.encode() + b"\0" + bytes + b"\0"`.
- Install writes into a temp dir `.second-brain-install-<pid>` in the skills dir's *parent* (`~/.claude/`, `~/.agents/`): same filesystem, so `os.replace` is atomic, but outside `skills/`, so an interrupted install never leaves a loadable folder. Then back up (if forced) or delete (if the marker matches) the old copy, and `os.replace` the temp dir into place. Remove the temp dir on failure.
- `backup`: destination `backup_root/<now:%Y%m%dT%H%M%SZ>/<target>/<src.name>`; if the timestamp dir already holds that name, append `-1`, `-2`, … to the timestamp segment. Use `shutil.move` (moves a symlink as a link).
- Catch `OSError` per target and return `Result("target", t, "error", str(exc))`.
- Version from `importlib.metadata.version("second-brain-mcp")`.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_skill_install.py -v`
Expected: all pass (the chmod test is skipped when running as root).

- [ ] **Step 5: Commit**

```bash
git add second_brain/skill_install.py tests/test_skill_install.py
git commit -m "feat(skills): install and uninstall the packaged skill with content-hash markers"
```

---

### Task 3: Retire `second-brain-research` and report stale references

**Files:**
- Modify: `second_brain/skill_install.py`
- Test: `tests/test_skill_install.py`

**Interfaces:**
- Consumes: `HarnessPaths`, `Result`, `backup` from Task 2; `second_brain.parser.parse_frontmatter`.
- Produces:
  ```python
  LEGACY_SKILL = "second-brain-research"
  def retire_legacy(paths: HarnessPaths, *, dry_run: bool, now: datetime) -> list[Result]   # kind "legacy"; status "retired" | "skipped" | "error"
  def scan_stale_references(paths: HarnessPaths) -> list[Result]                            # kind "reference"; status "found"; name "path:line"
  ```
  Scanned files: `claude_home/CLAUDE.md`, `codex_home/AGENTS.md`, `kimi_home/AGENTS.md`; missing files are skipped silently.

- [ ] **Step 1: Write the failing tests**

```python
def test_retire_moves_matching_legacy_from_both_dirs(paths):
    # write SKILL.md "---\nname: second-brain-research\n---\n" in both skills dirs
    res = retire_legacy(paths, dry_run=False, now=NOW)
    assert [r.status for r in res] == ["retired", "retired"]
    assert not any((paths.skills_dir(t) / LEGACY_SKILL).exists() for t in TARGETS)
    assert (paths.backup_root / "20261112T080000Z" / "agents" / LEGACY_SKILL / "SKILL.md").is_file()

def test_retire_skips_dir_with_other_frontmatter_name(paths): ...   # name: something-else -> "skipped", dir stays

def test_retire_moves_symlink_not_its_target(paths, tmp_path):      # Review Focus 2
    # real dir outside home with matching SKILL.md; symlink it as agents/second-brain-research
    # -> "retired"; backup entry is a symlink; real dir and its SKILL.md still exist

def test_retire_never_touches_other_symlinked_skills(paths): ...    # agents/superpowers symlink survives

def test_retire_dry_run_moves_nothing(paths): ...

def test_stale_reference_scan_reports_path_and_line(paths):
    (paths.claude_home / "CLAUDE.md").write_text("a\nuse the `second-brain-research` skill\n")
    [r] = scan_stale_references(paths)
    assert (r.kind, r.status, r.name) == ("reference", "found", f"{paths.claude_home / 'CLAUDE.md'}:2")

def test_stale_reference_scan_never_writes(paths): ...              # file bytes and mtime unchanged
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_skill_install.py -k "retire or stale" -v`
Expected: FAIL with `ImportError: cannot import name 'retire_legacy'`.

- [ ] **Step 3: Implement `retire_legacy` and `scan_stale_references`**

Read the frontmatter through the symlink (`path / "SKILL.md"`), but move the entry itself with `backup(...)`. Never follow or resolve a symlink when deciding what to move.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_skill_install.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add second_brain/skill_install.py tests/test_skill_install.py
git commit -m "feat(skills): retire second-brain-research and report stale references to it"
```

---

### Task 4: Per-harness MCP check

**Files:**
- Create: `second_brain/harness_mcp.py`
- Test: `tests/test_harness_mcp.py`

**Interfaces:**
- Consumes: `HarnessPaths`, `Result` from Task 2.
- Produces:
  ```python
  Runner = Callable[[list[str]], subprocess.CompletedProcess[str]]
  SERVER = "second-brain"
  def check_mcp(paths: HarnessPaths, config_path: Path, *, runner: Runner | None = None,
                which: Callable[[str], str | None] = shutil.which) -> list[Result]
  ```
  One `Result(kind="mcp")` per harness in the order `claude`, `codex`, `kimi`. Status `present` | `missing` | `unknown` | `not_installed` | `error`. On `missing`, `detail["fix"]` is the snippet, unless `config_path` does not exist: then `detail` has no `fix` and `message` warns that the config file is missing. The default runner calls `subprocess.run(argv, capture_output=True, text=True, timeout=20)`.
- Snippets (`<cfg>` = `config_path.resolve()`):
  - codex: `[mcp_servers.second-brain]\ncommand = "second-brain-mcp"\nargs = ["--config", <json.dumps(cfg)>]\n`
  - kimi: `json.dumps({"mcpServers": {"second-brain": {"command": "second-brain-mcp", "args": ["--config", cfg]}}}, indent=2)`, and `message` adds "or run /mcp-config in Kimi".
  - claude: `"claude mcp add --scope user second-brain -- second-brain-mcp --config " + shlex.quote(cfg)`

- [ ] **Step 1: Write the failing tests**

Helpers: fixture `paths` as in Task 2, plus `<home>/.kimi-code/`; fixture `cfg` = an existing `tmp_path / "rag_config.toml"`; `fake(code)` returns a runner that records argv and returns `CompletedProcess(argv, code, "", "")`; `by_name(results)` maps `name -> Result`.

```python
def test_codex_present(paths, cfg): ...          # config.toml with [mcp_servers.second-brain] -> "present"
def test_codex_missing_snippet_is_valid_toml(paths, cfg):
    fix = by_name(check_mcp(paths, cfg, runner=fake(1), which=lambda _: None))["codex"].detail["fix"]
    assert tomllib.loads(fix)["mcp_servers"]["second-brain"]["args"] == ["--config", str(cfg.resolve())]
def test_kimi_reads_kimi_code_home_env(paths, cfg, tmp_path): ...  # mcp.json under $KIMI_CODE_HOME -> "present"
def test_kimi_missing_snippet_is_valid_json(paths, cfg): ...
def test_malformed_config_reported_and_unchanged(paths, cfg):
    # codex config.toml "[[[" -> status "error", parser message in r.message, bytes unchanged
def test_harness_home_absent_is_not_installed(tmp_path, cfg): ...
def test_claude_present_and_missing_from_exit_code(paths, cfg):
    # runner returns code 0 -> "present"; code 1 -> "missing" with fix; argv == ["claude", "mcp", "get", "second-brain"]
def test_claude_not_on_path_is_unknown(paths, cfg): ...            # which -> None, runner never called
def test_claude_timeout_is_unknown(paths, cfg): ...                # runner raises subprocess.TimeoutExpired
def test_missing_config_file_warns_without_snippet(paths, tmp_path): ...
def test_snippets_survive_spaces_in_config_path(paths, tmp_path):  # Review Focus 5
    # cfg = tmp_path / "Application Support" / "rag config.toml"; all three missing
    # tomllib.loads(codex fix) and json.loads(kimi fix) round-trip str(cfg);
    # shlex.split(claude fix)[-1] == str(cfg)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_harness_mcp.py -v`
Expected: collection error, `ModuleNotFoundError: second_brain.harness_mcp`.

- [ ] **Step 3: Implement `check_mcp`** in `second_brain/harness_mcp.py`

The Claude harness counts as installed when `paths.claude_home` exists; Codex and Kimi when their home dirs exist. Catch `tomllib.TOMLDecodeError`, `json.JSONDecodeError` and `OSError` per harness.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_harness_mcp.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add second_brain/harness_mcp.py tests/test_harness_mcp.py
git commit -m "feat(skills): check each harness for the second-brain MCP server and print the fix"
```

---

### Task 5: `status`, CLI wiring, exit codes, docs

**Files:**
- Modify: `second_brain/skill_install.py`, `second_brain/cli.py:84-118`, `README.md`, `CHANGELOG.md`
- Test: `tests/test_skill_install.py`

**Interfaces:**
- Consumes: everything from Tasks 2–4.
- Produces:
  ```python
  def skill_status(paths: HarnessPaths) -> list[Result]
      # kind "target"; status "ok" | "absent" | "modified" | "outdated" | "not_installed";
      # detail: installed_version, modified (bool), update_available (bool)
  def exit_code(action: str, results: Sequence[Result]) -> int
      # status -> 0. install/uninstall -> 1 if any result of kind "target" or "legacy"
      # has status "error" or "refused"; "mcp" and "reference" results never change it.
  def run_skills(action: str, *, targets: Sequence[str], force: bool, dry_run: bool, config_path: Path,
                 paths: HarnessPaths, now: datetime, runner=None, which=shutil.which) -> tuple[list[Result], int]
      # install   = install_skill + retire_legacy + scan_stale_references + check_mcp
      # uninstall = uninstall_skill
      # status    = skill_status + check_mcp
  ```
- CLI: `second-brain [--config PATH] skills {install,status,uninstall}` with `--target {claude,agents,all}` (default `all`) on install/uninstall, `--dry-run` on install/uninstall, `--force` on install. It prints `json.dumps({"action": ..., "results": [r.to_dict() ...]}, indent=2)` and exits with `exit_code`.

- [ ] **Step 1: Write the failing tests**

```python
def test_status_reports_ok_modified_outdated_absent(paths): ...
    # before install: both "absent"; after install: "ok"; edit a file: "modified";
    # rewrite marker+tree to a different valid hash: "outdated" with update_available True
def test_status_writes_nothing(paths): ...
def test_exit_codes():
    assert exit_code("install", [Result("target", "claude", "ok"), Result("mcp", "kimi", "missing")]) == 0
    assert exit_code("install", [Result("target", "agents", "refused")]) == 1
    assert exit_code("uninstall", [Result("target", "claude", "error")]) == 1
    assert exit_code("status", [Result("target", "claude", "error")]) == 0
    assert exit_code("install", [Result("mcp", "codex", "error"), Result("reference", "x:1", "found")]) == 0
    assert exit_code("install", [Result("legacy", "agents", "error")]) == 1
def test_cli_install_end_to_end(tmp_path, monkeypatch, capsys):
    # HOME=tmp_path with .claude and .codex; PATH="" (claude unknown); legacy skill present;
    # cfg file exists; sys.argv = ["second-brain", "--config", str(cfg), "skills", "install"]
    # -> SystemExit code 0; JSON results include targets "ok", legacy "retired", mcp codex "missing" with fix
def test_cli_skills_does_not_load_config(tmp_path, monkeypatch, capsys):
    # --config points at a nonexistent file; "skills status" still exits 0 (load_config never called)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_skill_install.py -k "status or exit or cli" -v`
Expected: FAIL with `ImportError` for `skill_status` / `exit_code`.

- [ ] **Step 3: Implement `skill_status`, `exit_code`, `run_skills`, and the CLI branch**

In `cli.main`, dispatch `skills` immediately after the `init` branch and before eval validation and `load_config`: call `run_skills(..., paths=HarnessPaths.from_env(), now=datetime.now(UTC))`, print, `raise SystemExit(code)`.

- [ ] **Step 4: Write the docs**

- README: a "Use it from your AI agents" section after the MCP section. It covers: what the skill does (one paragraph); `second-brain --config <path> skills install` and `--dry-run`; upgrade = re-run install; `skills status`; what the MCP check prints; that `second-brain-research` is retired to the backup dir; and the five-prompt manual trigger checklist copied from the spec's "Manual trigger checklist", for `claude -p`, `codex exec`, `kimi -p`.
- CHANGELOG: one Unreleased entry at the top in the file's existing style: the skill, the three commands, legacy retirement, the read-only MCP check, and the drift-guard test.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all tests pass, including the pre-existing ones.

- [ ] **Step 6: Commit**

```bash
git add second_brain/skill_install.py second_brain/cli.py tests/test_skill_install.py README.md CHANGELOG.md
git commit -m "feat(skills): add the second-brain skills command and document agent setup"
```

---

### Task 6: Release check (no code)

- [ ] **Step 1: The wheel carries the skill**

Run: `uv build --wheel && unzip -l dist/*.whl | grep "second_brain/skills/second-brain/"`
Expected: `SKILL.md` and the three `references/*.md` are listed.

- [ ] **Step 2: Lint**

Run: `uvx ruff check second_brain tests`
Expected: no errors.

- [ ] **Step 3: Dry run against the real home directory (read-only)**

Run: `uv run second-brain --config ~/Documents/Claude/rag_config.toml skills install --dry-run`
Expected: both targets planned; `second-brain-research` planned for retirement from both dirs; stale references at `~/.claude/CLAUDE.md` and `~/.codex/AGENTS.md`; MCP `claude` present (or `unknown` inside a sandbox), `codex` present, `kimi` missing with a JSON fix. Report the output to the user. The real install and the manual trigger checklist are the user's call, not the executor's.
