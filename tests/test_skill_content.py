"""Drift guard: the shipped consumer skill must stay in step with the MCP server's tools."""

import ast
import importlib.resources
import re
from pathlib import Path

import second_brain
from second_brain.parser import parse_frontmatter

SKILL = importlib.resources.files("second_brain") / "skills" / "second-brain"
MCP_SERVER = Path(second_brain.__file__).parent / "mcp_server.py"


def _tool_decorated_functions() -> dict[str, ast.FunctionDef]:
    """Map each `@mcp.tool(name=...)` tool name to its function definition."""
    tree = ast.parse(MCP_SERVER.read_text("utf-8"))
    found: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if not (isinstance(decorator, ast.Call) and ast.unparse(decorator.func) == "mcp.tool"):
                continue
            for keyword in decorator.keywords:
                if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
                    found[keyword.value.value] = node
    return found


def registered_tools() -> set[str]:
    return set(_tool_decorated_functions())


def tool_docstring(name: str) -> str:
    return ast.get_docstring(_tool_decorated_functions()[name]) or ""


def skill_texts() -> dict[str, str]:
    texts: dict[str, str] = {}

    def walk(directory, prefix: str = "") -> None:
        for entry in directory.iterdir():
            if entry.is_dir():
                walk(entry, f"{prefix}{entry.name}/")
            elif entry.name.endswith(".md"):
                texts[f"{prefix}{entry.name}"] = entry.read_text("utf-8")

    walk(SKILL)
    return texts


def filter_table_keys(md: str) -> set[str]:
    """Backticked first-column keys of table rows under `## Filters`, up to the next heading."""
    keys: set[str] = set()
    in_section = False
    for line in md.splitlines():
        if re.match(r"^#{1,6}\s", line):
            in_section = line.strip() == "## Filters"
            continue
        if in_section and (match := re.match(r"^\|\s*`([^`]+)`\s*\|", line)):
            keys.add(match.group(1))
    return keys


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
    keys = filter_table_keys(skill_texts()["references/retrieval.md"])
    assert {"tags", "path_prefix", "doc_role", "kind", "repo"} <= keys
    doc = tool_docstring("rag.query")
    assert all(f"`{k}`" in doc for k in keys), [k for k in keys if f"`{k}`" not in doc]


def test_no_capture_instructions():
    assert all("rag.capture" not in t for t in skill_texts().values())


def test_every_registered_tool_is_mentioned_in_the_skill():
    # Reverse of the guard above: a tool added to the server must reach the skill.
    mentioned = {m for t in skill_texts().values() for m in re.findall(r"\brag\.[a-z_]+", t)}
    assert registered_tools() <= mentioned, registered_tools() - mentioned
