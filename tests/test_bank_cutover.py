"""Rehearse a clean cutover against disposable stores and a real MCP client."""

import asyncio
import json
from pathlib import Path

import pytest
from mcp.client import Client

from second_brain import cli, mcp_server
from second_brain.banks import FileRef
from second_brain.config import load_config
from second_brain.evaluation import load_golden, evaluate, report_to_dict
from second_brain.service import RagService


class StubEmbedder:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def health(self) -> bool:
        return True


def _config(tmp_path: Path, name: str, roots: str) -> Path:
    path = tmp_path / f"{name}.toml"
    path.write_text(
        f'qdrant_path = "{name}/vectors"\nfts_path = "{name}/fts.sqlite"\n'
        f'sync_state_path = "{name}/state.sqlite"\n' + roots,
        encoding="utf-8",
    )
    return path


BANKS = (
    '[[banks]]\nid = "learning"\ntype = "knowledge-base"\npath = "learning"\n'
    '[[banks]]\nid = "work"\ntype = "knowledge-base"\npath = "work"\n'
    '[[banks]]\nid = "repo"\ntype = "repo"\npath = "repo"\n'
)


def test_bank_tools_round_trip_over_mcp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("second_brain.service.OllamaEmbedder", StubEmbedder)
    for bank in ("learning", "work", "repo"):
        root = tmp_path / bank
        root.mkdir()
        (root / "README.md").write_text(f"# README\nneedle from {bank}")
    (tmp_path / "learning" / "Index.md").write_text("[[README]]")
    server = mcp_server.build_server(str(_config(tmp_path, "new", BANKS)))

    async def exercise() -> None:
        async with Client(server) as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            assert {"rag.list_banks", "rag.fetch"} <= tools.keys()
            assert {"file"} <= tools["rag.fetch"].input_schema["properties"].keys()
            assert {"banks"} <= tools["rag.query"].input_schema["properties"].keys()

            async def call(name: str, arguments: dict) -> dict:
                result = await client.call_tool(name, arguments)
                assert not result.is_error
                return result.structured_content or json.loads(result.content[0].text)

            inventory = (await call("rag.list_banks", {}))["banks"]
            assert {entry["id"] for entry in inventory} == {"learning", "work", "repo"}
            sync = await call("rag.sync", {"mode": "full"})
            assert sync["errors"] == []
            assert sync["processed"] == 4
            results = await call("rag.query", {"query": "needle", "banks": ["work"]})
            file = {"bank": "work", "path": "README.md"}
            assert results["citations"][0]["file"] == file
            fetched = await call("rag.fetch", {"file": results["citations"][0]["file"]})
            assert fetched["chunks"][0]["text"].strip() == "needle from work"
            refs = (await call("rag.query", {"query": "needle"}))["citations"]
            assert {item["file"]["bank"] for item in refs} == {"learning", "work", "repo"}

            index = {"bank": "learning", "path": "Index.md"}
            graph = await call(
                "rag.related", {"file": index, "banks": ["repo"], "edge_types": ["links"]}
            )
            assert graph["neighbors"][0]["file"] == {"bank": "repo", "path": "README.md"}
            conn = await call(
                "rag.connections",
                {"file_a": index, "file_b": {"bank": "learning", "path": "README.md"}, "banks": ["repo"]},
            )
            assert conn["direct_edge"]["signals"]["link"] > 0
            invalid = await client.call_tool("rag.fetch", {"file": {"bank": "work", "path": "../outside.md"}})
            assert invalid.is_error

    asyncio.run(exercise())


def test_bank_rehearsal_leaves_legacy_store_and_config_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("second_brain.service.OllamaEmbedder", StubEmbedder)
    (tmp_path / "learning").mkdir()
    (tmp_path / "learning" / "README.md").write_text("needle in the original corpus")
    legacy = _config(tmp_path, "old", 'vault_path = "learning"\n')
    old = RagService(load_config(legacy))
    assert old.sync("full")["processed"] == 1
    old.vector_store.client.close()
    previous = {
        path: path.read_bytes()
        for path in (tmp_path / "old").rglob("*") if path.is_file()
    }
    previous[legacy] = legacy.read_bytes()
    with pytest.raises(ValueError, match=r"\[\[banks\]\].*fresh"):
        mcp_server.build_server(str(legacy))

    reused = tmp_path / "reuse.toml"
    reused.write_text(legacy.read_text().split('vault_path')[0] + BANKS)
    with pytest.raises(ValueError, match="fresh index"):
        mcp_server.build_server(str(reused))

    updated_path = _config(tmp_path, "new", BANKS)
    updated = RagService(load_config(updated_path))
    assert updated.sync("full")["processed"] == 1
    updated.vector_store.client.close()
    reopened = RagService(load_config(updated_path))
    ref = FileRef("learning", "README.md")
    assert reopened.fetch(ref)["found"]
    assert [c["file"] for c in reopened.query("needle")["citations"]] == [ref.to_dict()]
    reopened.vector_store.client.close()
    assert all(path.read_bytes() == data for path, data in previous.items())


def test_cli_init_and_file_sync_use_bank_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("second_brain.service.OllamaEmbedder", StubEmbedder)
    config = tmp_path / "rag_config.toml"
    cli._write_init_config(config)
    assert load_config(config).banks["vault"].path == tmp_path
    (tmp_path / "Only.md").write_text("needle")
    monkeypatch.setattr("sys.argv", [
        "second-brain", "sync", "--mode", "file", "--bank", "vault", "--file-path", "Only.md",
    ])
    cli.main()
    assert '"processed": 1' in capsys.readouterr().out


def test_golden_cases_keep_same_path_in_different_banks_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("second_brain.service.OllamaEmbedder", StubEmbedder)
    for bank in ("learning", "work"):
        root = tmp_path / bank
        root.mkdir()
        (root / "README.md").write_text(f"needle in {bank}")
    service = RagService(load_config(_config(tmp_path, "new", BANKS)), use_in_memory_vector=True)
    assert service.sync("full")["processed"] == 2
    golden = tmp_path / "golden.toml"
    golden.write_text(
        '[[cases]]\nquery = "needle"\nexpected = ['
        '{bank = "learning", path = "README.md"}, {bank = "work", path = "README.md"}]\n'
    )
    result = report_to_dict(evaluate(service, load_golden(golden)))
    assert result["summary"]["recall@5"] == 1.0
    assert result["summary"]["unknown_expected"] == 0
    assert len(result["cases"][0]["top_paths"]) == 2
    assert {file["bank"] for file in result["cases"][0]["expected"]} == {"learning", "work"}
