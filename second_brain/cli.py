"""Command-line interface for operating the local RAG system."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from second_brain.config import DEFAULT_EXCLUDE_GLOBS, load_config
from second_brain.evaluation import (
    evaluate,
    load_golden,
    report_to_dict,
    validate_depth,
)
from second_brain.retrieval import validate_code_source_weight, validate_graph_boost
from second_brain.service import RagService
from second_brain.skill_install import TARGETS, HarnessPaths, run_skills

_DEFAULT_EXCLUDE_GLOBS_TOML = ", ".join(
    f'"{pattern}"' for pattern in DEFAULT_EXCLUDE_GLOBS
)

DEFAULT_INIT_CONFIG = f"""vault_path = \"$CWD\"
qdrant_path = \"$CWD/data/qdrant\"
fts_path = \"$CWD/data/fts.sqlite\"
sync_state_path = \"$CWD/data/sync_state.sqlite\"
collection_name = \"second_brain_chunks\"
ollama_url = \"http://127.0.0.1:11434\"
embedding_model = \"nomic-embed-text\"
chunk_size = 500
chunk_overlap = 80
watch_enabled = true
watch_debounce_seconds = 5
exclude_globs = [{_DEFAULT_EXCLUDE_GLOBS_TOML}]
exclude_status = []
max_context_chunks = 8
redact_patterns = []
graph_enabled = true
graph_knn_k = 8
graph_semantic_min = 0.35
graph_min_edge_score = 0.15
graph_weight_semantic = 0.5
graph_weight_link = 0.25
graph_weight_tag = 0.15
graph_weight_comention = 0.10
graph_comention_cap = 3
graph_comention_max_fanout = 20
skip_headings = []
embed_context_header = true
embed_batch_size = 32
"""


def _write_init_config(config_path: Path, force: bool = False) -> dict:
    """Create a CWD-scoped default config file for first-time setup."""

    existed_before = config_path.exists()
    if existed_before and not force:
        raise FileExistsError(
            f"Config file already exists at {config_path}. Use --force to overwrite."
        )

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(DEFAULT_INIT_CONFIG, encoding="utf-8")

    data_dir = Path.cwd() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    return {
        "initialized": True,
        "config_path": str(config_path),
        "data_dir": str(data_dir),
        "overwritten": bool(force and existed_before),
    }


def main() -> None:
    """CLI entrypoint.

    Exposes sync/search/query/status/health commands and prints JSON output.
    """

    parser = argparse.ArgumentParser(description="Second Brain CLI")
    parser.add_argument("--config", default="rag_config.toml")

    sub = parser.add_subparsers(dest="command", required=True)

    init_p = sub.add_parser("init", help="initialize CWD with a default config")
    init_p.add_argument("--force", action="store_true")

    sync_p = sub.add_parser("sync", help="sync vault content")
    sync_p.add_argument(
        "--mode", choices=["full", "incremental", "file"], default="incremental"
    )
    sync_p.add_argument("--file-path")

    search_p = sub.add_parser("search", help="hybrid search")
    search_p.add_argument("query")
    search_p.add_argument("--top-k", type=int, default=10)
    search_p.add_argument("--graph-boost", type=float, default=0.0)
    search_p.add_argument("--code-source-weight", type=float)

    query_p = sub.add_parser("query", help="query with citations")
    query_p.add_argument("query")
    query_p.add_argument("--top-k", type=int, default=8)
    query_p.add_argument("--graph-boost", type=float, default=0.0)
    query_p.add_argument("--code-source-weight", type=float)

    eval_p = sub.add_parser("eval", help="score retrieval against a golden query set")
    eval_p.add_argument("--golden", required=True)
    eval_p.add_argument("--depth", type=int, default=10)
    eval_p.add_argument("--graph-boost", type=float, default=0.0)
    # Unset uses the config's code_source_weight; set it to tune that value.
    eval_p.add_argument("--code-source-weight", type=float)

    skills_p = sub.add_parser("skills", help="install the agent skill into harnesses")
    skills_sub = skills_p.add_subparsers(dest="skills_action", required=True)
    for name, helptext in (
        ("install", "install or update the skill"),
        ("uninstall", "remove the skill this tool installed"),
    ):
        action_p = skills_sub.add_parser(name, help=helptext)
        action_p.add_argument("--target", choices=[*TARGETS, "all"], default="all")
        action_p.add_argument("--dry-run", action="store_true")
        if name == "install":
            action_p.add_argument("--force", action="store_true")
    skills_sub.add_parser("status", help="report installed skill state and MCP setup")

    sub.add_parser("status", help="status")
    sub.add_parser("health", help="health")

    args = parser.parse_args()

    if args.command == "init":
        result = _write_init_config(Path(args.config), force=args.force)
        print(json.dumps(result, indent=2))
        return

    if args.command == "skills":
        # Before load_config: installing a skill must work with no vault or index yet.
        target = getattr(args, "target", "all")
        results, code = run_skills(
            args.skills_action,
            targets=TARGETS if target == "all" else (target,),
            force=getattr(args, "force", False),
            dry_run=getattr(args, "dry_run", False),
            config_path=Path(args.config),
            paths=HarnessPaths.from_env(),
            now=datetime.now(UTC),
        )
        print(
            json.dumps(
                {
                    "action": args.skills_action,
                    "results": [r.to_dict() for r in results],
                },
                indent=2,
            )
        )
        raise SystemExit(code)

    # Validate eval inputs first so a typo'd golden file or --depth fails
    # immediately instead of after the stores and embedder have been opened.
    eval_cases = None
    if args.command == "eval":
        try:
            validate_depth(args.depth)
            validate_graph_boost(args.graph_boost)
            if args.code_source_weight is not None:
                validate_code_source_weight(args.code_source_weight)
            eval_cases = load_golden(args.golden)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))

    config = load_config(args.config)
    service = RagService(config)

    if args.command == "sync":
        result = service.sync(mode=args.mode, file_path=args.file_path)
    elif args.command == "search":
        result = service.search(
            query=args.query,
            top_k=args.top_k,
            graph_boost=args.graph_boost,
            code_source_weight=args.code_source_weight,
        )
    elif args.command == "query":
        result = service.query(
            query=args.query,
            top_k=args.top_k,
            graph_boost=args.graph_boost,
            code_source_weight=args.code_source_weight,
        )
    elif args.command == "eval":
        report = evaluate(
            service,
            eval_cases,
            depth=args.depth,
            graph_boost=args.graph_boost,
            code_source_weight=args.code_source_weight,
        )
        result = report_to_dict(report)
    elif args.command == "status":
        result = service.status()
    elif args.command == "health":
        result = service.health()
    else:
        raise ValueError(f"Unknown command: {args.command}")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
