"""Command-line interface for operating the local RAG system."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from obsidian_rag.config import load_config
from obsidian_rag.service import RagService


DEFAULT_INIT_CONFIG = """vault_path = \"$CWD\"
qdrant_path = \"$CWD/data/qdrant\"
fts_path = \"$CWD/data/fts.sqlite\"
sync_state_path = \"$CWD/data/sync_state.sqlite\"
collection_name = \"obsidian_chunks\"
ollama_url = \"http://127.0.0.1:11434\"
embedding_model = \"nomic-embed-text\"
chunk_size = 500
chunk_overlap = 80
watch_enabled = true
exclude_globs = [\".obsidian/**\", \".git/**\", \"Templates/**\"]
max_context_chunks = 8
redact_patterns = []
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

    parser = argparse.ArgumentParser(description="Obsidian RAG CLI")
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

    query_p = sub.add_parser("query", help="query with answer draft")
    query_p.add_argument("query")
    query_p.add_argument("--top-k", type=int, default=8)

    sub.add_parser("status", help="status")
    sub.add_parser("health", help="health")

    args = parser.parse_args()

    if args.command == "init":
        result = _write_init_config(Path(args.config), force=args.force)
        print(json.dumps(result, indent=2))
        return

    config = load_config(args.config)
    service = RagService(config)

    if args.command == "sync":
        result = service.sync(mode=args.mode, file_path=args.file_path)
    elif args.command == "search":
        result = service.search(query=args.query, top_k=args.top_k)
    elif args.command == "query":
        result = service.query(query=args.query, top_k=args.top_k)
    elif args.command == "status":
        result = service.status()
    elif args.command == "health":
        result = service.health()
    else:
        raise ValueError(f"Unknown command: {args.command}")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
