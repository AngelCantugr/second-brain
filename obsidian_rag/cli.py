"""Command-line interface for operating the local RAG system."""

from __future__ import annotations

import argparse
import json

from obsidian_rag.config import load_config
from obsidian_rag.service import RagService


def main() -> None:
    """CLI entrypoint.

    Exposes sync/search/query/status/health commands and prints JSON output.
    """

    parser = argparse.ArgumentParser(description="Obsidian RAG CLI")
    parser.add_argument("--config", default="rag_config.toml")

    sub = parser.add_subparsers(dest="command", required=True)

    sync_p = sub.add_parser("sync", help="sync vault content")
    sync_p.add_argument("--mode", choices=["full", "incremental", "file"], default="incremental")
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
