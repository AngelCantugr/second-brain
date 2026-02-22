# Building a semantic search engine for my second brain with Codex CLI

## Why this project

I wanted a portfolio-grade project that I would use daily, not a demo. My Week 7-8 ultralearning roadmap included "chat with my vault," so this became the natural build target.

## Stack and design decisions

- Python for fast iteration and alignment with existing repo scripts.
- Ollama + `nomic-embed-text` for local embeddings and zero API cost.
- ChromaDB for a simple local persistent vector store.
- MCP stdio server so Claude Code can call tools directly.

## Build process

I split implementation into ingestion and MCP serving:

1. Parse markdown and frontmatter.
2. Chunk by heading context at ~500-token windows.
3. Embed and upsert into ChromaDB.
4. Expose search and retrieval tools via MCP.

I used strict test-first loops to keep behavior stable while adding features.

## Challenges and fixes

- Python 3.14 + `chromadb` had compatibility issues; pinning to Python 3.13 fixed runtime stability.
- Incremental indexing required careful checkpoint handling to avoid needless re-embedding.
- Tool contracts needed explicit tests so output stayed stable for Claude calls.

## Results

I now have a practical local search layer over my notes, with fast incremental updates and a clear interface for Claude Code.

## Next steps

- Add a relevance tuning pass for chunk size and top-k defaults.
- Add a third blog post focused on real daily prompts and workflows.
