from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol

import ollama

from vault_mcp.chunker import chunk_markdown
from vault_mcp.config import Settings
from vault_mcp.index_store import ChromaIndexStore
from vault_mcp.metadata import Note, iter_markdown_files, parse_note


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...


class Store(Protocol):
    def replace_file_chunks(self, file_path: str, chunks, embeddings, note: Note) -> None:
        ...


@dataclass(slots=True)
class OllamaEmbedder:
    model: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = ollama.embed(model=self.model, input=texts)
        return response["embeddings"]


CHECKPOINT_FILE = "checkpoint.json"


def _checkpoint_path(index_dir: Path) -> Path:
    return index_dir / CHECKPOINT_FILE


def _load_checkpoint(index_dir: Path) -> dict[str, float]:
    path = _checkpoint_path(index_dir)
    if not path.exists():
        return {}
    return {k: float(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


def _save_checkpoint(index_dir: Path, checkpoint: dict[str, float]) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(index_dir)
    path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True), encoding="utf-8")


def run_ingestion(
    settings: Settings,
    embedder: Embedder | None = None,
    store: Store | None = None,
) -> dict[str, int]:
    vault_root = settings.resolved_vault_root()
    index_dir = settings.resolved_index_dir()

    checkpoint = _load_checkpoint(index_dir)
    next_checkpoint = checkpoint.copy()

    if embedder is None:
        embedder = OllamaEmbedder(model=settings.embedding_model)
    if store is None:
        store = ChromaIndexStore(str(index_dir), settings.collection_name)

    indexed_files = 0
    indexed_chunks = 0
    scanned_files = 0

    for path in iter_markdown_files(vault_root):
        scanned_files += 1
        note = parse_note(path, vault_root)
        last_indexed_mtime = checkpoint.get(note.relative_path, 0.0)
        if note.date_modified <= last_indexed_mtime:
            continue

        chunks = chunk_markdown(note.content, max_tokens=500)
        embeddings = embedder.embed_documents([chunk.text for chunk in chunks])
        store.replace_file_chunks(note.relative_path, chunks, embeddings, note)

        indexed_files += 1
        indexed_chunks += len(chunks)
        next_checkpoint[note.relative_path] = note.date_modified

    _save_checkpoint(index_dir, next_checkpoint)
    return {
        "scanned_files": scanned_files,
        "indexed_files": indexed_files,
        "indexed_chunks": indexed_chunks,
    }


def main() -> None:
    result = run_ingestion(Settings())
    print(
        f"scanned={result['scanned_files']} indexed_files={result['indexed_files']} indexed_chunks={result['indexed_chunks']}"
    )


if __name__ == "__main__":
    main()
