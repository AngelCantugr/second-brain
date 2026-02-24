from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from vault_mcp.chunker import Chunk
from vault_mcp.metadata import Note


@dataclass(slots=True)
class SearchResult:
    note: str
    excerpt: str
    score: float
    path: str


class ChromaIndexStore:
    def __init__(self, index_dir: str, collection_name: str) -> None:
        import chromadb

        client = chromadb.PersistentClient(path=index_dir)
        self.collection = client.get_or_create_collection(name=collection_name)

    def replace_file_chunks(
        self,
        file_path: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        note: Note,
    ) -> None:
        self.collection.delete(where={"file_path": file_path})
        if not chunks:
            return

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for chunk, embedding in zip(chunks, embeddings):
            payload = f"{file_path}:{chunk.chunk_index}:{chunk.heading}:{chunk.text}"
            chunk_id = hashlib.sha1(payload.encode("utf-8")).hexdigest()
            ids.append(chunk_id)
            documents.append(chunk.text)
            metadatas.append(
                {
                    "file_path": file_path,
                    "heading": chunk.heading,
                    "tags": ",".join(note.tags),
                    "date_modified": float(note.date_modified),
                    "chunk_index": int(chunk.chunk_index),
                    "title": note.path.stem,
                }
            )
            _ = embedding

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        raw = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)

        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]

        results: list[SearchResult] = []
        for doc, meta, distance in zip(docs, metas, dists):
            path = str(meta.get("file_path", ""))
            results.append(
                SearchResult(
                    note=str(meta.get("title", "")),
                    excerpt=str(doc),
                    score=1.0 - float(distance),
                    path=path,
                )
            )
        return results
