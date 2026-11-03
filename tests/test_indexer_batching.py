"""Cross-note embedding batches: packing, unchanged vectors, failure isolation."""

from pathlib import Path

from second_brain.config import RagConfig
from second_brain.graph import GraphStore
from second_brain.indexer import Indexer
from second_brain.keyword_store import KeywordStore
from second_brain.sync_state import SyncStateStore
from second_brain.vector_store import InMemoryVectorStore


class _StubEmbedder:
    """Vectors depend only on the text, so batching cannot change them."""

    def __init__(self, poison: tuple[str, ...] = ()) -> None:
        self.calls: list[list[str]] = []
        self.poison = poison

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if any(marker in text for marker in self.poison for text in texts):
            raise RuntimeError("Ollama /api/embed batch 1/1 failed with HTTP 500")
        return [[float(len(text)), float(sum(map(ord, text)) % 997)] for text in texts]


def _build_indexer(
    tmp_path: Path, vault: Path, embedder: _StubEmbedder, batch_size: int
) -> Indexer:
    store_dir = tmp_path / f"stores-{batch_size}"
    store_dir.mkdir(exist_ok=True)
    config = RagConfig(
        vault_path=vault,
        qdrant_path=store_dir / "qdrant",
        fts_path=store_dir / "fts.sqlite",
        sync_state_path=store_dir / "sync_state.sqlite",
        chunk_size=20,
        chunk_overlap=0,
        embed_batch_size=batch_size,
    )
    indexer = Indexer(
        config=config,
        embedder=embedder,
        vector_store=InMemoryVectorStore(),
        keyword_store=KeywordStore(config.fts_path),
        sync_state=SyncStateStore(config.sync_state_path),
        graph_store=GraphStore(config.fts_path),
    )
    indexer.initialize()
    return indexer


def _write_notes(vault: Path, notes: dict[str, str]) -> None:
    vault.mkdir(exist_ok=True)
    for name, body in notes.items():
        (vault / name).write_text(body, encoding="utf-8")


def _vectors_by_chunk(indexer: Indexer) -> dict[str, list[float]]:
    return {
        chunk_id: vector for chunk_id, (vector, _chunk) in indexer.vector_store._vectors.items()
    }


def _texts_by_path(indexer: Indexer, rel_path: str) -> list[str]:
    return [hit.text for hit in indexer.keyword_store.chunks_by_path(rel_path)]


def _long_body(words: int) -> str:
    return " ".join(f"w{i}" for i in range(words))


def test_full_sync_packs_chunks_from_several_notes_into_one_embed_call(
    tmp_path: Path,
) -> None:
    vault = tmp_path / "vault"
    _write_notes(vault, {f"n{i}.md": f"# N{i}\nbody of note {i}" for i in range(5)})
    embedder = _StubEmbedder()
    indexer = _build_indexer(tmp_path, vault, embedder, batch_size=4)

    result = indexer.sync(mode="full")

    assert result.errors == []
    assert result.processed == 5
    assert [len(call) for call in embedder.calls] == [4, 1]


def test_cross_note_batches_store_the_same_vectors_as_per_note_calls(
    tmp_path: Path,
) -> None:
    vault = tmp_path / "vault"
    _write_notes(
        vault,
        {
            "a.md": "# A\n" + _long_body(50),
            "b.md": "# B\nshort",
            "c.md": "# C\n" + _long_body(30),
            "empty.md": "",
            "d.md": "# D\nanother short one",
        },
    )
    batched = _build_indexer(tmp_path, vault, _StubEmbedder(), batch_size=32)
    per_note = _build_indexer(tmp_path, vault, _StubEmbedder(), batch_size=1)

    batched.sync(mode="full")
    per_note.sync(mode="full")

    assert len(batched.embedder.calls) == 1
    assert _vectors_by_chunk(batched) == _vectors_by_chunk(per_note)
    for path in ("a.md", "b.md", "c.md", "d.md"):
        meta = batched.graph_store.note_meta_for(path)
        assert meta == per_note.graph_store.note_meta_for(path)


def test_note_larger_than_a_batch_is_embedded_alone(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    _write_notes(vault, {"a.md": "# A\nsmall", "big.md": "# Big\n" + _long_body(100)})
    embedder = _StubEmbedder()
    indexer = _build_indexer(tmp_path, vault, embedder, batch_size=2)

    indexer.sync(mode="full")

    big_inputs = [text for text in sum(embedder.calls, []) if text.startswith("big > Big")]
    assert len(big_inputs) == 5
    assert big_inputs in embedder.calls


def test_failed_batch_reports_only_the_note_ollama_rejects(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    _write_notes(vault, {"a.md": "# A\nold a", "b.md": "# B\nold b", "c.md": "# C\nold c"})
    embedder = _StubEmbedder()
    indexer = _build_indexer(tmp_path, vault, embedder, batch_size=32)
    indexer.sync(mode="full")

    _write_notes(vault, {"a.md": "# A\nnew a", "b.md": "# B\nnew POISON", "c.md": "# C\nnew c"})
    embedder.poison = ("POISON",)
    embedder.calls.clear()
    result = indexer.sync(mode="incremental")

    # The packed call fails, then each note is retried alone.
    assert [len(call) for call in embedder.calls] == [3, 1, 1, 1]
    assert result.processed == 2
    assert len(result.errors) == 1
    assert result.errors[0].startswith(str(vault / "b.md"))
    assert _texts_by_path(indexer, "a.md") == ["new a"]
    assert _texts_by_path(indexer, "c.md") == ["new c"]
    # Embed before write: the rejected note keeps its previous entries and
    # is neither marked pending nor recorded as current.
    assert _texts_by_path(indexer, "b.md") == ["old b"]
    assert indexer.sync_state.pending_paths() == []

    embedder.poison = ()
    retry = indexer.sync(mode="incremental")

    assert retry.errors == []
    assert retry.processed == 1
    assert _texts_by_path(indexer, "b.md") == ["new POISON"]


def test_embedding_outage_leaves_every_note_untouched_and_retried(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    _write_notes(vault, {"a.md": "# A\nold a", "b.md": "# B\nold b"})
    embedder = _StubEmbedder()
    indexer = _build_indexer(tmp_path, vault, embedder, batch_size=32)
    indexer.sync(mode="full")
    vectors_before = _vectors_by_chunk(indexer)

    _write_notes(vault, {"a.md": "# A\nnew a", "b.md": "# B\nnew b"})
    embedder.poison = ("",)  # every call fails, as when Ollama is down
    result = indexer.sync(mode="incremental")

    assert result.processed == 0
    assert sorted(error.split(":")[0] for error in result.errors) == [
        str(vault / "a.md"),
        str(vault / "b.md"),
    ]
    assert _vectors_by_chunk(indexer) == vectors_before
    assert indexer.sync_state.pending_paths() == []

    embedder.poison = ()
    retry = indexer.sync(mode="incremental")

    assert retry.errors == []
    assert retry.processed == 2
