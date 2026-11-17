"""Bank identity checks against disposable indexes."""

from pathlib import Path

import pytest

from second_brain.banks import FileRef
from second_brain.config import load_config
from second_brain.service import RagService


class StubEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def health(self) -> bool:
        return True


@pytest.fixture
def service(tmp_path: Path) -> RagService:
    for bank in ("learning", "work", "repo"):
        (tmp_path / bank).mkdir()
    config_path = tmp_path / "rag.toml"
    config_path.write_text(
        'qdrant_path = "vectors"\nfts_path = "fts.sqlite"\nsync_state_path = "state.sqlite"\n'
        'redact_patterns = ["private-token"]\n'
        '[[banks]]\nid = "learning"\ntype = "obsidian"\npath = "learning"\n'
        '[[banks]]\nid = "work"\ntype = "obsidian"\npath = "work"\n'
        '[[banks]]\nid = "repo"\ntype = "repo"\npath = "repo"\n',
        encoding="utf-8",
    )
    svc = RagService(load_config(config_path))
    svc.embedder = svc.indexer.embedder = StubEmbedder()
    yield svc
    svc.vector_store.client.close()


def _write(service: RagService, bank: str, path: str, text: str) -> FileRef:
    target = service.config.banks[bank].path / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return FileRef(bank, path)


def test_duplicate_paths_keep_independent_chunks_and_checkpoints(service: RagService) -> None:
    first = _write(service, "learning", "README.md", "# Shared\nSame body")
    second = _write(service, "work", "README.md", "# Shared\nSame body")
    assert service.sync("full")["processed"] == 2
    assert set(service.sync_state.tracked_paths()) == {first.to_key(), second.to_key()}
    first_ids = set(service.keyword_store.chunk_ids_by_path(first.to_key()))
    second_ids = set(service.keyword_store.chunk_ids_by_path(second.to_key()))
    assert first_ids and second_ids and first_ids.isdisjoint(second_ids)
    assert set(service.vector_store.chunk_ids_by_path(first.to_key())) == first_ids
    assert set(service.vector_store.chunk_ids_grouped_by_path()) == {first.to_key(), second.to_key()}

    _write(service, "learning", "README.md", "# Changed\nUpdated body")
    assert service.sync("file", file=first)["processed"] == 1
    assert set(service.keyword_store.chunk_ids_by_path(second.to_key())) == second_ids
    assert service.graph_store.note_meta_for(second.to_key())["title"] == "README"
    (service.config.banks["learning"].path / first.path).unlink()
    assert service.sync("file", file=first)["deleted"] == 1
    assert service.vector_store.chunk_ids_by_path(first.to_key()) == []
    assert set(service.sync_state.tracked_paths()) == {second.to_key()}


@pytest.mark.parametrize("mode", ["incremental", "full"])
def test_unavailable_bank_retains_state(service: RagService, mode: str) -> None:
    old = _write(service, "repo", "README.md", "# Repo\nPreviously indexed")
    assert service.sync("full")["processed"] == 1
    root = service.config.banks["repo"].path
    root.rename(root.with_name("unmounted"))
    _write(service, "work", "README.md", "# Work\nStill available")
    result = service.sync(mode)
    assert result["processed"] == 1
    assert result["deleted"] == 0
    assert any("repo" in error for error in result["errors"])
    assert service.keyword_store.chunks_by_path(old.to_key())
    assert service.graph_store.note_meta_for(old.to_key()) is not None


def test_file_sync_rejects_escaping_symlink(service: RagService) -> None:
    outside = _write(service, "work", "Private.md", "# Private\nPrivate work note")
    work_root = service.config.banks["work"].path
    learning_root = service.config.banks["learning"].path
    (learning_root / "Linked.md").symlink_to(work_root / outside.path)
    result = service.sync("full")
    assert result["processed"] == 1
    with pytest.raises(ValueError, match="inside"):
        service.sync("file", file=FileRef("learning", "Linked.md"))


def test_bank_watch_ignores_other_roots_and_index_data(
    service: RagService, monkeypatch: pytest.MonkeyPatch
) -> None:
    from watchfiles import Change

    from second_brain.watcher import VaultWatcher

    ref = _write(service, "learning", "Watched.md", "# Watched\nLearn here")
    other = _write(service, "work", "Unwatched.md", "# Other\nWork here")
    bank = service.config.banks[ref.bank]
    monkeypatch.setattr(
        "second_brain.watcher.watch",
        lambda *args, **kwargs: iter([{
            (Change.modified, str(bank.path / ref.path)),
            (Change.modified, str(service.config.banks[other.bank].path / other.path)),
            (Change.modified, str(service.config.fts_path)),
        }]),
    )
    VaultWatcher(service, debounce_seconds=0, bank=bank).run()
    assert service.sync_state.tracked_paths() == [ref.to_key()]
    assert service.sync_state.watcher_last_event() is not None


def test_failed_listing_does_not_delete_indexed_files(
    service: RagService, monkeypatch: pytest.MonkeyPatch
) -> None:
    from second_brain.sources import SourceListingError, iter_bank_files

    old = _write(service, "repo", "README.md", "# Repo\nPreviously indexed")
    assert service.sync("full")["processed"] == 1

    def list_bank(bank):
        if bank.id == "repo":
            raise SourceListingError("cannot list root")
        return iter_bank_files(bank)

    monkeypatch.setattr("second_brain.sources.iter_bank_files", list_bank)
    assert service.sync()["deleted"] == 0
    assert service.keyword_store.chunks_by_path(old.to_key())
    repo_status = next(bank for bank in service.status()["banks"] if bank["id"] == "repo")
    assert repo_status["indexed_files"] == 1
    assert repo_status["available"] is False


def test_bank_scope_precedes_keyword_and_vector_limits(service: RagService) -> None:
    class RankedEmbedder(StubEmbedder):
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.5, 0.5] if "weak" in text else [1.0, 0.0] for text in texts]

    service.embedder = service.indexer.embedder = RankedEmbedder()
    other = _write(service, "work", "Needle.md", "# Strong\nneedle needle needle")
    wanted = _write(service, "learning", "Needle.md", "# Weak\nneedle weak match")
    repo = _write(service, "repo", "README.md", "# Repo\nneedle documentation")
    assert service.sync("full")["processed"] == 3
    all_hits = service.search("needle", verbose=False)["hits"]
    assert {hit["file"]["bank"] for hit in all_hits} == {wanted.bank, other.bank, repo.bank}
    assert all_hits[0]["file"] == other.to_dict()

    hits = service.search(
        "needle", banks=["learning"], filters={"kind": "markdown", "path_prefix": "Needle"},
        top_k=1, verbose=False,
    )["hits"]
    assert [hit["file"] for hit in hits] == [wanted.to_dict()]
    assert hits[0]["semantic_score"] is not None
    assert hits[0]["keyword_score"] is not None
    scoped = service.search("needle", banks=["learning", "repo"], verbose=False)["hits"]
    assert {hit["file"]["bank"] for hit in scoped} == {"learning", "repo"}
    with pytest.raises(ValueError, match="unknown bank"):
        service.search("needle", banks=["missing"])


def test_fetch_round_trips_citation_to_ordered_redacted_chunks(service: RagService) -> None:
    ref = _write(
        service, "learning", "README.md", "# First\nneedle private-token\n\n## Second\nnext passage"
    )
    _write(service, "work", "README.md", "# Other\nneedle elsewhere")
    assert service.sync("full")["processed"] == 2
    result = service.query("needle", banks=["learning"], verbose=False)
    citation = result["citations"][0]
    assert citation["file"] == ref.to_dict()
    assert citation["bank_type"] == "obsidian"
    _write(service, "learning", ref.path, "# Updated\ncontent changed without syncing")
    fetched = service.fetch(FileRef(**citation["file"]))
    assert fetched["file"] == ref.to_dict()
    assert fetched["found"] is True
    assert [chunk["text"].strip() for chunk in fetched["chunks"]] == [
        "needle [REDACTED]", "next passage",
    ]
    assert fetched["chunks"][0]["chunk_id"] == citation["chunk_id"]
    assert service.fetch(FileRef("work", "missing.md"))["found"] is False

    root = service.config.banks["repo"].path
    root.rename(root.with_name("unmounted"))
    inventory = {bank["id"]: bank for bank in service.list_banks()}
    assert inventory["repo"] == {
        "id": "repo", "name": "repo", "type": "repo", "available": False,
    }
