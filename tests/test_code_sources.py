"""Code sources: config, .gitignore-aware listing, repo_doc indexing, isolation from the vault graph."""

import shutil
import subprocess
from pathlib import Path

import pytest

from second_brain.config import DEFAULT_DOC_GLOBS, CodeSource, RagConfig, load_config
from second_brain.graph import GraphStore
from second_brain.indexer import Indexer
from second_brain.keyword_store import KeywordStore, matches_filters
from second_brain.service import RagService
from second_brain.sources import (
    MAX_SOURCE_BYTES,
    code_source_path,
    iter_code_source_files,
    split_code_source_path,
)
from second_brain.sync_state import SyncStateStore
from second_brain.vector_store import InMemoryVectorStore


class _StubEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    _write(repo / "README.md", "# Repo\nWhy this repo exists.\n")
    _write(repo / "CHANGELOG.md", "# Changelog\n- first release\n")
    _write(repo / "docs" / "design.md", "# Design\nWe chose RRF over score blending.\n")
    _write(repo / "docs" / "adr" / "0001-sqlite.md", "# Use SQLite\nFTS5 is enough.\n")
    _write(repo / "pkg" / "README.md", "# Pkg\nNested readme.\n")
    _write(repo / "pkg" / "notes.md", "# Not a doc\nOutside doc_globs.\n")
    _write(repo / "pkg" / "main.py", "print('code is not a repo doc')\n")
    _write(repo / "node_modules" / "lib" / "README.md", "# Vendored\n")
    _write(repo / "dist" / "docs" / "built.md", "# Built\n")
    _write(repo / ".github" / "docs" / "hidden.md", "# Hidden\n")
    return repo


def _indexer(tmp_path: Path, vault: Path, sources: list[CodeSource]) -> Indexer:
    config = RagConfig(
        vault_path=vault,
        qdrant_path=tmp_path / "qdrant",
        fts_path=tmp_path / "fts.sqlite",
        sync_state_path=tmp_path / "sync_state.sqlite",
        chunk_size=50,
        chunk_overlap=0,
        code_sources=sources,
    )
    indexer = Indexer(
        config=config,
        embedder=_StubEmbedder(),
        vector_store=InMemoryVectorStore(),
        keyword_store=KeywordStore(config.fts_path),
        sync_state=SyncStateStore(config.sync_state_path),
        graph_store=GraphStore(config.fts_path),
    )
    indexer.initialize()
    return indexer


# -- config -----------------------------------------------------------------


def _config_file(tmp_path: Path, extra: str) -> Path:
    path = tmp_path / "rag_config.toml"
    path.write_text(
        'vault_path = "./vault"\nqdrant_path = "./q"\nfts_path = "./f.sqlite"\n' + extra,
        encoding="utf-8",
    )
    return path


def test_code_sources_default_to_none(tmp_path: Path) -> None:
    assert load_config(_config_file(tmp_path, "")).code_sources == []


def test_code_sources_parse_with_defaults_and_relative_paths(tmp_path: Path) -> None:
    config = load_config(
        _config_file(
            tmp_path,
            '[[code_sources]]\nname = "sb"\npath = "./repos/sb"\n'
            '[[code_sources]]\nname = "other"\npath = "/abs/other"\nexclude_globs = ["docs/old/**"]\n',
        )
    )

    first, second = config.code_sources
    assert first == CodeSource("sb", tmp_path / "repos" / "sb")
    assert first.doc_globs == DEFAULT_DOC_GLOBS
    assert second.exclude_globs == ("docs/old/**",)


@pytest.mark.parametrize(
    ("toml", "message"),
    [
        ('code_sources = "x"', "array of tables"),
        ('[[code_sources]]\nname = "a/b"\npath = "."', "name must be"),
        ('[[code_sources]]\npath = "."', "name must be"),
        ('[[code_sources]]\nname = "a"', "path must be"),
        ('[[code_sources]]\nname = "a"\npath = "."\nlanguage = "py"', "unknown keys"),
        ('[[code_sources]]\nname = "a"\npath = "."\ndoc_globs = "README*"', "doc_globs"),
        ('[[code_sources]]\nname = "a"\npath = "."\n[[code_sources]]\nname = "a"\npath = "x"', "used by another"),
    ],
)
def test_code_sources_reject_invalid_entries(tmp_path: Path, toml: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(_config_file(tmp_path, toml))


# -- paths and listing -----------------------------------------------------------


def test_code_source_paths_round_trip() -> None:
    path = code_source_path("sb", "docs/a.md")

    assert path == "repo:sb/docs/a.md"
    assert split_code_source_path(path) == ("sb", "docs/a.md")
    assert split_code_source_path("Notes/repo.md") is None
    assert split_code_source_path("repo:sb") is None


def test_listing_selects_docs_and_skips_vendored_hidden_and_non_doc_files(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write(repo / "docs" / "huge.md", "x" * (MAX_SOURCE_BYTES + 1))

    files = iter_code_source_files(CodeSource("sb", repo))

    assert [f.rel_path for f in files] == [
        "repo:sb/CHANGELOG.md",
        "repo:sb/README.md",
        "repo:sb/docs/adr/0001-sqlite.md",
        "repo:sb/docs/design.md",
        "repo:sb/pkg/README.md",
    ]
    assert all(f.repo == "sb" for f in files)
    assert files[1].abs_path == (repo / "README.md").resolve()


def test_listing_applies_source_exclude_globs_and_custom_doc_globs(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    excluded = iter_code_source_files(CodeSource("sb", repo, exclude_globs=("docs/adr/**",)))
    custom = iter_code_source_files(CodeSource("sb", repo, doc_globs=("pkg/*.md",)))

    assert "repo:sb/docs/adr/0001-sqlite.md" not in {f.rel_path for f in excluded}
    assert [f.rel_path for f in custom] == ["repo:sb/pkg/README.md", "repo:sb/pkg/notes.md"]


def test_listing_never_follows_a_symlink_out_of_the_repo(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    outside = _write(tmp_path / "secret" / "docs.md", "# Outside\n")
    (repo / "docs" / "leak.md").symlink_to(outside)

    paths = {f.rel_path for f in iter_code_source_files(CodeSource("sb", repo))}

    assert "repo:sb/docs/leak.md" not in paths


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_listing_honors_gitignore_in_a_git_repo(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write(repo / ".gitignore", "docs/design.md\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)

    paths = {f.rel_path for f in iter_code_source_files(CodeSource("sb", repo))}

    assert "repo:sb/docs/design.md" not in paths
    # Untracked files that are not ignored are still listed.
    assert "repo:sb/README.md" in paths


# -- indexing ----------------------------------------------------------------------


def test_sync_indexes_repo_docs_with_kind_repo_and_title(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    _write(vault / "Note.md", "# Note\nA vault note.\n")
    indexer = _indexer(tmp_path, vault, [CodeSource("sb", _repo(tmp_path))])

    result = indexer.sync(mode="full")

    assert result.errors == []
    assert result.processed == 6
    hits = indexer.keyword_store.chunks_by_path("repo:sb/docs/design.md")
    assert hits and hits[0].metadata["kind"] == "repo_doc"
    assert hits[0].metadata["repo"] == "sb"
    assert hits[0].metadata["note_title"] == "sb: design"
    assert "repo" not in indexer.keyword_store.chunks_by_path("Note.md")[0].metadata


def test_repo_filter_matches_only_named_code_sources() -> None:
    assert matches_filters({"repo": "sb"}, {"repo": "sb"})
    assert matches_filters({"repo": "sb"}, {"repo": ["x", "sb"]})
    assert not matches_filters({"path": "Note.md"}, {"repo": "sb"})
    with pytest.raises(ValueError, match="repo"):
        matches_filters({"repo": "sb"}, {"repo": 3})


def test_code_sources_stay_out_of_the_vault_graph(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    # [[README]] must stay a ghost link, not resolve to the repo's README.
    _write(vault / "A.md", "# A\nSee [[B]] and [[README]].\n")
    _write(vault / "B.md", "# B\nSee [[A]].\n")
    indexer = _indexer(tmp_path, vault, [CodeSource("sb", _repo(tmp_path))])

    indexer.sync(mode="full")

    edges = indexer.graph_store.all_edges()
    assert {(e.src, e.dst) for e in edges} == {("A.md", "B.md")}
    assert indexer.graph_store.note_meta_for("repo:sb/README.md") is not None


def test_dropping_a_code_source_deletes_its_files_on_next_sync(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    _write(vault / "Note.md", "# Note\nbody\n")
    repo = _repo(tmp_path)
    indexer = _indexer(tmp_path, vault, [CodeSource("sb", repo)])
    indexer.sync(mode="full")

    indexer.config.code_sources = []
    result = indexer.sync(mode="incremental")

    assert result.deleted == 5
    assert indexer.sync_state.tracked_paths() == ["Note.md"]
    assert indexer.keyword_store.chunks_by_path("repo:sb/README.md") == []


def test_missing_code_source_root_keeps_its_entries_and_reports_it(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    _write(vault / "Note.md", "# Note\nbody\n")
    repo = _repo(tmp_path)
    indexer = _indexer(tmp_path, vault, [CodeSource("sb", repo)])
    indexer.sync(mode="full")

    shutil.rmtree(repo)
    result = indexer.sync(mode="incremental")

    assert result.deleted == 0
    assert any("code source 'sb'" in error for error in result.errors)
    assert "repo:sb/README.md" in indexer.sync_state.tracked_paths()


def test_incremental_sync_reindexes_only_changed_repo_docs(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    repo = _repo(tmp_path)
    indexer = _indexer(tmp_path, vault, [CodeSource("sb", repo)])
    indexer.sync(mode="full")

    _write(repo / "docs" / "design.md", "# Design\nWe switched to weighted RRF.\n")
    (repo / "CHANGELOG.md").unlink()
    result = indexer.sync(mode="incremental")

    assert (result.processed, result.skipped, result.deleted) == (1, 3, 1)


def test_status_counts_code_source_files(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    _write(vault / "Note.md", "# Note\nbody\n")
    repo = _repo(tmp_path)
    config = RagConfig(
        vault_path=vault,
        qdrant_path=tmp_path / "qdrant",
        fts_path=tmp_path / "fts.sqlite",
        sync_state_path=tmp_path / "sync_state.sqlite",
        code_sources=[CodeSource("sb", repo)],
    )
    service = RagService(config, use_in_memory_vector=True)
    service.indexer.embedder = _StubEmbedder()
    service.sync(mode="full")

    _write(repo / "docs" / "new.md", "# New\n")
    _write(repo / "README.md", "# Repo\nChanged.\n")
    status = service.status()

    assert status["last_tracked_files"] == 6
    assert (status["stale_files"], status["untracked_files"], status["missing_files"]) == (1, 1, 0)
