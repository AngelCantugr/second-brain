"""Tree-sitter symbol outlines: extraction per language, chunking, redaction, indexing."""

from pathlib import Path

import pytest

pytest.importorskip("tree_sitter")

from second_brain.chunker import chunk_note, embedding_text  # noqa: E402
from second_brain.code import extract_symbols, load_code_file  # noqa: E402
from second_brain.config import CodeSource, RagConfig, load_config  # noqa: E402
from second_brain.graph import GraphStore  # noqa: E402
from second_brain.indexer import Indexer  # noqa: E402
from second_brain.keyword_store import KeywordStore  # noqa: E402
from second_brain.redaction import Redactor  # noqa: E402
from second_brain.sources import iter_code_source_files  # noqa: E402
from second_brain.sync_state import SyncStateStore  # noqa: E402
from second_brain.vector_store import InMemoryVectorStore  # noqa: E402

PYTHON = '''"""Retry helpers for flaky network calls."""

import time

API_KEY = "sk-live-123"


# Backoff is capped so a long outage never sleeps for minutes.
@functools.cache
def backoff(attempt: int) -> float:
    """Return the delay before retry ``attempt``."""
    return min(2 ** attempt, 30)


def undocumented(x):
    y = x + 1
    return y


class Retrier(Base):
    """Retries a call with exponential backoff."""

    def run(self, fn):
        """Call ``fn`` until it succeeds."""
        while True:
            def inner():
                pass
'''

GO = '''package retry

import "time"

// Backoff returns the delay before retry n.
// It is capped at 30 seconds.
func Backoff(n int) time.Duration {
	return time.Second
}

// Retrier retries calls.
type Retrier struct {
	Max int
}

// Run calls fn until it succeeds.
func (r *Retrier) Run(fn func() error) error { return nil }

// Doer does things.
type Doer interface {
	Do() error
}
'''

TYPESCRIPT = '''import { sleep } from "./sleep";

/** Delay before retry `n`, capped at 30s. */
export function backoff(n: number): number {
  return Math.min(2 ** n, 30);
}

/** Retries a call. */
export class Retrier {
  /** Call fn until it succeeds. */
  async run(fn: () => Promise<void>): Promise<void> {}
}

// Options for a retrier.
export interface RetryOptions { max: number }

export const jitter = (ms: number): number => ms * Math.random();
'''


def _by_name(symbols):
    return {symbol.qualname: symbol for symbol in symbols}


def test_python_symbols_carry_signature_docstring_comment_and_lines() -> None:
    symbols = _by_name(extract_symbols(PYTHON, "python"))

    assert list(symbols) == ["(module)", "backoff", "undocumented", "Retrier", "Retrier.run"]
    assert symbols["(module)"].doc == "Retry helpers for flaky network calls."
    backoff = symbols["backoff"]
    assert backoff.symbol_kind == "function" and backoff.parent is None
    assert backoff.signature.startswith("@functools.cache\ndef backoff(attempt: int) -> float:")
    assert backoff.doc == (
        "Backoff is capped so a long outage never sleeps for minutes.\n\n"
        "Return the delay before retry ``attempt``."
    )
    assert (backoff.start_line, backoff.end_line) == (9, 12)  # from the decorator
    # The outline is the why: comment, signature and docstring, not the body.
    assert backoff.text.startswith("# Backoff is capped")
    assert "return min" not in backoff.text
    run = symbols["Retrier.run"]
    assert (run.symbol_kind, run.parent, run.name) == ("method", "Retrier", "run")
    assert "inner" not in symbols  # nested functions are details


def test_undocumented_symbol_falls_back_to_its_first_body_lines() -> None:
    undocumented = _by_name(extract_symbols(PYTHON, "python"))["undocumented"]

    assert undocumented.doc == ""
    assert "return y" in undocumented.text


def test_go_symbols_include_methods_with_receiver_parent_and_types() -> None:
    symbols = _by_name(extract_symbols(GO, "go"))

    assert list(symbols) == ["Backoff", "Retrier", "Retrier.Run", "Doer"]
    assert symbols["Backoff"].doc == "Backoff returns the delay before retry n.\nIt is capped at 30 seconds."
    assert symbols["Backoff"].signature == "func Backoff(n int) time.Duration"
    assert symbols["Retrier"].symbol_kind == "type"
    assert "Max int" in symbols["Retrier"].text
    assert symbols["Retrier.Run"].parent == "Retrier"
    assert symbols["Doer"].symbol_kind == "interface"


def test_typescript_symbols_unwrap_exports_and_arrow_functions() -> None:
    symbols = _by_name(extract_symbols(TYPESCRIPT, "typescript"))

    assert list(symbols) == ["backoff", "Retrier", "Retrier.run", "RetryOptions", "jitter"]
    assert symbols["backoff"].doc == "Delay before retry `n`, capped at 30s."
    assert symbols["backoff"].signature.startswith("export function backoff(n: number): number")
    assert symbols["Retrier.run"].doc == "Call fn until it succeeds."
    assert symbols["RetryOptions"].symbol_kind == "interface"
    assert symbols["jitter"].symbol_kind == "function"


def test_tsx_files_use_the_tsx_grammar(tmp_path: Path) -> None:
    path = tmp_path / "Button.tsx"
    path.write_text("/** A button. */\nexport function Button() { return <b>hi</b>; }\n", encoding="utf-8")

    parsed = load_code_file(path, "Button.tsx", "ui", "repo:ui/Button.tsx")

    assert [s.qualname for s in parsed.symbols] == ["Button"]
    assert parsed.metadata == {"repo": "ui", "language": "typescript", "file": "Button.tsx"}


def test_code_file_becomes_one_chunk_per_symbol_with_symbol_metadata(tmp_path: Path) -> None:
    path = tmp_path / "retry.py"
    path.write_text(PYTHON, encoding="utf-8")
    parsed = load_code_file(path, "pkg/retry.py", "sb", "repo:sb/pkg/retry.py")

    chunks = chunk_note(parsed, chunk_size=200, chunk_overlap=0)

    assert parsed.kind == "code" and parsed.title == "sb: pkg/retry.py"
    assert parsed.links == [] and parsed.tags == []
    assert [c.heading_path for c in chunks] == [
        "(module)", "backoff", "undocumented", "Retrier", "Retrier.run",
    ]
    run = chunks[-1].metadata
    assert run["kind"] == "code" and run["repo"] == "sb" and run["language"] == "python"
    assert run["file"] == "pkg/retry.py" and run["path"] == "repo:sb/pkg/retry.py"
    assert (run["symbol"], run["symbol_name"], run["parent_symbol"]) == ("Retrier.run", "run", "Retrier")
    assert run["symbol_kind"] == "method" and run["doc"] == "Call ``fn`` until it succeeds."
    assert run["start_line"] < run["end_line"]
    assert embedding_text(chunks[1]).startswith("sb: pkg/retry.py > backoff\n# Backoff")
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_long_outline_is_cut_to_chunk_size_words(tmp_path: Path) -> None:
    path = tmp_path / "long.py"
    words = " ".join(f"w{i}" for i in range(50))
    path.write_text(f'def f():\n    """{words}"""\n', encoding="utf-8")
    parsed = load_code_file(path, "long.py", "sb", "repo:sb/long.py")

    (chunk,) = chunk_note(parsed, chunk_size=10, chunk_overlap=0)

    assert len(chunk.text.split()) == 10


def test_redaction_covers_symbol_outlines(tmp_path: Path) -> None:
    path = tmp_path / "secret.py"
    path.write_text('def f(key="sk-live-123"):\n    """Uses sk-live-123."""\n', encoding="utf-8")
    parsed = load_code_file(path, "secret.py", "sb", "repo:sb/secret.py")

    redacted = Redactor([r"sk-live-\d+"]).note(parsed)
    (chunk,) = chunk_note(redacted, chunk_size=100, chunk_overlap=0)

    assert "sk-live" not in chunk.text and "sk-live" not in chunk.metadata["doc"]
    assert "[REDACTED]" in redacted.symbols[0].signature


def _config_file(tmp_path: Path, extra: str) -> Path:
    path = tmp_path / "rag_config.toml"
    path.write_text(
        'vault_path = "./vault"\nqdrant_path = "./q"\nfts_path = "./f.sqlite"\n' + extra,
        encoding="utf-8",
    )
    return path


def test_languages_config_validates_names(tmp_path: Path) -> None:
    config = load_config(
        _config_file(tmp_path, '[[code_sources]]\nname = "a"\npath = "."\nlanguages = ["python", "go"]\n')
    )
    assert config.code_sources[0].languages == ("python", "go")
    assert config.code_sources[0].code_globs == ("*",)

    with pytest.raises(ValueError, match="unsupported"):
        load_config(_config_file(tmp_path, '[[code_sources]]\nname = "a"\npath = "."\nlanguages = ["cobol"]\n'))


def test_languages_config_names_the_missing_extra(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)

    with pytest.raises(ValueError, match=r"second-brain-mcp\[code\]"):
        load_config(_config_file(tmp_path, '[[code_sources]]\nname = "a"\npath = "."\nlanguages = ["go"]\n'))


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    for rel, text in {
        "README.md": "# Repo\nUses `backoff`.\n",
        "pkg/retry.py": PYTHON,
        "pkg/retry.go": GO,
        "web/retry.ts": TYPESCRIPT,
        "node_modules/lib/index.ts": "export function vendored() {}\n",
        "pkg/data.json": "{}",
    }.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return repo


def test_listing_adds_source_files_only_for_configured_languages(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    docs_only = iter_code_source_files(CodeSource("sb", repo))
    python = iter_code_source_files(CodeSource("sb", repo, languages=("python",)))
    scoped = iter_code_source_files(
        CodeSource("sb", repo, languages=("python", "go", "typescript"), code_globs=("pkg/**",))
    )

    assert [f.rel_path for f in docs_only] == ["repo:sb/README.md"]
    assert [f.rel_path for f in python] == ["repo:sb/README.md", "repo:sb/pkg/retry.py"]
    assert [f.rel_path for f in scoped] == [
        "repo:sb/README.md", "repo:sb/pkg/retry.go", "repo:sb/pkg/retry.py",
    ]


class _StubEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]


def test_sync_indexes_symbol_chunks_and_filters_by_kind(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Note.md").write_text("# Note\nretry notes\n", encoding="utf-8")
    config = RagConfig(
        vault_path=vault,
        qdrant_path=tmp_path / "q",
        fts_path=tmp_path / "fts.sqlite",
        sync_state_path=tmp_path / "s.sqlite",
        code_sources=[CodeSource("sb", _repo(tmp_path), languages=("python", "go", "typescript"))],
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

    result = indexer.sync(mode="full")

    assert result.errors == [] and result.processed == 5
    go_chunks = indexer.keyword_store.chunks_by_path("repo:sb/pkg/retry.go")
    assert {c.metadata["symbol"] for c in go_chunks} == {"Backoff", "Retrier", "Retrier.Run", "Doer"}
    code_hits = indexer.keyword_store.search("backoff", filters={"kind": "code"})
    assert code_hits and all(hit.metadata["kind"] == "code" for hit in code_hits)
    # Code files are graph nodes without vault-graph edges.
    assert indexer.graph_store.all_edges() == []
