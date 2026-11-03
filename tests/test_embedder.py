"""OllamaEmbedder batching against a mocked ``requests.post``.

No Ollama server is needed: each fake response returns one vector per input,
derived from the input text, so tests can check order and batch boundaries.
"""

from __future__ import annotations

import pytest
import requests

from second_brain import embedder as embedder_module
from second_brain.embedder import EmbeddingError, OllamaEmbedder


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload
        self.ok = status_code < 400
        self.text = str(payload)

    def json(self) -> dict:
        return self._payload


def _vector_for(text: str) -> list[float]:
    return [float(len(text)), float(ord(text[0]))]


class _FakeOllama:
    """Records each request and answers like ``/api/embed``."""

    def __init__(self, fail_on_call: int | None = None, drop_one: bool = False) -> None:
        self.calls: list[dict] = []
        self.fail_on_call = fail_on_call
        self.drop_one = drop_one

    def post(self, url: str, json: dict, timeout: int) -> _FakeResponse:
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if self.fail_on_call == len(self.calls):
            return _FakeResponse(
                500, {"error": "the input length exceeds the context length"}
            )
        vectors = [_vector_for(text) for text in json["input"]]
        if self.drop_one:
            vectors = vectors[:-1]
        return _FakeResponse(200, {"model": json["model"], "embeddings": vectors})


@pytest.fixture
def fake_ollama(monkeypatch: pytest.MonkeyPatch) -> _FakeOllama:
    fake = _FakeOllama()
    monkeypatch.setattr(embedder_module.requests, "post", fake.post)
    return fake


def test_embed_sends_batches_to_api_embed_and_preserves_order(
    fake_ollama: _FakeOllama,
) -> None:
    texts = [f"{chr(97 + i)}{'x' * i}" for i in range(7)]
    embedder = OllamaEmbedder("http://ollama:11434/", "nomic-embed-text", batch_size=3)

    vectors = embedder.embed(texts)

    assert vectors == [_vector_for(text) for text in texts]
    assert [call["json"]["input"] for call in fake_ollama.calls] == [
        texts[0:3],
        texts[3:6],
        texts[6:7],
    ]
    first = fake_ollama.calls[0]
    assert first["url"] == "http://ollama:11434/api/embed"
    assert first["json"]["model"] == "nomic-embed-text"
    # Legacy /api/embeddings rejected over-length input; /api/embed would
    # silently truncate it unless told not to.
    assert first["json"]["truncate"] is False


def test_embed_uses_one_request_when_texts_fit_in_a_batch(
    fake_ollama: _FakeOllama,
) -> None:
    embedder = OllamaEmbedder("http://ollama:11434", "nomic-embed-text", batch_size=32)

    vectors = embedder.embed(["alpha", "beta"])

    assert len(fake_ollama.calls) == 1
    assert vectors == [_vector_for("alpha"), _vector_for("beta")]


def test_embed_of_empty_list_makes_no_request(fake_ollama: _FakeOllama) -> None:
    embedder = OllamaEmbedder("http://ollama:11434", "nomic-embed-text")

    assert embedder.embed([]) == []
    assert fake_ollama.calls == []


def test_failed_batch_raises_with_ollama_error_and_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeOllama(fail_on_call=2)
    monkeypatch.setattr(embedder_module.requests, "post", fake.post)
    embedder = OllamaEmbedder("http://ollama:11434", "nomic-embed-text", batch_size=2)

    with pytest.raises(EmbeddingError) as exc_info:
        embedder.embed(["a1", "b2", "c3", "d4", "e5"])

    message = str(exc_info.value)
    assert "batch 2/3" in message
    assert "texts 2-3" in message
    assert "HTTP 500" in message
    assert "the input length exceeds the context length" in message
    # No retry and no later batches once one fails.
    assert len(fake.calls) == 2


def test_connection_error_is_wrapped_with_batch_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(url: str, json: dict, timeout: int) -> _FakeResponse:
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(embedder_module.requests, "post", refuse)
    embedder = OllamaEmbedder("http://ollama:11434", "nomic-embed-text")

    with pytest.raises(
        EmbeddingError, match="batch 1/1.*connection refused"
    ) as exc_info:
        embedder.embed(["alpha"])
    assert isinstance(exc_info.value.__cause__, requests.ConnectionError)


def test_response_with_wrong_vector_count_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeOllama(drop_one=True)
    monkeypatch.setattr(embedder_module.requests, "post", fake.post)
    embedder = OllamaEmbedder("http://ollama:11434", "nomic-embed-text")

    with pytest.raises(EmbeddingError, match="returned 1 embeddings for 2 inputs"):
        embedder.embed(["alpha", "beta"])


@pytest.mark.parametrize("batch_size", [0, -1])
def test_batch_size_must_be_positive(batch_size: int) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        OllamaEmbedder("http://ollama:11434", "nomic-embed-text", batch_size=batch_size)


def test_non_json_success_body_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class _HtmlResponse(_FakeResponse):
        def json(self) -> dict:
            raise ValueError("Expecting value")

    monkeypatch.setattr(
        embedder_module.requests,
        "post",
        lambda url, json, timeout: _HtmlResponse(200, {}),
    )
    embedder = OllamaEmbedder("http://ollama:11434", "nomic-embed-text")

    with pytest.raises(EmbeddingError, match="returned 0 embeddings for 1 inputs"):
        embedder.embed(["alpha"])
