"""Gemini live paths, tested offline with mocked clients.

These tests patch `google.genai.Client` so no network call ever happens — they verify that we
call the SDK with the right arguments and translate its responses into our own types correctly.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from rag_assistant.config import Settings
from rag_assistant.embeddings import GeminiEmbedder
from rag_assistant.errors import ConfigError
from rag_assistant.factory import build_answerer, build_embedder, build_vector_store
from rag_assistant.generation import GeminiAnswerer, extract_citations
from rag_assistant.models import Chunk, RetrievedChunk


def _contexts(n: int = 1) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk=Chunk(id=f"doc::{i}", doc_id="doc", text=f"Passage {i} about RRF.", index=i),
            score=0.9,
            source="hybrid",
        )
        for i in range(n)
    ]


def _settings(**overrides: object) -> Settings:
    """Settings isolated from the machine's real env vars / .env file."""
    base: dict[str, object] = {"gemini_api_key": None, "_env_file": None}
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


# --- GeminiEmbedder ---


def test_gemini_embedder_returns_normalized_vectors_in_order() -> None:
    fake_client = MagicMock()
    fake_client.models.embed_content.return_value = SimpleNamespace(
        embeddings=[SimpleNamespace(values=[3.0, 4.0]), SimpleNamespace(values=[0.0, 2.0])]
    )
    with patch("google.genai.Client", return_value=fake_client):
        embedder = GeminiEmbedder(model="gemini-embedding-001", dim=2, api_key="test-key")
    vectors = embedder.embed(["first text", "second text"])

    assert vectors == [[0.6, 0.8], [0.0, 1.0]]  # L2-normalized, input order preserved
    for vec in vectors:
        assert math.isclose(math.sqrt(sum(x * x for x in vec)), 1.0)


def test_gemini_embedder_requests_configured_model_and_dim() -> None:
    fake_client = MagicMock()
    fake_client.models.embed_content.return_value = SimpleNamespace(
        embeddings=[SimpleNamespace(values=[1.0] * 8)]
    )
    with patch("google.genai.Client", return_value=fake_client):
        embedder = GeminiEmbedder(model="gemini-embedding-001", dim=8, api_key="test-key")
    embedder.embed(["hello"])

    call = fake_client.models.embed_content.call_args
    assert call.kwargs["model"] == "gemini-embedding-001"
    assert call.kwargs["contents"] == ["hello"]
    assert call.kwargs["config"].output_dimensionality == 8


def test_gemini_embedder_batches_large_inputs() -> None:
    fake_client = MagicMock()
    fake_client.models.embed_content.return_value = SimpleNamespace(
        embeddings=[SimpleNamespace(values=[1.0, 0.0])] * 100
    )
    with patch("google.genai.Client", return_value=fake_client):
        embedder = GeminiEmbedder(model="gemini-embedding-001", dim=2, api_key="test-key")
    embedder.embed([f"text {i}" for i in range(250)])

    # 250 texts at a batch size of 100 → 3 API calls.
    assert fake_client.models.embed_content.call_count == 3


# --- GeminiAnswerer ---


def test_gemini_answerer_returns_cited_answer() -> None:
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = SimpleNamespace(
        text="RRF combines lists by rank. [1]"
    )
    answerer = GeminiAnswerer(model="gemini-2.5-flash", max_tokens=256, api_key="test-key")
    with patch("google.genai.Client", return_value=fake_client):
        answer = answerer.answer("How are ranked lists combined?", _contexts())

    assert answer.text == "RRF combines lists by rank. [1]"
    assert answer.citations and answer.citations[0].doc_id == "doc"

    call = fake_client.models.generate_content.call_args
    assert call.kwargs["model"] == "gemini-2.5-flash"
    assert "Passage 0 about RRF." in call.kwargs["contents"]  # context passed in prompt
    assert call.kwargs["config"].max_output_tokens == 256
    assert "cite" in (call.kwargs["config"].system_instruction or "")


def test_gemini_answerer_without_contexts_never_calls_the_api() -> None:
    answerer = GeminiAnswerer(model="gemini-2.5-flash", max_tokens=256, api_key="test-key")
    with patch("google.genai.Client") as client_cls:
        answer = answerer.answer("anything?", [])
    assert "don't know" in answer.text.lower()
    client_cls.assert_not_called()


# --- Citation extraction ---


def test_citations_reflect_only_cited_passages() -> None:
    contexts = _contexts(3)
    citations = extract_citations("The answer is X [1][3].", contexts)
    assert [c.chunk_id for c in citations] == ["doc::0", "doc::2"]


def test_out_of_range_markers_are_ignored() -> None:
    contexts = _contexts(2)
    citations = extract_citations("Something [1], and a hallucinated [9].", contexts)
    assert [c.chunk_id for c in citations] == ["doc::0"]


def test_uncited_answer_falls_back_to_all_contexts() -> None:
    contexts = _contexts(2)
    citations = extract_citations("An answer with no markers.", contexts)
    assert len(citations) == 2  # provenance is never silently empty


# --- Factory ---


def test_factory_builds_gemini_when_key_is_set() -> None:
    settings = _settings(gemini_api_key="test-key")
    with patch("google.genai.Client", return_value=MagicMock()):
        embedder = build_embedder(settings)
    assert isinstance(embedder, GeminiEmbedder)
    assert embedder.dim == settings.gemini_embedding_dim

    answerer = build_answerer(settings)
    assert isinstance(answerer, GeminiAnswerer)
    assert answerer._model == "gemini-2.5-flash"  # asserting internal wiring


def test_factory_fails_loudly_without_gemini_key() -> None:
    settings = _settings()
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        build_embedder(settings)
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        build_answerer(settings)


def test_factory_fails_loudly_without_pinecone_key() -> None:
    settings = _settings(vector_store="pinecone", pinecone_api_key=None)
    with pytest.raises(ConfigError, match="PINECONE_API_KEY"):
        build_vector_store(settings)


def test_factory_rejects_unknown_vector_store() -> None:
    settings = _settings(vector_store="bogus")
    with pytest.raises(ConfigError, match="bogus"):
        build_vector_store(settings)
