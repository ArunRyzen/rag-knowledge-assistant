"""Gemini live paths, tested offline with mocked clients.

These tests patch `google.genai.Client` so no network call ever happens — they verify that we
call the SDK with the right arguments and translate its responses into our own types correctly.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rag_assistant.config import Settings
from rag_assistant.embeddings import GeminiEmbedder, HashingEmbedder
from rag_assistant.factory import build_answerer, build_embedder
from rag_assistant.generation import FakeAnswerer, LLMAnswerer
from rag_assistant.models import Chunk, RetrievedChunk


def _contexts() -> list[RetrievedChunk]:
    chunk = Chunk(id="doc::0", doc_id="doc", text="RRF fuses ranked lists by rank.", index=0)
    return [RetrievedChunk(chunk=chunk, score=0.9, source="hybrid")]


def _settings(**overrides: object) -> Settings:
    """Settings isolated from the machine's real env vars / .env file."""
    base: dict[str, object] = {
        "gemini_api_key": None,
        "openai_api_key": None,
        "anthropic_api_key": None,
        "_env_file": None,
    }
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


# --- Gemini answerer ---


def test_gemini_answerer_returns_cited_answer() -> None:
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = SimpleNamespace(
        text="RRF combines lists by rank. [1]"
    )
    answerer = LLMAnswerer(
        provider="gemini", model="gemini-2.5-flash", max_tokens=256, api_key="test-key"
    )
    with patch("google.genai.Client", return_value=fake_client):
        answer = answerer.answer("How are ranked lists combined?", _contexts())

    assert answer.text == "RRF combines lists by rank. [1]"
    assert answer.citations and answer.citations[0].doc_id == "doc"

    call = fake_client.models.generate_content.call_args
    assert call.kwargs["model"] == "gemini-2.5-flash"
    assert "RRF fuses ranked lists by rank." in call.kwargs["contents"]  # context passed in prompt
    assert call.kwargs["config"].max_output_tokens == 256
    assert "cite" in (call.kwargs["config"].system_instruction or "")


def test_gemini_answerer_without_contexts_never_calls_the_api() -> None:
    answerer = LLMAnswerer(
        provider="gemini", model="gemini-2.5-flash", max_tokens=256, api_key="test-key"
    )
    with patch("google.genai.Client") as client_cls:
        answer = answerer.answer("anything?", [])
    assert "don't know" in answer.text.lower()
    client_cls.assert_not_called()


# --- Factory selection ---


def test_factory_prefers_gemini_when_key_is_set() -> None:
    settings = _settings(gemini_api_key="test-key")
    with patch("google.genai.Client", return_value=MagicMock()):
        embedder = build_embedder(settings)
    assert isinstance(embedder, GeminiEmbedder)
    assert embedder.dim == settings.gemini_embedding_dim

    answerer = build_answerer(settings)
    assert isinstance(answerer, LLMAnswerer)
    assert answerer._provider == "gemini"  # asserting internal wiring
    assert answerer._model == "gemini-2.5-flash"


def test_factory_defaults_to_offline_fakes_without_keys() -> None:
    settings = _settings()
    assert isinstance(build_embedder(settings), HashingEmbedder)
    assert isinstance(build_answerer(settings), FakeAnswerer)
