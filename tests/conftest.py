"""Shared fixtures: a fully offline pipeline (hashing embedder + in-memory store + fake answerer).

No API keys, no database, no network — the whole RAG pipeline runs in-process, so tests exercise
chunking, retrieval, fusion, and evaluation for real.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from rag_assistant.debuglog import debug_enabled
from rag_assistant.embeddings import HashingEmbedder
from rag_assistant.generation import FakeAnswerer
from rag_assistant.pipeline import RAGPipeline
from rag_assistant.sample_data import SAMPLE_DOCS
from rag_assistant.vectorstore import InMemoryVectorStore


@pytest.fixture(autouse=True)
def _pin_llm_debug(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep every test hermetic no matter what the developer's real ``.env`` says.

    ``debug_enabled()`` falls back to a ``.env`` file in the current directory, and a developer
    may well have ``LLM_DEBUG=1`` in theirs. Because the real environment variable always beats
    the file, pinning it to ``"0"`` here guarantees debug tracing stays off unless a test
    explicitly opts in with ``monkeypatch.setenv("LLM_DEBUG", "1")``. The cache is cleared on
    both sides of the test so no test ever sees a value cached by a neighbour.
    """
    monkeypatch.setenv("LLM_DEBUG", "0")
    debug_enabled.cache_clear()
    yield
    debug_enabled.cache_clear()


def make_pipeline() -> RAGPipeline:
    pipeline = RAGPipeline(
        embedder=HashingEmbedder(dim=128),
        vector_store=InMemoryVectorStore(),
        answerer=FakeAnswerer(),
        chunk_size=400,
        chunk_overlap=60,
        candidate_k=10,
        top_k=5,
    )
    for doc_id, text in SAMPLE_DOCS.items():
        pipeline.ingest(doc_id, text)
    return pipeline


@pytest.fixture
def pipeline() -> RAGPipeline:
    return make_pipeline()
