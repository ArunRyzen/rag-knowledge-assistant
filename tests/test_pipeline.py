"""End-to-end pipeline: ingest, ask, and grounded citations."""

from __future__ import annotations

from rag_assistant.embeddings import HashingEmbedder
from rag_assistant.generation import FakeAnswerer
from rag_assistant.pipeline import RAGPipeline
from rag_assistant.vectorstore import InMemoryVectorStore
from tests.conftest import make_pipeline


def test_ask_returns_answer_with_contexts() -> None:
    pipeline = make_pipeline()
    answer = pipeline.ask("How are two ranked lists combined?", mode="hybrid", k=3)
    assert answer.text
    assert answer.contexts
    assert answer.citations
    # Citations point back to real chunks.
    assert all(c.chunk_id for c in answer.citations)


def test_empty_index_answers_dont_know() -> None:
    pipeline = RAGPipeline(
        embedder=HashingEmbedder(dim=64),
        vector_store=InMemoryVectorStore(),
        answerer=FakeAnswerer(),
    )
    answer = pipeline.ask("anything?")
    assert "don't know" in answer.text.lower()


def test_ingest_counts_chunks() -> None:
    pipeline = RAGPipeline(
        embedder=HashingEmbedder(dim=64),
        vector_store=InMemoryVectorStore(),
        answerer=FakeAnswerer(),
        chunk_size=100,
        chunk_overlap=20,
    )
    n = pipeline.ingest("d", "word " * 200)
    assert n >= 1
