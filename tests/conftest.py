"""Shared fixtures: a fully offline pipeline (hashing embedder + in-memory store + fake answerer).

No API keys, no database, no network — the whole RAG pipeline runs in-process, so tests exercise
chunking, retrieval, fusion, and evaluation for real.
"""

from __future__ import annotations

import pytest

from rag_assistant.embeddings import HashingEmbedder
from rag_assistant.generation import FakeAnswerer
from rag_assistant.pipeline import RAGPipeline
from rag_assistant.sample_data import SAMPLE_DOCS
from rag_assistant.vectorstore import InMemoryVectorStore


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
