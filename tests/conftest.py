"""Shared fixtures and offline test doubles.

Production code has exactly one real path (Gemini + memory/Pinecone), so the offline stand-ins
live HERE, as test doubles: `StubEmbedder` is a deterministic bag-of-words embedder (hash each
token into a bucket — real lexical similarity, no network), `StubAnswerer` returns a canned
cited answer. They implement the same `Embedder`/`Answerer` protocols the real classes do,
which is exactly the point of coding against protocols: the whole pipeline — chunking,
retrieval, fusion, evaluation, caching — is exercised for real with no API keys or network.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterator

import pytest

from rag_assistant.debuglog import debug_enabled
from rag_assistant.embeddings import _log_embed_request, _log_embed_response
from rag_assistant.generation import _SYSTEM, _log_answer_request, _log_answer_response
from rag_assistant.models import Answer, Citation, RetrievedChunk
from rag_assistant.pipeline import RAGPipeline
from rag_assistant.vectorstore import InMemoryVectorStore

_TOKEN = re.compile(r"[a-z0-9]+")

# A tiny corpus with one distinct topic per doc, so relevance labels are unambiguous.
TEST_DOCS: dict[str, str] = {
    "pgvector": (
        "pgvector is a Postgres extension for storing and querying vector embeddings. It adds a "
        "`vector` column type and distance operators such as `<=>` for cosine distance. For large "
        "collections you create an HNSW index to make approximate nearest-neighbour search fast. "
        "Keeping embeddings in Postgres lets you query them alongside your relational data."
    ),
    "bm25": (
        "BM25 is a classic lexical ranking function. It scores a document by how many query terms "
        "it contains, weighting each term by its inverse document frequency so that rare terms "
        "count more, and saturating term frequency so repeated words give diminishing returns. "
        "BM25 excels at exact keyword matches that dense embeddings can miss."
    ),
    "rrf": (
        "Reciprocal Rank Fusion combines several ranked lists into one. Each item gets a score of "
        "one divided by a constant plus its rank in each list, and the scores are summed. Because "
        "it uses rank rather than raw score, RRF fuses results from systems on different scales, "
        "like cosine similarity and BM25, without any normalization."
    ),
    "rerank": (
        "A cross-encoder reranker re-scores candidate passages by feeding the query and passage "
        "together through a model, which is far more accurate than comparing independent "
        "embeddings. It is expensive, so you only rerank the top candidates from a cheaper "
        "first-stage retriever. Retrieve broadly, then rerank precisely."
    ),
}

# Golden set for TEST_DOCS — mirrors the shape of src/rag_assistant/golden.py.
TEST_GOLDEN: list[dict[str, object]] = [
    {"question": "How do I store vector embeddings in Postgres?", "relevant_doc_ids": ["pgvector"]},
    {"question": "Which index makes vector search fast?", "relevant_doc_ids": ["pgvector"]},
    {"question": "What does BM25 reward in a document?", "relevant_doc_ids": ["bm25"]},
    {"question": "How are two ranked lists combined into one?", "relevant_doc_ids": ["rrf"]},
    {"question": "Why use a cross-encoder over plain embeddings?", "relevant_doc_ids": ["rerank"]},
]


class StubEmbedder:
    """Deterministic offline embedder: hash each token into one of `dim` buckets and count."""

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.md5(token.encode()).digest()  # noqa: S324 - not security-sensitive
            idx = int.from_bytes(digest[:4], "big") % self.dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm else vec

    def embed(self, texts: list[str]) -> list[list[float]]:
        _log_embed_request("offline fake embedder", f"hashing-{self.dim}d", texts)
        vectors = [self._embed_one(t) for t in texts]
        _log_embed_response("offline fake embedder", vectors)
        return vectors


class StubAnswerer:
    """Deterministic answerer: canned, cited answer with no model call."""

    def answer(self, question: str, contexts: list[RetrievedChunk]) -> Answer:
        _log_answer_request("offline fake answerer", _SYSTEM, question, contexts)
        if not contexts:
            text = "I don't know — no relevant context found."
            _log_answer_response("offline fake answerer", text)
            return Answer(question=question, text=text)
        top = contexts[0].chunk
        text = f"Based on {len(contexts)} passage(s), see doc '{top.doc_id}'. [1]"
        _log_answer_response("offline fake answerer", text)
        citations = [
            Citation(chunk_id=c.chunk.id, doc_id=c.chunk.doc_id, quote=c.chunk.text[:160])
            for c in contexts
        ]
        return Answer(question=question, text=text, citations=citations, contexts=contexts)


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
        embedder=StubEmbedder(dim=128),
        vector_store=InMemoryVectorStore(),
        answerer=StubAnswerer(),
        chunk_size=400,
        chunk_overlap=60,
        candidate_k=10,
        top_k=5,
    )
    for doc_id, text in TEST_DOCS.items():
        pipeline.ingest(doc_id, text)
    return pipeline


@pytest.fixture
def pipeline() -> RAGPipeline:
    return make_pipeline()
