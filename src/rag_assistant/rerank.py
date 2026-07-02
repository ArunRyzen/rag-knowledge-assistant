"""Rerankers behind one interface.

Retrieval (dense/sparse) optimizes for recall over a large corpus with cheap scoring. A
**cross-encoder reranker** then re-scores the top candidates by feeding (query, chunk) *together*
through a model — far more accurate than comparing independent embeddings, but too expensive to
run over the whole corpus. So: retrieve many cheaply, rerank a few precisely. That two-stage shape
is the production RAG pattern.

`NoopReranker` is the default (identity). `CrossEncoderReranker` needs the `rerank` optional extra.
"""

from __future__ import annotations

from typing import Protocol

from rag_assistant.models import RetrievedChunk


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: list[RetrievedChunk], k: int
    ) -> list[RetrievedChunk]: ...


class NoopReranker:
    """Keeps the retriever's order; just truncates to k. The default."""

    def rerank(self, query: str, candidates: list[RetrievedChunk], k: int) -> list[RetrievedChunk]:
        return candidates[:k]


class CrossEncoderReranker:
    """Re-scores candidates with a cross-encoder (precision stage)."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[RetrievedChunk], k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        # The key difference from retrieval: the model reads the query AND the chunk together
        # (one pair per candidate), instead of comparing two separately computed vectors.
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self._model.predict(pairs)
        reranked = sorted(zip(candidates, scores, strict=True), key=lambda pair: -float(pair[1]))
        return [
            RetrievedChunk(chunk=c.chunk, score=float(s), source="reranked")
            for c, s in reranked[:k]
        ]
