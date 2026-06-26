"""Retrieval: dense, sparse, and hybrid (fused) — with an optional rerank stage.

The headline method is `retrieve(query, mode="hybrid")`. Hybrid fuses the dense and sparse
rankings with **Reciprocal Rank Fusion (RRF)**, which combines lists by *rank* rather than by raw
score — so it doesn't matter that cosine similarity and BM25 live on different scales. Exposing
the mode lets the eval harness measure dense vs sparse vs hybrid (± rerank) head to head.
"""

from __future__ import annotations

from rag_assistant.embeddings import Embedder
from rag_assistant.models import RetrievedChunk
from rag_assistant.rerank import NoopReranker, Reranker
from rag_assistant.sparse import BM25Index
from rag_assistant.vectorstore import VectorStore


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]], *, rrf_k: int = 60
) -> list[RetrievedChunk]:
    """Fuse several ranked lists into one. RRF score = sum of 1/(rrf_k + rank) across lists."""
    fused: dict[str, float] = {}
    best_chunk: dict[str, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            cid = item.chunk.id
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            best_chunk.setdefault(cid, item)
    ordered = sorted(fused.items(), key=lambda kv: -kv[1])
    return [
        RetrievedChunk(chunk=best_chunk[cid].chunk, score=score, source="hybrid")
        for cid, score in ordered
    ]


class Retriever:
    """Owns the dense store, the sparse index, the embedder, and an optional reranker."""

    def __init__(
        self,
        *,
        vector_store: VectorStore,
        bm25: BM25Index,
        embedder: Embedder,
        reranker: Reranker | None = None,
        candidate_k: int = 20,
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> None:
        self._store = vector_store
        self._bm25 = bm25
        self._embedder = embedder
        self._reranker = reranker or NoopReranker()
        self._candidate_k = candidate_k
        self._top_k = top_k
        self._rrf_k = rrf_k

    def _dense(self, query: str, k: int) -> list[RetrievedChunk]:
        embedding = self._embedder.embed([query])[0]
        return self._store.search(embedding, k)

    def retrieve(
        self,
        query: str,
        *,
        mode: str = "hybrid",
        k: int | None = None,
        rerank: bool = False,
    ) -> list[RetrievedChunk]:
        """Return the top-k chunks for `query` using the chosen retrieval mode."""
        final_k = k or self._top_k
        if mode == "dense":
            candidates = self._dense(query, self._candidate_k)
        elif mode == "sparse":
            candidates = self._bm25.search(query, self._candidate_k)
        elif mode == "hybrid":
            dense = self._dense(query, self._candidate_k)
            sparse = self._bm25.search(query, self._candidate_k)
            candidates = reciprocal_rank_fusion([dense, sparse], rrf_k=self._rrf_k)
        else:
            raise ValueError(f"Unknown retrieval mode '{mode}' (dense|sparse|hybrid).")

        if rerank:
            return self._reranker.rerank(query, candidates, final_k)
        return candidates[:final_k]
