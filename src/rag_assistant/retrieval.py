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
    """Fuse several ranked lists into one. RRF score = sum of 1/(rrf_k + rank) across lists.

    This is where "hybrid" happens. Each list votes for its items: rank 1 is worth 1/(k+1),
    rank 2 is worth 1/(k+2), and so on — a chunk that ranks well in BOTH lists collects both
    votes and floats to the top. `rrf_k` (default 60, set via RRF_K in config.py) controls how
    steep the vote drop-off is: a smaller k makes top ranks dominate, a bigger k treats
    rank 1 and rank 10 more equally. This constant is the only "weight" hybrid has — the dense
    and sparse lists are otherwise treated identically.
    """
    fused: dict[str, float] = {}
    best_chunk: dict[str, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            cid = item.chunk.id
            # Add this list's vote to the chunk's running total.
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            best_chunk.setdefault(cid, item)
    # Highest combined vote first.
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
        # Deliberately over-fetch (`candidate_k`, default 20) then cut down to `final_k`
        # (default 5) at the end — fusion and reranking work better with more raw material.
        if mode == "dense":
            candidates = self._dense(query, self._candidate_k)
        elif mode == "sparse":
            candidates = self._bm25.search(query, self._candidate_k)
        elif mode == "hybrid":
            # Run BOTH retrievers, then merge their rankings with RRF (see above).
            dense = self._dense(query, self._candidate_k)
            sparse = self._bm25.search(query, self._candidate_k)
            candidates = reciprocal_rank_fusion([dense, sparse], rrf_k=self._rrf_k)
        else:
            raise ValueError(f"Unknown retrieval mode '{mode}' (dense|sparse|hybrid).")

        # Optional precision stage: re-score the shortlist with a cross-encoder (rerank.py).
        if rerank:
            return self._reranker.rerank(query, candidates, final_k)
        return candidates[:final_k]
