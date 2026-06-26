"""Retrieval modes and Reciprocal Rank Fusion."""

from __future__ import annotations

from rag_assistant.models import Chunk, RetrievedChunk
from rag_assistant.retrieval import reciprocal_rank_fusion
from tests.conftest import make_pipeline


def _rc(cid: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(chunk=Chunk(id=cid, doc_id=cid, text=cid, index=0), score=score)


def test_rrf_rewards_items_ranked_high_in_both_lists() -> None:
    list_a = [_rc("x", 0.9), _rc("y", 0.8), _rc("z", 0.1)]
    list_b = [_rc("y", 5.0), _rc("x", 4.0), _rc("w", 1.0)]
    fused = reciprocal_rank_fusion([list_a, list_b], rrf_k=60)
    ids = [r.chunk.id for r in fused]
    # x and y appear near the top of both lists, so they should lead.
    assert set(ids[:2]) == {"x", "y"}
    assert "w" in ids and "z" in ids  # union of both lists


def test_hybrid_retrieves_the_relevant_document() -> None:
    pipeline = make_pipeline()
    results = pipeline.retrieve("How do I store vectors in Postgres?", mode="hybrid", k=3)
    assert results
    assert any(r.chunk.doc_id == "pgvector" for r in results)


def test_unknown_mode_raises() -> None:
    pipeline = make_pipeline()
    try:
        pipeline.retrieve("q", mode="bogus")
    except ValueError as exc:
        assert "bogus" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
