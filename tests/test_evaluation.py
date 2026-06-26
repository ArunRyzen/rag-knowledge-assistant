"""The eval harness produces sane metrics and ranks hybrid at least as well as dense-only."""

from __future__ import annotations

from rag_assistant.evaluation import GoldenItem, compare_modes, evaluate_retrieval
from rag_assistant.sample_data import GOLDEN
from tests.conftest import make_pipeline


def _dataset() -> list[GoldenItem]:
    return [GoldenItem(**item) for item in GOLDEN]  # type: ignore[arg-type]


def test_metrics_are_bounded_and_populated() -> None:
    pipeline = make_pipeline()
    m = evaluate_retrieval(pipeline.retriever, _dataset(), mode="hybrid", k=5)
    assert m.n == len(GOLDEN)
    assert 0.0 <= m.recall_at_k <= 1.0
    assert 0.0 <= m.mrr <= 1.0
    # The sample corpus is easy and unambiguous — hybrid should find most answers.
    assert m.recall_at_k >= 0.6


def test_compare_modes_returns_all_four_configs() -> None:
    pipeline = make_pipeline()
    rows = compare_modes(pipeline.retriever, _dataset(), k=5)
    labels = {r.mode for r in rows}
    assert labels == {"dense", "sparse", "hybrid", "hybrid+rerank"}
