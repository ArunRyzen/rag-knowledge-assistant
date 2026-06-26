"""Retrieval evaluation harness.

"If you can't measure it, you can't ship it." This turns "the RAG feels good" into numbers:
**recall@k** (did a relevant doc make the top-k?), **MRR** (how high did the first relevant doc
rank?), and **hit-rate**. Run it across retrieval modes to *prove* that hybrid beats dense-only and
that reranking helps — the exact comparison an interviewer wants to see you reason about.

Relevance here is keyed by source document id (which doc holds the answer), which is robust to how
the documents happen to be chunked.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from rag_assistant.retrieval import Retriever


class GoldenItem(BaseModel):
    """One labelled query: the question and the doc id(s) that actually answer it."""

    question: str
    relevant_doc_ids: list[str]


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    mode: str
    k: int
    recall_at_k: float  # fraction of queries with a relevant doc in the top-k
    mrr: float  # mean reciprocal rank of the first relevant doc
    n: int

    def as_row(self) -> str:
        label = self.mode
        return (
            f"{label:16s} recall@{self.k}={self.recall_at_k:.2f}  MRR={self.mrr:.2f}  (n={self.n})"
        )


def evaluate_retrieval(
    retriever: Retriever,
    dataset: list[GoldenItem],
    *,
    mode: str = "hybrid",
    k: int = 5,
    rerank: bool = False,
    label: str | None = None,
) -> RetrievalMetrics:
    """Score one retrieval configuration over the golden set."""
    if not dataset:
        return RetrievalMetrics(mode=label or mode, k=k, recall_at_k=0.0, mrr=0.0, n=0)

    hits = 0
    reciprocal_ranks = 0.0
    for item in dataset:
        results = retriever.retrieve(item.question, mode=mode, k=k, rerank=rerank)
        relevant = set(item.relevant_doc_ids)
        first_rank = next(
            (i for i, r in enumerate(results, start=1) if r.chunk.doc_id in relevant), None
        )
        if first_rank is not None:
            hits += 1
            reciprocal_ranks += 1.0 / first_rank

    n = len(dataset)
    return RetrievalMetrics(
        mode=label or mode,
        k=k,
        recall_at_k=hits / n,
        mrr=reciprocal_ranks / n,
        n=n,
    )


def compare_modes(
    retriever: Retriever, dataset: list[GoldenItem], *, k: int = 5
) -> list[RetrievalMetrics]:
    """Run the standard comparison: dense, sparse, hybrid, and hybrid + rerank."""
    configs = [
        ("dense", False, "dense"),
        ("sparse", False, "sparse"),
        ("hybrid", False, "hybrid"),
        ("hybrid", True, "hybrid+rerank"),
    ]
    return [
        evaluate_retrieval(retriever, dataset, mode=mode, k=k, rerank=rerank, label=label)
        for mode, rerank, label in configs
    ]
