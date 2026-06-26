"""BM25: exact-keyword retrieval and rare-term weighting."""

from __future__ import annotations

from rag_assistant.chunking import chunk_document
from rag_assistant.sample_data import SAMPLE_DOCS
from rag_assistant.sparse import BM25Index


def _index() -> BM25Index:
    bm25 = BM25Index()
    for doc_id, text in SAMPLE_DOCS.items():
        bm25.add(chunk_document(doc_id=doc_id, text=text, size=400, overlap=60))
    return bm25


def test_keyword_query_finds_right_doc() -> None:
    bm25 = _index()
    results = bm25.search("HNSW index for Postgres vectors", k=3)
    assert results
    assert results[0].chunk.doc_id == "pgvector"


def test_zero_score_results_excluded() -> None:
    bm25 = _index()
    # A query sharing no terms with any doc should return nothing.
    assert bm25.search("xyzzy plugh frobnicate", k=5) == []
