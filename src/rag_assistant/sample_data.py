"""A tiny built-in corpus + golden eval set.

Bundled in code so the CLI and tests work out of the box regardless of working directory. Each
document covers a distinct topic, so document-level relevance labels are unambiguous. Point the CLI
at your own folder with `--data` to use real content.
"""

from __future__ import annotations

SAMPLE_DOCS: dict[str, str] = {
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

# Golden questions: each is answered by exactly one document.
GOLDEN: list[dict[str, object]] = [
    {"question": "How do I store vector embeddings in Postgres?", "relevant_doc_ids": ["pgvector"]},
    {"question": "Which index makes vector search fast?", "relevant_doc_ids": ["pgvector"]},
    {"question": "What does BM25 reward in a document?", "relevant_doc_ids": ["bm25"]},
    {"question": "How are two ranked lists combined into one?", "relevant_doc_ids": ["rrf"]},
    {"question": "Why use a cross-encoder over plain embeddings?", "relevant_doc_ids": ["rerank"]},
]
