"""The golden evaluation set for the `data/` corpus.

This is the labelled dataset `rag eval` scores against. Format: one entry per test question,
with the id(s) of the document(s) that contain the answer:

    {"question": "<your question>", "relevant_doc_ids": ["<doc id>"]}

Doc ids are filenames in `data/` without the extension (embeddings.md → "embeddings"). Labels
are keyed to DOCUMENTS, not chunks, so the set survives any change to chunking parameters.

Growing this set is Task 3 in `tasks/README.md` — every question you add makes the metrics less
noisy and the harness more trustworthy.
"""

from __future__ import annotations

GOLDEN: list[dict[str, object]] = [
    # embeddings.md
    {
        "question": "Why must query vectors come from the same model as document vectors?",
        "relevant_doc_ids": ["embeddings"],
    },
    {
        "question": "What does L2 normalization let us compute similarity with?",
        "relevant_doc_ids": ["embeddings"],
    },
    # chunking.md
    {
        "question": "What goes wrong when chunks are too large or too small?",
        "relevant_doc_ids": ["chunking"],
    },
    {
        "question": "What is the purpose of overlap between neighbouring chunks?",
        "relevant_doc_ids": ["chunking"],
    },
    # bm25.md
    {
        "question": "What do the k1 and b parameters control in BM25?",
        "relevant_doc_ids": ["bm25"],
    },
    {
        "question": "Which kinds of queries does lexical search handle better than embeddings?",
        "relevant_doc_ids": ["bm25"],
    },
    # hybrid-rrf.md
    {
        "question": "How does Reciprocal Rank Fusion combine two ranked lists?",
        "relevant_doc_ids": ["hybrid-rrf"],
    },
    {
        "question": "Why can't cosine scores and BM25 scores just be added together?",
        "relevant_doc_ids": ["hybrid-rrf"],
    },
    # reranking.md
    {
        "question": "What is the difference between a bi-encoder and a cross-encoder?",
        "relevant_doc_ids": ["reranking"],
    },
    {
        "question": "Which metrics improve most when you add a reranker?",
        "relevant_doc_ids": ["reranking"],
    },
    # vector-databases.md
    {
        "question": "How does the HNSW algorithm make nearest-neighbour search fast?",
        "relevant_doc_ids": ["vector-databases"],
    },
    {
        "question": "What does eventual consistency mean for Pinecone upserts?",
        "relevant_doc_ids": ["vector-databases"],
    },
    # evaluation.md
    {
        "question": "What is the difference between recall@k and MRR?",
        "relevant_doc_ids": ["evaluation"],
    },
    {
        "question": "Why label golden-set relevance at the document level instead of chunks?",
        "relevant_doc_ids": ["evaluation"],
    },
    # grounded-generation.md
    {
        "question": "How does the system prompt prevent hallucinated answers?",
        "relevant_doc_ids": ["grounded-generation"],
    },
    {
        "question": "What is prompt injection through retrieved documents?",
        "relevant_doc_ids": ["grounded-generation"],
    },
    # query-transformation.pdf — yes, a PDF: extraction happens in corpus.py, then it's
    # chunked/embedded exactly like every markdown file.
    {
        "question": "How does HyDE use a hypothetical answer to improve search?",
        "relevant_doc_ids": ["query-transformation"],
    },
    {
        "question": "What does multi-query expansion generate before fusing results?",
        "relevant_doc_ids": ["query-transformation"],
    },
]
