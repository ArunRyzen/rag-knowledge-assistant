# Learning resources — RAG & embeddings, end to end

Curated links, ordered the way you should learn them. Each stage names the file in THIS project
where the concept lives — read the resource, then read the code, then do the matching task in
[`tasks/README.md`](../tasks/README.md). Prefer these over random blog posts: they're official
docs, original papers, or widely-trusted explainers.

## Stage 1 — Embeddings (the foundation)

| Resource | Why |
|---|---|
| [Gemini API: Embeddings guide](https://ai.google.dev/gemini-api/docs/embeddings) | The exact API this project calls in `embeddings.py` — dimensions, task types, normalization notes. |
| [Gemini cookbook: Talk to documents with embeddings](https://github.com/google-gemini/cookbook/blob/main/examples/Talk_to_documents_with_embeddings.ipynb) | A minimal RAG built with the same SDK — good to compare against this repo. |
| [Pinecone Learning Center](https://www.pinecone.io/learn/) | Free, high-quality explainers on vectors, similarity, ANN — the best single hub for retrieval fundamentals. |

**In this project:** `embeddings.py`, `data/embeddings.md`.

## Stage 2 — Vector databases & search

| Resource | Why |
|---|---|
| [Pinecone docs: Build a RAG chatbot](https://docs.pinecone.io/guides/get-started/build-a-rag-chatbot) | Official quickstart for the store you're actually using. |
| [Pinecone: HNSW explained](https://www.pinecone.io/learn/series/faiss/hnsw/) | The ANN index algorithm — expect an interview question on this. |
| [HNSW paper (arXiv 1603.09320)](https://arxiv.org/abs/1603.09320) | The original; skim after the explainer. |
| [Elastic: Practical BM25](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables) | The clearest walk-through of the BM25 formula and its k1/b parameters. |

**In this project:** `vectorstore.py`, `sparse.py`, `data/vector-databases.md`, `data/bm25.md`.

## Stage 3 — Retrieval quality: hybrid, fusion, reranking

| Resource | Why |
|---|---|
| [RRF paper (Cormack & Clarke, SIGIR 2009)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) | Two pages. The origin of the k=60 constant in `retrieval.py`. |
| [Sentence-Transformers docs](https://www.sbert.net/) | Bi-encoders vs cross-encoders, and the reranker family used in `rerank.py`. |
| [HyDE paper (arXiv 2212.10496)](https://arxiv.org/abs/2212.10496) | Hypothetical Document Embeddings — the query-side trick in `data/query-transformation.pdf`. |
| [Anthropic: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) | A modern chunking/retrieval upgrade; great "what would you improve?" interview material. |

**In this project:** `retrieval.py`, `rerank.py`, `data/hybrid-rrf.md`, `data/reranking.md`.

## Stage 4 — Generation & its failure modes

| Resource | Why |
|---|---|
| [Lost in the Middle (arXiv 2307.03172)](https://arxiv.org/abs/2307.03172) | Why context ORDER matters — cited in nearly every serious RAG discussion. |
| [OWASP GenAI: Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) | The retrieval-poisoning risk described in `data/grounded-generation.md`, formally. |

**In this project:** `generation.py`, `data/grounded-generation.md`.

## Stage 5 — Evaluation (what separates experts)

| Resource | Why |
|---|---|
| [RAGAS documentation](https://docs.ragas.io/) | The standard framework for faithfulness / answer-relevance / context-precision metrics. |
| [Langfuse: Evaluating RAG with RAGAS](https://langfuse.com/guides/cookbook/evaluation_of_rag_with_ragas) | A practical, runnable walkthrough of LLM-as-judge evaluation. |

**In this project:** `evaluation.py`, `golden.py`, `data/evaluation.md`.

## Stage 6 — Serving

| Resource | Why |
|---|---|
| [FastAPI docs](https://fastapi.tiangolo.com/) | The framework behind `api.py` — the tutorial's first 5 pages are enough. |
| [Pinecone: Vector DBs in production](https://www.pinecone.io/learn/series/vector-databases-in-production-for-busy-engineers/) | Multi-tenancy, freshness, and ops concerns interviewers like to probe. |

**In this project:** `api.py`, `cache.py`, `ratelimit.py`, `docs/deployment.md`.

## How to use this list in interview week

Don't read everything. Priority order if time is short: **Gemini embeddings guide → Pinecone
HNSW explainer → RRF paper (2 pages) → Lost in the Middle (abstract + figures) → RAGAS docs
front page.** Everything else is depth for after the interview — the `data/` corpus already
summarizes each topic, and you can literally `rag ask` it questions while you study.
