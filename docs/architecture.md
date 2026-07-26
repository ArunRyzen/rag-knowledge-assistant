# Architecture & Design Decisions

Why the pipeline is shaped the way it is. Read alongside the source.

## The pipeline

```
ingest:  documents ─▶ chunk ─▶ embed ─▶ [vector store] + [BM25 index]
ask:     question  ─▶ retrieve (dense + sparse → RRF) ─▶ rerank? ─▶ generate (cited)
```

Each stage is an interface (`Embedder`, `VectorStore`, `Reranker`, `Answerer`). The pipeline depends
only on those abstractions, so backends swap without touching the orchestration — and the whole thing
runs offline for tests.

## Key decisions

### 1. Hybrid retrieval, fused with RRF
Dense embeddings capture meaning but miss exact terms (names, codes, rare keywords); BM25 nails exact
terms but misses paraphrase. We run both and fuse with **Reciprocal Rank Fusion**: each item scores
`1 / (rrf_k + rank)` summed across lists. Because RRF uses **rank, not raw score**, it fuses systems
on totally different scales (cosine ∈ [-1,1] vs unbounded BM25) with **no normalization** — the common
failure mode of score-weighted fusion. The eval harness shows hybrid ≥ either alone.

**Alternative considered:** weighted score combination. Rejected — requires per-corpus score
normalization and tuning; RRF is parameter-light and robust.

### 2. Two-stage retrieve-then-rerank
First-stage retrieval optimizes recall cheaply over the whole corpus. A **cross-encoder reranker**
then re-scores only the top candidates by reading query + passage *together* — far more accurate, far
too expensive to run corpus-wide. Default is a no-op reranker; the cross-encoder is an optional extra
so the core stays light. This mirrors how production RAG actually trades cost for precision.

### 3. Swappable stores: in-memory for experiments, Pinecone for persistence
`InMemoryVectorStore` (numpy cosine) needs no infrastructure — instant, free, gone at process
exit. `PineconeVectorStore` is the persistent path: a managed serverless vector database, upserts
idempotent by chunk id, chunk text carried in record metadata so results need no second lookup.
Same `VectorStore` protocol, so the retriever is identical. (Honest scope note: BM25 is in-memory
in both paths — with Pinecone, `rag ask` rebuilds the sparse index locally from `data/` while
dense search hits the persistent index; Pinecone's sparse vectors could make hybrid fully
persistent later. The fusion logic would be unchanged.)

### 4. One real path, loud failures, stubs at the seams
Gemini is required — one key covers semantic embeddings AND synthesis; a missing key raises a
`ConfigError` naming exactly what to set, instead of silently degrading. Offline testability is
preserved where it belongs: `tests/conftest.py` injects stub implementations of the same
`Embedder`/`Answerer` protocols, so the full pipeline — chunking, retrieval, fusion, evaluation —
still runs in CI with no keys or network.

### 5. Grounded, cited generation
The generator is instructed to answer **only** from numbered contexts, cite them, or say it doesn't
know. Passing retrieved chunks as numbered context + attaching them as citations is what makes the
answer attributable instead of a confident hallucination. Faithfulness *scoring* (LLM-as-judge) comes
in Milestone 4.

### 6. Evaluation is a first-class feature, not an afterthought
`evaluation.py` turns "the RAG feels good" into **recall@k** and **MRR** over a labelled golden set,
and `compare_modes` runs dense / sparse / hybrid / +rerank head-to-head. Relevance is keyed by source
**document id**, which is robust to chunking choices. This is the "ship gate" muscle — built here,
deepened in Milestone 4.

## Chunking notes
Recursive, structure-aware splitting (paragraph → line → sentence → word) with overlap. Overlap
prefixes each chunk with the previous chunk's tail so a fact split across a boundary survives. Chunk
size is the highest-leverage knob: too big bloats context and hurts precision; too small loses the
context a passage needs. Tune it against the eval harness, not by feel.

## Trade-offs left open
- Persistent sparse retrieval (currently in-memory BM25 → Pinecone sparse vectors or Postgres FTS).
- Hosted reranker (Voyage/Cohere) vs the local cross-encoder.
- Query-side techniques: multi-query expansion (Task 6 in `tasks/`), HyDE, query rewriting.
- Answer-quality / faithfulness evals — Milestone 4.
