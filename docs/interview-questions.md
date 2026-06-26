# RAG Interview Questions This Project Answers

---

### Q. Walk me through a production RAG pipeline.
Ingest (chunk → embed → index), retrieve (hybrid: dense + sparse, fused), optionally rerank, then
generate an answer grounded in the retrieved context with citations — and **evaluate** retrieval with
recall@k / MRR so changes are measurable. The LLM is maybe 20% of it; retrieval quality and evaluation
are the rest.

### Q. How do you choose a chunk size?
It's the highest-leverage knob. Too large → imprecise retrieval and bloated, costly context (and
"lost in the middle"). Too small → chunks lack the context to be meaningful. Use structure-aware
splitting with overlap so boundary-straddling facts survive, then **tune against an eval set**, not by
feel. There's no universal number; it depends on document structure and query type.

### Q. Why hybrid search instead of dense-only?
Dense embeddings capture semantics but miss exact terms — product codes, names, rare keywords. BM25
(lexical) nails those but misses paraphrase. Combining them covers both failure modes; the eval
harness in this repo shows hybrid ≥ either alone.

### Q. What is Reciprocal Rank Fusion and why use it?
RRF merges ranked lists by summing `1/(k + rank)` across lists. It uses **rank, not score**, so it
fuses systems on incompatible scales (cosine vs BM25) without normalization or tuning — which is
exactly the trap of weighted-score fusion.

### Q. Why add a reranker, and why not rerank everything?
A cross-encoder scores query + passage *together*, which is much more accurate than comparing
independent embeddings — but quadratically expensive, so you can't run it over the whole corpus. The
pattern is retrieve broadly (cheap, high recall) then rerank the top-N (expensive, high precision).

### Q. How do you evaluate a RAG system?
Two layers. **Retrieval:** recall@k (did a relevant doc make the top-k?) and MRR (how high did the
first relevant doc rank?) against a labelled golden set — this repo's harness. **Generation:**
faithfulness (is the answer supported by the context?) and answer relevance, typically via LLM-as-judge
(Milestone 4). You need a golden set and a numeric gate to ship safely.

### Q. What are common RAG failure modes?
Retrieval misses (wrong/no chunk retrieved), "lost in the middle" (right chunk retrieved but ignored
in a long context), hallucination beyond the context, stale data, and bad chunking splitting a fact in
half. Mitigations: hybrid + rerank, place top chunks at context edges, instruct "answer only from
context / say you don't know," refresh the index, and overlap chunks.

### Q. Embeddings vs a vector database — what's the difference?
The **embedding model** turns text into vectors (the semantics). The **vector store/DB** indexes those
vectors and answers nearest-neighbour queries fast (HNSW/IVF), with filtering and persistence. This
repo separates them cleanly — `Embedder` vs `VectorStore`.

### Q. How would you scale this and control cost?
Batch embeddings, cache them, use an ANN index (HNSW) in pgvector, cap candidate/top-k, cache repeat
queries, and pick a cheaper embedding/generation tier — validating each change against the eval set so
quality doesn't silently regress.
