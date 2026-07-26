# Lessons Learned

Notes to my future self from building this (Milestone 2).

## Technical
- **Evaluation changes how you build.** Once `rag eval` existed, every decision (chunk size, hybrid
  vs dense, rerank) became a measurement instead of an argument. Build the harness early.
- **RRF is underrated.** Fusing by rank sidesteps the whole "how do I normalize cosine against BM25"
  problem that sinks weighted-score fusion. Simple and robust.
- **A good offline default unlocks everything.** The hashing embedder + fake answerer mean the repo
  runs, tests, and demos with zero keys or infra — which made TDD on retrieval logic fast and free.
  A deterministic-but-real fake beats a mock.
- **Protocols at every seam paid off again.** Embedder / VectorStore / Reranker / Answerer are all
  swappable; `factory.py` is the only place that knows which concrete class runs.
- **Keep relevance labels chunk-agnostic.** Keying golden relevance to *document id* (not chunk id)
  meant the eval set didn't break every time I changed chunk size.

## Process
- **Scope honesty matters.** BM25 is in-memory even in the pgvector path; I documented that rather
  than pretend it's full Postgres FTS. Naming a limitation is more credible than hiding it.
- **Library first, transports thin.** CLI and API are tiny shells over `RAGPipeline`; all the logic
  (and all the tests) live in the library.

## If I did it again
- Add faithfulness/answer evals from the start (Milestone 4 will retrofit them here).
- Move sparse retrieval into Postgres for the production path so hybrid is fully persistent.

---

## Update (July 2026): the real-stack refactor

The project moved from "offline-first with optional live mode" to **one real path**: Gemini
(required) + Pinecone (persistent vectors), with pgvector and the OpenAI/Anthropic branches
removed. What I learned in the process:

- **The offline fakes did their job, then became test doubles.** They bootstrapped TDD when the
  project had no keys; once real keys existed, keeping them in the production factory meant a
  typo'd key silently degraded to fake answers. Moving them to `tests/conftest.py` kept the free
  offline test suite AND made production failures loud. Fakes belong at the seams, not in the
  factory.
- **A persistent store splits the pipeline into write path and read path.** With everything
  in-memory, ingest-per-run hid the distinction; Pinecone forced an explicit `rag ingest` (chunk →
  embed → upsert, idempotent by chunk id) separate from query-time — which is how production RAG
  actually works, eventual consistency and all.
- **Protocols proved their worth again**: swapping pgvector for Pinecone touched one class, one
  factory branch, and zero retrieval logic.
