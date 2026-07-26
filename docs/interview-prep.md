# Interview prep — Q&A, resources, lessons

## The questions and model answers



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

---

## Learning resources — official docs & papers

Curated links, ordered the way you should learn them. Each stage names the file in THIS project
where the concept lives — read the resource, then read the code, then do the matching task in
[`tasks/README.md`](../tasks/README.md). Prefer these over random blog posts: they're official
docs, original papers, or widely-trusted explainers.

### Stage 1 — Embeddings (the foundation)

| Resource | Why |
|---|---|
| [Gemini API: Embeddings guide](https://ai.google.dev/gemini-api/docs/embeddings) | The exact API this project calls in `embeddings.py` — dimensions, task types, normalization notes. |
| [Gemini cookbook: Talk to documents with embeddings](https://github.com/google-gemini/cookbook/blob/main/examples/Talk_to_documents_with_embeddings.ipynb) | A minimal RAG built with the same SDK — good to compare against this repo. |
| [Pinecone Learning Center](https://www.pinecone.io/learn/) | Free, high-quality explainers on vectors, similarity, ANN — the best single hub for retrieval fundamentals. |

**In this project:** `embeddings.py`, the embeddings section of `docs/rag-concepts.md`.

### Stage 2 — Vector databases & search

| Resource | Why |
|---|---|
| [Pinecone docs: Build a RAG chatbot](https://docs.pinecone.io/guides/get-started/build-a-rag-chatbot) | Official quickstart for the store you're actually using. |
| [Pinecone: HNSW explained](https://www.pinecone.io/learn/series/faiss/hnsw/) | The ANN index algorithm — expect an interview question on this. |
| [HNSW paper (arXiv 1603.09320)](https://arxiv.org/abs/1603.09320) | The original; skim after the explainer. |
| [Elastic: Practical BM25](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables) | The clearest walk-through of the BM25 formula and its k1/b parameters. |

**In this project:** `vectorstore.py`, `sparse.py`, the vector-databases and BM25 sections of `docs/rag-concepts.md`.

### Stage 3 — Retrieval quality: hybrid, fusion, reranking

| Resource | Why |
|---|---|
| [RRF paper (Cormack & Clarke, SIGIR 2009)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) | Two pages. The origin of the k=60 constant in `retrieval.py`. |
| [Sentence-Transformers docs](https://www.sbert.net/) | Bi-encoders vs cross-encoders, and the reranker family used in `rerank.py`. |
| [HyDE paper (arXiv 2212.10496)](https://arxiv.org/abs/2212.10496) | Hypothetical Document Embeddings — the query-side trick in the query-transformation section of `docs/rag-concepts.md`. |
| [Anthropic: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) | A modern chunking/retrieval upgrade; great "what would you improve?" interview material. |

**In this project:** `retrieval.py`, `rerank.py`, the hybrid-RRF and reranking sections of `docs/rag-concepts.md`.

### Stage 4 — Generation & its failure modes

| Resource | Why |
|---|---|
| [Lost in the Middle (arXiv 2307.03172)](https://arxiv.org/abs/2307.03172) | Why context ORDER matters — cited in nearly every serious RAG discussion. |
| [OWASP GenAI: Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) | The retrieval-poisoning risk described in the grounded-generation section of `docs/rag-concepts.md`, formally. |

**In this project:** `generation.py`, `chat.py` (guardrails + web agent), the grounded-generation section of `docs/rag-concepts.md`.

### Stage 5 — Evaluation (what separates experts)

| Resource | Why |
|---|---|
| [RAGAS documentation](https://docs.ragas.io/) | The standard framework for faithfulness / answer-relevance / context-precision metrics. |
| [Langfuse: Evaluating RAG with RAGAS](https://langfuse.com/guides/cookbook/evaluation_of_rag_with_ragas) | A practical, runnable walkthrough of LLM-as-judge evaluation. |

**In this project:** `evaluation.py`, `golden.py`, the evaluation section of `docs/rag-concepts.md`.

### Stage 6 — Serving

| Resource | Why |
|---|---|
| [FastAPI docs](https://fastapi.tiangolo.com/) | The framework behind `api.py` — the tutorial's first 5 pages are enough. |
| [Pinecone: Vector DBs in production](https://www.pinecone.io/learn/series/vector-databases-in-production-for-busy-engineers/) | Multi-tenancy, freshness, and ops concerns interviewers like to probe. |

**In this project:** `api.py`, `cache.py`, `ratelimit.py`, the serving section of `docs/how-it-works.md`.

### Big documents to practice chunking on

The `data/` corpus (the Class 1 textbook) is small. Real chunking experience needs real size —
download from these free sources into a `bigdocs/` folder (gitignored) and point the CLI at it
with `--data .\bigdocs`. For more textbook terms/classes, the Tamil Nadu books portal is
[tntextbooks.in](https://www.tntextbooks.in/) (official SCERT PDFs, free):

| Source | What you get | Grab one |
|---|---|---|
| [Project Gutenberg](https://www.gutenberg.org/) | 70,000+ full public-domain books as plain `.txt` — hundreds of chunks each | `curl.exe -L -o bigdocs\frankenstein.txt https://www.gutenberg.org/cache/epub/84/pg84.txt` |
| [arXiv](https://arxiv.org/) | Research papers as PDF — realistic technical PDFs with sections/references | e.g. the RAG survey [arXiv 2312.10997](https://arxiv.org/abs/2312.10997) → "Download PDF" |
| [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany) | Company annual reports (10-K) — long, messy, table-heavy: the hard case | Search any company → 10-K filing |
| [RFC Editor](https://www.rfc-editor.org/) | Internet standards as clean `.txt` — long structured technical prose | e.g. [RFC 9110 (HTTP)](https://www.rfc-editor.org/rfc/rfc9110.txt) |
| [Wikipedia](https://en.wikipedia.org/wiki/Special:Export) | Any article as export; or just save a long article as text | Pick a topic you know deeply — best for judging retrieval quality |

Try: chunk a whole book offline first (no API cost) —

```powershell
uv run python -c "from pathlib import Path; from rag_assistant.chunking import chunk_document; t = Path('bigdocs/frankenstein.txt').read_text(encoding='utf-8'); print(len(chunk_document(doc_id='book', text=t, size=800, overlap=120)), 'chunks')"
```

— then `rag ingest --data .\bigdocs` when you're ready to spend embedding calls on it.
(Frankenstein: ~439k characters → ~769 chunks at the default size.)

### How to use this list in interview week

Don't read everything. Priority order if time is short: **Gemini embeddings guide → Pinecone
HNSW explainer → RRF paper (2 pages) → Lost in the Middle (abstract + figures) → RAGAS docs
front page.** Everything else is depth for after the interview — `docs/rag-concepts.md` already
summarizes each topic in one page apiece.

---

## Lessons learned building this

Notes to my future self from building this (Milestone 2).

### Technical
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

### Process
- **Scope honesty matters.** BM25 is in-memory even in the pgvector path; I documented that rather
  than pretend it's full Postgres FTS. Naming a limitation is more credible than hiding it.
- **Library first, transports thin.** CLI and API are tiny shells over `RAGPipeline`; all the logic
  (and all the tests) live in the library.

### If I did it again
- Add faithfulness/answer evals from the start (Milestone 4 will retrofit them here).
- Move sparse retrieval into Postgres for the production path so hybrid is fully persistent.

---

### Update (July 2026): the real-stack refactor

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
