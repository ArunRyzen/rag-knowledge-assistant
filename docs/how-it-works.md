# How it works — the full data journey, design decisions, and serving

This is the complete data journey through the system — every transformation, every storage
format, and the exact file where each step lives. Read it top to bottom once, then trace a real
run with `LLM_DEBUG=1` and watch each step happen.

There are two separate journeys, and keeping them apart in your head is the single most
clarifying idea in RAG:

- **The WRITE path** (`rag ingest`) — documents go IN. Runs once, or when documents change.
- **The READ path** (`rag ask`, or `POST /ask` on the API) — questions come in, answers go out.
  Runs on every question.

---

## The WRITE path: `rag ingest`

### Step 1 — Extraction: file → plain text (`corpus.py`)

`load_corpus()` walks the `data/` folder. For `.md`/`.txt` files the bytes on disk already ARE
text — one `read_text()` call. For `.pdf` files there's real work: a PDF internally stores
drawing instructions ("place these glyphs at coordinate x,y"), not paragraphs, so
`_extract_pdf()` uses `pypdf` to walk each page and reconstruct readable text. Pages are joined
with blank lines so the next step sees page breaks as paragraph boundaries.

> Whatever the format, the output is identical: a list of `(doc_id, text)` pairs, where
> `doc_id` is the filename without extension (`unit-1-my-pet.txt` → `"unit-1-my-pet"`). The
> bundled corpus — the Class 1 English textbook — was itself extracted from the official PDF
> exactly this way, then split into one file per unit.
> Everything downstream only ever sees plain text — this is why adding a new format (Word,
> HTML, OCR'd scans) would only ever touch this one file.

### Step 2 — Chunking: text → chunks (`chunking.py`)

`chunk_document()` splits each document into overlapping pieces of ~`CHUNK_SIZE` (800)
characters, cutting at the most natural boundary available — paragraph, then line, then
sentence, then word. Each chunk gets a **stable id** `"<doc_id>::<index>"` (e.g. `"unit-1-my-pet::2"`)
and carries its `doc_id` for provenance. The last 120 characters (`CHUNK_OVERLAP`) of each chunk
are repeated at the start of the next, so a fact cut at a boundary survives whole in at least
one chunk.

> Data shape now: `Chunk(id="unit-1-my-pet::2", doc_id="unit-1-my-pet", text="...", index=2)` — see `models.py`.

### Step 3 — Embedding: chunk text → vector (`embeddings.py`)

`GeminiEmbedder.embed()` sends the chunk texts (batched 100 at a time) to
`gemini-embedding-001`, which returns one **768-dimensional vector** per chunk — a point in
space where similar meanings sit close together. Each vector is L2-normalized (scaled to length
1) so that later, comparing two vectors is a plain dot product = cosine similarity.

> Data shape now: `Chunk` + `[0.0123, -0.0456, ...]` (768 floats), paired.

### Step 4a — Dense storage: vector → Pinecone record (`vectorstore.py`)

`PineconeVectorStore.add()` turns each (chunk, vector) pair into one Pinecone **record**:

```python
{
  "id": "unit-1-my-pet::2",                     # the stable chunk id — upserting again OVERWRITES
  "values": [0.0123, -0.0456, ...],    # the 768-dim embedding
  "metadata": {"doc_id": "unit-1-my-pet", "index": 2, "text": "..."}   # the chunk itself, carried along
}
```

Storing the text IN the record's metadata is a deliberate choice: search results come back
self-contained, no second database needed to look up what the vector meant. This is what you
see in the Pinecone console. Writes are *eventually consistent* — a record takes a few seconds
to become searchable.

### Step 4b — Sparse indexing: chunk tokens → BM25 index (`sparse.py`)

The same chunks are also tokenized (lowercased words) into an in-memory BM25 index that scores
by exact word overlap, weighted by term rarity. **Every chunk is indexed twice** — once by
meaning (vector), once by exact words (BM25). That dual indexing is what makes hybrid retrieval
possible. The BM25 side lives in-process, which is why the read path rebuilds it (cheaply, no
API calls) at startup.

**Write path complete.** 3 textbook units → ~30 chunks → ~30 vectors in Pinecone + a BM25
index. Nothing about your question has happened yet.

---

## The READ path: `rag ask "..."`

### Step 5 — The question becomes a vector too (`retrieval.py` → `embeddings.py`)

Your question goes through the SAME embedder as the documents did (this matters: vectors from
different models are incomparable). One API call, one 768-dim query vector.

### Step 6 — Two searches run (`vectorstore.py` + `sparse.py`)

- **Dense:** Pinecone finds the 20 (`CANDIDATE_K`) stored vectors nearest the query vector —
  "what MEANS most like this question?"
- **Sparse:** BM25 scores chunks by shared rare words — "what CONTAINS these exact terms?"

Two ranked lists, on incomparable score scales (cosine ~0–1, BM25 unbounded).

### Step 7 — Fusion: two rankings → one (`retrieval.py`)

`reciprocal_rank_fusion()` ignores the raw scores and uses only *positions*: an item at rank r
earns `1/(60+r)` from each list, summed. A chunk ranked highly by BOTH searches accumulates two
big votes and rises to the top. The fused list is cut to the top 5 (`TOP_K`). (With
`--rerank`, a cross-encoder re-scores those candidates first — `rerank.py`.)

### Step 8 — The prompt is assembled (`generation.py`)

The 5 winning chunks are formatted into a numbered context block and combined with the
grounding system prompt:

```
system: Answer ONLY using the numbered context passages... say you don't know... cite [1], [2].
user:   Context passages:
        [1] (doc: unit-1-my-pet) This is my pet, Chittu...
        [2] (doc: unit-1-my-pet) ...
        Question: Who is Valli's pet?
```

Run with `LLM_DEBUG=1` to see this exact block on stderr — it's the whole "AI magic" laid bare:
the model is just answering a well-constructed prompt.

### Step 9 — Generation + citation extraction (`generation.py`)

`gemini-2.5-flash` (temperature 0) writes the answer, marking claims with `[n]`.
`extract_citations()` then parses those markers and attaches only the passages the model
*actually cited* as `Citation` objects — provenance you can check. If the answer isn't in the
contexts, the system prompt's escape hatch means the honest output is "I don't know."

> Final data shape: `Answer(question, text, citations=[...], contexts=[...])` — `models.py`.
> The CLI prints `text` to stdout and the sources to stderr; the API returns the whole thing
> as JSON.

---

## The same journey through the API (`api.py`)

`POST /ask {"question": "..."}` runs steps 5–9 with two production layers in front:

1. **Rate limiter** (`ratelimit.py`) — per-client sliding window; over the limit → HTTP 429.
2. **Semantic cache** (`cache.py`) — the question is embedded and compared to previously
   answered ones; a near-duplicate (cosine ≥ 0.97) returns the stored answer instantly with
   `"cached": true`. Cheapest checks run first; the expensive retrieve-and-generate path only
   runs when both let the request through.

`POST /ingest` runs steps 2–4 for a document sent in the request body. `GET /eval` runs the
golden-set comparison. `GET /metrics` shows the counters.

---

## Where each knob lives (recap)

| Step | Knob | Where |
|---|---|---|
| Extraction | supported formats | `corpus.py` (`_SUFFIXES`) |
| Chunking | `CHUNK_SIZE`, `CHUNK_OVERLAP` | `.env` → `config.py` |
| Embedding | model, `GEMINI_EMBEDDING_DIM` | `.env` → `config.py` |
| Storage | `VECTOR_STORE`, `PINECONE_INDEX` | `.env` → `config.py` |
| Retrieval | `TOP_K`, `CANDIDATE_K`, `RRF_K` | `.env` → `config.py` |
| Generation | `GEMINI_MODEL`, `MAX_TOKENS` | `.env` → `config.py` |
| Caching | similarity threshold | `cache.py` |
| Rate limit | max/window | `api.py` |

---

## Design decisions (the "why")

Why the pipeline is shaped the way it is. Read alongside the source.

### The pipeline

```
ingest:  documents ─▶ chunk ─▶ embed ─▶ [vector store] + [BM25 index]
ask:     question  ─▶ retrieve (dense + sparse → RRF) ─▶ rerank? ─▶ generate (cited)
```

Each stage is an interface (`Embedder`, `VectorStore`, `Reranker`, `Answerer`). The pipeline depends
only on those abstractions, so backends swap without touching the orchestration — and the whole thing
runs offline for tests.

### Key decisions

#### 1. Hybrid retrieval, fused with RRF
Dense embeddings capture meaning but miss exact terms (names, codes, rare keywords); BM25 nails exact
terms but misses paraphrase. We run both and fuse with **Reciprocal Rank Fusion**: each item scores
`1 / (rrf_k + rank)` summed across lists. Because RRF uses **rank, not raw score**, it fuses systems
on totally different scales (cosine ∈ [-1,1] vs unbounded BM25) with **no normalization** — the common
failure mode of score-weighted fusion. The eval harness shows hybrid ≥ either alone.

**Alternative considered:** weighted score combination. Rejected — requires per-corpus score
normalization and tuning; RRF is parameter-light and robust.

#### 2. Two-stage retrieve-then-rerank
First-stage retrieval optimizes recall cheaply over the whole corpus. A **cross-encoder reranker**
then re-scores only the top candidates by reading query + passage *together* — far more accurate, far
too expensive to run corpus-wide. Default is a no-op reranker; the cross-encoder is an optional extra
so the core stays light. This mirrors how production RAG actually trades cost for precision.

#### 3. Swappable stores: in-memory for experiments, Pinecone for persistence
`InMemoryVectorStore` (numpy cosine) needs no infrastructure — instant, free, gone at process
exit. `PineconeVectorStore` is the persistent path: a managed serverless vector database, upserts
idempotent by chunk id, chunk text carried in record metadata so results need no second lookup.
Same `VectorStore` protocol, so the retriever is identical. (Honest scope note: BM25 is in-memory
in both paths — with Pinecone, `rag ask` rebuilds the sparse index locally from `data/` while
dense search hits the persistent index; Pinecone's sparse vectors could make hybrid fully
persistent later. The fusion logic would be unchanged.)

#### 4. One real path, loud failures, stubs at the seams
Gemini is required — one key covers semantic embeddings AND synthesis; a missing key raises a
`ConfigError` naming exactly what to set, instead of silently degrading. Offline testability is
preserved where it belongs: `tests/conftest.py` injects stub implementations of the same
`Embedder`/`Answerer` protocols, so the full pipeline — chunking, retrieval, fusion, evaluation —
still runs in CI with no keys or network.

#### 5. Grounded, cited generation
The generator is instructed to answer **only** from numbered contexts, cite them, or say it doesn't
know. Passing retrieved chunks as numbered context + attaching them as citations is what makes the
answer attributable instead of a confident hallucination. Faithfulness *scoring* (LLM-as-judge) comes
in Milestone 4.

#### 6. Evaluation is a first-class feature, not an afterthought
`evaluation.py` turns "the RAG feels good" into **recall@k** and **MRR** over a labelled golden set,
and `compare_modes` runs dense / sparse / hybrid / +rerank head-to-head. Relevance is keyed by source
**document id**, which is robust to chunking choices. This is the "ship gate" muscle — built here,
deepened in Milestone 4.

### Chunking notes
Recursive, structure-aware splitting (paragraph → line → sentence → word) with overlap. Overlap
prefixes each chunk with the previous chunk's tail so a fact split across a boundary survives. Chunk
size is the highest-leverage knob: too big bloats context and hurts precision; too small loses the
context a passage needs. Tune it against the eval harness, not by feel.

### Trade-offs left open
- Persistent sparse retrieval (currently in-memory BM25 → Pinecone sparse vectors or Postgres FTS).
- Hosted reranker (Voyage/Cohere) vs the local cross-encoder.
- Query-side techniques: multi-query expansion (Task 6 in `tasks/`), HyDE, query rewriting.
- Answer-quality / faithfulness evals — Milestone 4.

---

## Serving & deployment

How this service goes from "runs locally" to "serving traffic," and the production-serving features
built into the API.

### Serving features (in the API)

- **Semantic response cache** (`cache.py`) — embeds each query and serves a cached answer when a
  *similar* query is within a similarity threshold, so paraphrased repeats skip retrieval +
  generation. Cuts cost and tail latency. The response includes `"cached": true|false`.
- **Rate limiting** (`ratelimit.py`) — a per-client sliding window caps requests (default 60/min),
  returning HTTP 429 over the limit. Caps cost and abuse on a public endpoint.
- **Metrics** — `GET /metrics` exposes request counts, cache hit rate, cache size, and the rate-limit
  config for monitoring/alerting.
- **Health check** — `GET /health` for load-balancer probes.

> In production, the cache and rate limiter would be **Redis-backed** so they're shared across
> replicas; the in-process versions here have identical semantics for a single instance.

### Local (Docker)

```bash
docker build -t rag-knowledge-assistant .
docker run -p 8000:8000 --env-file .env rag-knowledge-assistant
curl localhost:8000/health
```

### Cloud (Render — no GPU needed)

1. Push to GitHub (done).
2. Create a Render account, **New → Blueprint**, point it at this repo. Render reads
   [`render.yaml`](../render.yaml) and builds the Docker web service.
3. Set secrets in the dashboard: `GEMINI_API_KEY` (embeddings **and** answers) and
   `PINECONE_API_KEY` + `PINECONE_INDEX` with `VECTOR_STORE=pinecone` for persistent vectors —
   the natural cloud setup, since Pinecone is already managed. `VECTOR_STORE=memory` also works
   for a stateless demo (the corpus re-embeds on each boot).

Fly.io / Railway are equivalent: they build the same `Dockerfile`; the start command is already the
`uvicorn` `CMD`.

### CI/CD

- [`ci.yml`](../.github/workflows/ci.yml) gates every push/PR on lint + types + tests.
- [`deploy.yml`](../.github/workflows/deploy.yml) triggers a Render deploy **after CI passes on
  main** — but only if the `RENDER_DEPLOY_HOOK_URL` secret is set, so it's safe to commit before any
  cloud account exists. Add the secret (Render → Settings → Deploy Hook) to turn it on.

### Scaling & cost notes
- **Caching** is the highest-leverage cost lever for repeated queries; semantic caching extends it to
  paraphrases.
- **Model tiering** — cheaper embedding/generation models for high volume; validate each switch
  against the eval harness so quality doesn't silently regress.
- **Async + concurrency** — FastAPI is async; run multiple workers (`uvicorn --workers N`) behind a
  load balancer; the vector store is already shared state (Pinecone).
- **Inference at scale** (conceptual) — self-hosted open models use **vLLM** (continuous batching +
  KV-cache) on GPUs; **Kubernetes** for orchestration. This service stays CPU/API-friendly by design.
- **Observability** — wire request traces + eval gates from
  [`llm-eval-kit`](https://github.com/ArunRyzen/llm-eval-kit) (Milestone 4).
