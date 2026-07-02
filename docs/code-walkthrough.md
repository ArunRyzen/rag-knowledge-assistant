# Code walkthrough — a plain-English tour of every file

This guide assumes you can read Python but have never built a RAG system. It walks the code in
an order that builds understanding step by step, explains the most important snippets line by
line, and ends each section with "why it's built this way".

All paths below are relative to the repo root; the source lives in `src/rag_assistant/`.

---

## Where to find X (the cheat sheet)

The knobs you'll want to change for the exercises, and exactly where they live:

| I want to change… | File | Look for | Default |
|---|---|---|---|
| **Chunk size** (how big each text piece is) | `src/rag_assistant/config.py` | `chunk_size: int = Field(default=800)` | 800 characters |
| Chunk overlap | `src/rag_assistant/config.py` | `chunk_overlap: int = Field(default=120)` | 120 characters |
| **Golden eval set** (the questions `rag eval` scores) | `src/rag_assistant/sample_data.py` | the `GOLDEN` list | 5 questions |
| **Hybrid fusion / RRF** (how dense + sparse results merge) | `src/rag_assistant/retrieval.py` | `reciprocal_rank_fusion(...)`; the `rrf_k` constant is `rrf_k: int = Field(default=60)` in `config.py` | rrf_k = 60 |
| **Semantic-cache similarity threshold** | `src/rag_assistant/cache.py` | `threshold: float = 0.97` in `SemanticCache.__init__` | 0.97 |
| **Rate limit** (requests per window) | `src/rag_assistant/api.py` | `RATE_LIMIT_MAX = 60` and `RATE_LIMIT_WINDOW_S = 60.0` (top of file) | 60 requests / 60 s |
| How many chunks the answer is based on | `src/rag_assistant/config.py` | `top_k` / `candidate_k` | 5 / 20 |
| Which models are used (Gemini etc.) | `src/rag_assistant/config.py` + `.env` | `gemini_model`, `gemini_embedding_model` | gemini-2.5-flash / gemini-embedding-001 |
| BM25 tuning constants | `src/rag_assistant/sparse.py` | `_K1 = 1.5`, `_B = 0.75` | classic defaults |

Anything in `config.py` can also be changed without touching code: set the upper-cased env var
(e.g. `CHUNK_SIZE=400`) in your `.env` file. The cache threshold and rate limit are **not** in
`config.py` — edit them where the table says.

### The exercise knobs, up close

**Chunk size** — `src/rag_assistant/config.py`:

```python
chunk_size: int = Field(default=800)   # target characters per chunk
chunk_overlap: int = Field(default=120)  # characters shared between neighbouring chunks
```

Halve it (400) → smaller, sharper chunks: retrieval gets more precise but each hit carries less
context. Quadruple it (3200) → whole documents in one chunk: nothing gets fragmented, but every
hit drags in lots of irrelevant text. Run `uv run rag eval` after each change and watch
recall@5 / MRR move.

**Golden eval set** — `src/rag_assistant/sample_data.py`, the `GOLDEN` list. One dict per test
question; `relevant_doc_ids` names the document(s) that contain the answer:

```python
GOLDEN: list[dict[str, object]] = [
    {"question": "How do I store vector embeddings in Postgres?", "relevant_doc_ids": ["pgvector"]},
    ...
]
```

Doc ids are the keys of `SAMPLE_DOCS` in the same file (or the filename without extension when
you ingest your own folder with `--data`: `notes.md` → `"notes"`). To add your own 10 questions,
append 10 more dicts in exactly this shape.

**RRF / hybrid fusion** — `src/rag_assistant/retrieval.py`, function `reciprocal_rank_fusion`
(explained line by line below). The constant it uses comes from `rrf_k` in `config.py`
(env `RRF_K`, default 60).

**Semantic-cache threshold** — `src/rag_assistant/cache.py`:

```python
def __init__(self, embedder: Embedder, *, threshold: float = 0.97, max_size: int = 512) -> None:
```

Lower `0.97` to, say, `0.85` and rewordings of earlier questions start hitting the cache
(`"cached": true` in the API response). Too low and *different* questions get someone else's
answer — that's the trade-off the exercise wants you to feel.

**Rate limit** — `src/rag_assistant/api.py`, right at the top:

```python
RATE_LIMIT_MAX = 60        # requests allowed per window, per client
RATE_LIMIT_WINDOW_S = 60.0  # window length in seconds
```

Set `RATE_LIMIT_MAX = 3`, restart the API, and the fourth `/ask` within a minute returns
HTTP 429.

---

## The 30-second big picture

A RAG ("Retrieval-Augmented Generation") system answers questions about *your* documents by:

1. **Ingesting**: cutting documents into chunks, turning each chunk into a vector
   (an "embedding"), and indexing the chunks two ways — by meaning and by exact words.
2. **Retrieving**: given a question, finding the handful of chunks most likely to contain
   the answer.
3. **Generating**: handing those chunks to an LLM with strict instructions: *answer only from
   these passages and cite them*.

```
documents ──chunking──> chunks ──embedding──> vector store ─┐
                          └──────words──────> BM25 index ───┤
                                                            ├─ retriever (fuses both) ─> LLM ─> cited answer
question ───────────────────────────────────────────────────┘
```

Everything else in the repo is either plumbing around that pipeline (CLI, API, config) or
production hardening (cache, rate limiter, metrics, eval harness).

---

## Suggested reading order

| # | File | What you'll learn |
|---|------|-------------------|
| 1 | `models.py` | The 4 data shapes everything passes around |
| 2 | `config.py` | Every knob, and how `.env` feeds it |
| 3 | `chunking.py` | How documents become chunks |
| 4 | `embeddings.py` | How text becomes vectors (offline, Gemini, OpenAI) |
| 5 | `vectorstore.py` | Where vectors live; cosine search |
| 6 | `sparse.py` | BM25 — exact-word search |
| 7 | `retrieval.py` | Hybrid search + RRF fusion (the heart) |
| 8 | `rerank.py` | The optional precision stage |
| 9 | `generation.py` | Turning chunks into a cited answer |
| 10 | `pipeline.py` | Gluing 3–9 together |
| 11 | `factory.py` | How keys in `.env` pick the implementations |
| 12 | `sample_data.py` + `evaluation.py` | The golden set and recall@k / MRR |
| 13 | `cache.py` + `ratelimit.py` | Production serving concerns |
| 14 | `api.py` + `cli.py` + `corpus.py` | The two front doors |
| 15 | `errors.py` | The exception family |

---

## 1. `models.py` — the vocabulary

Four tiny Pydantic models; every other file talks in these terms:

- **`Chunk`** — one piece of a document: `id` (like `"pgvector::0"`), `doc_id` (which document
  it came from), `text`, and its `index` within the document. The `doc_id` is what lets an
  answer cite its source and what the eval harness checks against.
- **`RetrievedChunk`** — a `Chunk` plus the `score` that ranked it and which retriever
  (`"dense"`, `"sparse"`, `"hybrid"`, `"reranked"`) produced it.
- **`Citation`** — a pointer from an answer back to a chunk (`chunk_id`, `doc_id`, a short quote).
- **`Answer`** — the final product: the question, the generated text, citations, and the
  full contexts used.

*Why Pydantic instead of plain dicts?* Typos in field names become instant errors instead of
silent bugs, and mypy can check every hand-off between modules.

## 2. `config.py` — every knob in one place

One `Settings` class. Each field reads from an environment variable of the same name
(upper-cased), falling back to `.env`, then to the default in code:

```python
chunk_size: int = Field(default=800)   # ← env var CHUNK_SIZE beats this default
```

Notable fields: `gemini_api_key` (set this and BOTH embeddings and answers go live via Gemini),
`chunk_size` / `chunk_overlap`, `top_k` / `candidate_k`, `rrf_k`, `use_reranker`,
`vector_store` (`memory` or `pgvector`).

*Why one class?* So there is exactly one place to answer "what can I tune?" — and tests can
construct `Settings(...)` directly with whatever they need.

## 3. `chunking.py` — documents → chunks

The public function is `chunk_document(doc_id, text, size, overlap)` — `size` is the
`chunk_size` from config. The interesting part is *where* it cuts. `_split_recursive` tries
separators in order of niceness:

```python
_SEPARATORS = ["\n\n", "\n", ". ", " "]   # paragraph → line → sentence → word
```

It packs paragraphs greedily into chunks of at most `size` characters; if a single paragraph is
itself too big, it recurses with the next finer separator (lines, then sentences, then words),
and only hard-cuts mid-word as a last resort.

Then `chunk_document` adds **overlap**: each chunk (after the first) is prefixed with the last
`overlap` characters of the previous one. Line by line:

```python
prev_tail = ""
for i, body in enumerate(raw):                                   # raw = the split pieces
    content = (prev_tail + " " + body).strip() if prev_tail else body  # glue on the previous tail
    chunks.append(Chunk(id=f"{doc_id}::{i}", doc_id=doc_id, text=content, index=i))
    prev_tail = body[-overlap:] if overlap > 0 else ""            # remember MY tail for the next one
```

*Why overlap?* If a key sentence straddles a chunk boundary, the overlap guarantees it appears
whole in at least one chunk — otherwise it would be unfindable.

## 4. `embeddings.py` — text → vectors

Three implementations of one 2-method interface (`Embedder`: a `dim` and an
`embed(texts) -> vectors`):

- **`HashingEmbedder`** (the offline default). Hashes each word into one of `dim` buckets and
  counts. Texts sharing words get similar vectors. Not semantic — "car" and "automobile" look
  unrelated — but deterministic, free, and honest enough to exercise the whole pipeline.
- **`GeminiEmbedder`** (the live path with `GEMINI_API_KEY`). Calls `gemini-embedding-001` via
  the `google-genai` SDK, asking for `dim`-sized vectors (`output_dimensionality`), and
  L2-normalizes them because Gemini only pre-normalizes at the full 3072 dimensions:

  ```python
  response = self._client.models.embed_content(
      model=self._model,            # "gemini-embedding-001"
      contents=texts,               # a whole batch in one API call
      config=types.EmbedContentConfig(output_dimensionality=self.dim),
  )
  return [_l2_normalize(list(item.values or [])) for item in response.embeddings or []]
  ```
- **`OpenAIEmbedder`** — same idea via OpenAI, if that's the key you have.

*Why normalize to length 1?* Then "similarity" is just a dot product (cosine similarity), which
is what the vector store computes.

## 5. `vectorstore.py` — where vectors live

`InMemoryVectorStore` keeps all chunk vectors in one NumPy matrix. Search is two lines: dot the
query vector against the matrix (that IS cosine similarity, since everything is normalized),
then take the k largest scores. Fine up to a few hundred thousand chunks.

`PgVectorStore` is the production twin: Postgres + the pgvector extension, using the `<=>`
cosine-distance operator. Both satisfy the same `VectorStore` protocol, so nothing downstream
knows or cares which one is running.

## 6. `sparse.py` — BM25, the exact-word search

Dense vectors are great at meaning but can miss exact tokens — error codes, names, rare
keywords. BM25 is the classic fix: score a chunk by which query words it contains, where

- rare words count more (`_idf`: a word in 1 chunk out of 1000 is a strong clue; "the" is noise),
- repeated words give diminishing returns (`_K1 = 1.5`),
- long chunks get gently penalized (`_B = 0.75`).

The whole thing is ~70 lines of pure Python — worth reading once to demystify "keyword search".

## 7. `retrieval.py` — hybrid search + RRF (the heart of the repo)

`Retriever.retrieve(query, mode=...)` supports `dense`, `sparse`, and `hybrid`. Hybrid runs
*both* retrievers and merges their rankings with **Reciprocal Rank Fusion**:

```python
fused: dict[str, float] = {}
for ranking in rankings:                      # e.g. [dense_results, sparse_results]
    for rank, item in enumerate(ranking, start=1):   # rank 1 = that list's best
        cid = item.chunk.id
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank)   # this list's "vote"
        best_chunk.setdefault(cid, item)
ordered = sorted(fused.items(), key=lambda kv: -kv[1])            # biggest total first
```

Line by line: each list votes for its items; a rank-1 item gets `1/(60+1)`, rank-2 gets
`1/(60+2)`, and so on. A chunk that appears high in *both* lists collects both votes and beats a
chunk that only one retriever liked. Because RRF uses only ranks — never raw scores — it doesn't
matter that cosine similarity (0–1) and BM25 (unbounded) are on completely different scales.

`rrf_k` (config `RRF_K`, default 60) is the only "weight" hybrid has: smaller → the top ranks
dominate; larger → rank 1 and rank 10 are treated more equally.

One more idea: `retrieve` over-fetches `candidate_k` (20) candidates and only returns `top_k`
(5) — fusion and reranking need spare material to work with.

## 8. `rerank.py` — the optional precision stage

Retrieval compares two *separately computed* vectors — fast but approximate. A cross-encoder
reads the query and a chunk *together* and outputs a relevance score — far more accurate, far
more expensive. So the production pattern is: retrieve 20 cheaply, rerank to 5 precisely.
`NoopReranker` (just truncate) is the default; `CrossEncoderReranker` needs the `rerank` extra.

## 9. `generation.py` — chunks → cited answer

The system prompt is the safety mechanism — read `_SYSTEM` in the file: answer **only** from the
numbered passages, say "I don't know" otherwise, cite passage numbers. The contexts are
numbered `[1] (doc: ...) text...` so the model's citations map back to real chunks.

`FakeAnswerer` returns a canned-but-grounded answer with real citations (that's what all the
offline tests use). `LLMAnswerer` does the real call — for Gemini:

```python
response = gclient.models.generate_content(
    model=self._model,                       # "gemini-2.5-flash"
    contents=prompt,                         # numbered contexts + the question
    config=types.GenerateContentConfig(
        system_instruction=system,           # the grounding rules above
        max_output_tokens=self._max_tokens,
        temperature=0,                       # no creativity wanted — just grounded answers
    ),
)
```

with equivalent branches for Anthropic and OpenAI. All providers get an identical prompt, so
switching keys never changes the grounding behavior.

## 10. `pipeline.py` — the glue

`RAGPipeline` owns two verbs:

- `ingest(doc_id, text)`: chunk → embed → add to the vector store **and** the BM25 index
  (every chunk is indexed twice — that dual indexing is what makes hybrid possible).
- `ask(question)`: retrieve → answer.

It depends only on the interfaces (`Embedder`, `VectorStore`, `Answerer`), never on a concrete
implementation — which is exactly why the test suite can run the *entire* pipeline offline.

## 11. `factory.py` — how `.env` picks the implementations

`build_embedder` / `build_answerer` / `build_vector_store` inspect `Settings` and choose:

- `GEMINI_API_KEY` set → `GeminiEmbedder` + Gemini `LLMAnswerer` (one key, whole live path).
- Only `OPENAI_API_KEY` → OpenAI embeddings (and OpenAI answers if `GENERATION_PROVIDER=openai`).
- No keys → `HashingEmbedder` + `FakeAnswerer`: everything still runs, offline, for free.

*Why a factory?* All "which implementation?" if-statements live in this one file; the CLI and
API just say "give me a pipeline".

## 12. `sample_data.py` + `evaluation.py` — proving it works

`sample_data.py` bundles 4 tiny documents (`SAMPLE_DOCS`) and **the golden eval set**
(`GOLDEN`) — see the cheat-sheet section above for the exact format and how to add questions.

`evaluation.py` turns retrieval quality into numbers. For each golden question it retrieves
top-k and checks where the first chunk from a relevant document landed:

- **recall@k** — fraction of questions where a relevant doc appeared *anywhere* in the top-k.
- **MRR** — mean of `1/rank` of the first relevant hit: 1.0 means "always ranked first";
  0.5 means "typically second". It rewards putting the right doc *on top*, not just somewhere.

`compare_modes` runs dense vs sparse vs hybrid vs hybrid+rerank so you can see, in numbers, why
hybrid earns its complexity. This is what `rag eval` and `GET /eval` call.

## 13. `cache.py` + `ratelimit.py` — production hardening

**`SemanticCache`** — an exact-match cache would miss "capital of France?" vs "what's France's
capital?". This cache embeds each query and serves a stored answer when a new query's cosine
similarity to a cached one is ≥ the **threshold (0.97, set in `SemanticCache.__init__`)**.
`get` is a linear scan for the most similar cached entry:

```python
for cached_embedding, answer in self._entries:
    sim = _cosine(embedding, cached_embedding)   # 0..1, how alike the two questions are
    if sim > best_sim:
        best_sim, best_answer = sim, answer
if best_answer is not None and best_sim >= self._threshold:   # close enough → reuse
    self.stats.hits += 1
    return best_answer
```

**`RateLimiter`** — `allow(key)` keeps a list of recent request timestamps per client, drops the
ones older than the window, and refuses when the count reaches the max. The actual numbers (60
per 60 s) are passed in from the top of `api.py`. Rejected requests are deliberately *not*
recorded, so hammering a blocked endpoint doesn't push your reset time further away.

## 14. `api.py`, `cli.py`, `corpus.py` — the front doors

**`api.py`** (FastAPI): `/ingest`, `/ask`, `/eval`, `/health`, `/metrics`. The `/ask` handler
shows the production order of operations — cheapest checks first:

```python
if not _limiter.allow(client):        # 1. rate limit (free to check)
    raise HTTPException(429, ...)
cached = _cache().get(key)            # 2. semantic cache (one embedding call)
if cached is not None:
    return {**cached, "cached": True}
answer = _pipeline().ask(...)         # 3. full retrieve + generate (the expensive bit)
```

`GET /metrics` exposes request counts, cache hit-rate/size, and the rate-limit config — enough
to watch your cache-threshold experiments work.

**`cli.py`** (Typer): `rag ask "..."` and `rag eval`, each ingesting the corpus first (the
in-memory store starts empty every process). **`corpus.py`** loads `.md`/`.txt` files from a
`--data` folder (doc id = filename without extension) or falls back to the bundled samples.

## 15. `errors.py` — the exception family

A tiny hierarchy (`RAGError` → `ConfigError`, `IngestionError`, `RetrievalError`,
`GenerationError`) so callers can tell a bad config from a failed API call.

---

## Where the tests live

| Test file | Covers |
|---|---|
| `tests/test_chunking.py` | splitting + overlap behavior |
| `tests/test_retrieval.py` | dense/sparse/hybrid + RRF |
| `tests/test_sparse.py` | BM25 scoring |
| `tests/test_pipeline.py` | end-to-end ingest → ask |
| `tests/test_evaluation.py` | recall@k / MRR math |
| `tests/test_cache.py` | semantic cache hits/misses |
| `tests/test_ratelimit.py` | window behavior |
| `tests/test_api.py` | endpoints, 429s, cached flag |
| `tests/test_gemini.py` | Gemini embedder/answerer with mocked clients (no network) |

Run everything (offline, no keys needed):

```bash
uv run ruff check . && uv run mypy . && uv run pytest
```
