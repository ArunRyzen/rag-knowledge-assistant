# End-to-end: from a PDF on disk to a cited answer

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
> `doc_id` is the filename without extension (`query-transformation.pdf` → `"query-transformation"`).
> Everything downstream only ever sees plain text — this is why adding a new format (Word,
> HTML, OCR'd scans) would only ever touch this one file.

### Step 2 — Chunking: text → chunks (`chunking.py`)

`chunk_document()` splits each document into overlapping pieces of ~`CHUNK_SIZE` (800)
characters, cutting at the most natural boundary available — paragraph, then line, then
sentence, then word. Each chunk gets a **stable id** `"<doc_id>::<index>"` (e.g. `"bm25::2"`)
and carries its `doc_id` for provenance. The last 120 characters (`CHUNK_OVERLAP`) of each chunk
are repeated at the start of the next, so a fact cut at a boundary survives whole in at least
one chunk.

> Data shape now: `Chunk(id="bm25::2", doc_id="bm25", text="...", index=2)` — see `models.py`.

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
  "id": "bm25::2",                     # the stable chunk id — upserting again OVERWRITES
  "values": [0.0123, -0.0456, ...],    # the 768-dim embedding
  "metadata": {"doc_id": "bm25", "index": 2, "text": "..."}   # the chunk itself, carried along
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

**Write path complete.** 9 documents → 40-odd chunks → 40-odd vectors in Pinecone + a BM25
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
        [1] (doc: bm25) BM25 is a classic lexical ranking function...
        [2] (doc: hybrid-rrf) ...
        Question: What does BM25 reward?
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
