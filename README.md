<div align="center">

# 📚 rag-knowledge-assistant

**Production-shaped Retrieval-Augmented Generation — with the eval harness that proves it works.**

Hybrid retrieval (dense + BM25, fused with RRF) · optional cross-encoder reranking · cited answers ·
recall@k / MRR evaluation. Runs zero-infra in memory, or on Postgres + pgvector.

</div>

---

## ⚡ Quick Start

```bash
git clone https://github.com/ArunRyzen/rag-knowledge-assistant.git && cd rag-knowledge-assistant
uv sync --extra dev          # installs everything — no API keys needed
uv run rag eval              # compare dense vs sparse vs hybrid retrieval
```
*Runs fully offline.* Add a single `GEMINI_API_KEY` to `.env` for live semantic embeddings **and** real answers — see [Live mode](#live-mode-one-gemini-key).

---

## Problem

RAG is the most-deployed pattern in production AI — and the easiest to do badly. Naive "embed +
nearest-neighbour" misses exact keywords, buries the right passage, and gives no way to tell whether
a change helped or hurt. This project is RAG done the way teams actually ship it: **hybrid retrieval**
(semantic *and* lexical), a **rerank** stage, **grounded, cited answers**, and — the centerpiece — a
**retrieval evaluation harness** so quality is a number, not a vibe.

## What it does

```bash
rag ask "How are two ranked lists combined into one?"
# → Reciprocal Rank Fusion combines lists by rank... [1]
#   Sources: [1] rrf (score=0.033)

rag eval        # compare retrieval strategies on a labelled golden set
```
```
dense            recall@5=0.80  MRR=0.73  (n=5)
sparse           recall@5=1.00  MRR=0.90  (n=5)
hybrid           recall@5=1.00  MRR=0.95  (n=5)
hybrid+rerank    recall@5=1.00  MRR=1.00  (n=5)
```
*(Illustrative numbers from the bundled sample corpus — run it yourself.)*

## Architecture

```mermaid
flowchart LR
    D[Documents] --> C[Chunker]
    C --> E[Embedder]
    E --> VS[(Vector store<br/>memory / pgvector)]
    C --> BM[BM25 index]
    Q[Question] --> R{Retriever}
    VS -->|dense| R
    BM -->|sparse| R
    R -->|RRF fusion| RR[Reranker<br/>optional]
    RR --> G[Generator<br/>cited answer]
    R -. evaluated by .-> EV[Eval harness<br/>recall@k · MRR]
```

Every stage sits behind a small interface (`Embedder`, `VectorStore`, `Reranker`, `Answerer`), so
backends swap freely and the whole thing is testable offline. Full reasoning in
[`docs/architecture.md`](docs/architecture.md).

## Tech stack

`Python 3.12` · `Pydantic v2` · `NumPy` · `Gemini` (embeddings + generation) · `OpenAI`
(embeddings + generation) · `Anthropic` (generation) · `pgvector` / Postgres (optional) ·
`FastAPI` · `Typer` · `uv` · `ruff` · `mypy` · `pytest` · `Docker` · `GitHub Actions`

## Setup

```bash
git clone https://github.com/ArunRyzen/rag-knowledge-assistant.git
cd rag-knowledge-assistant
uv sync --extra dev
cp .env.example .env     # optional — works with no keys using the offline embedder + fake answerer
```

**Runs with zero configuration.** With no API keys it uses a deterministic hashing embedder and a
fake answerer — enough to exercise chunking, hybrid retrieval, and the eval harness end to end.

## Live mode (one Gemini key)

The easiest way to go live is a **Gemini API key** — one free key covers both halves of the
pipeline. Get one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey), then:

```bash
cp .env.example .env
# in .env:
GEMINI_API_KEY=your-key-here
```

That's it. With `GEMINI_API_KEY` set the factory automatically switches:

| Stage | Offline default | With `GEMINI_API_KEY` |
|---|---|---|
| Embeddings | Hashing embedder (lexical) | `gemini-embedding-001` (semantic) |
| Answers | Fake answerer (canned, cited) | `gemini-2.5-flash` (real, cited) |

Models and embedding size are configurable via `GEMINI_MODEL`, `GEMINI_EMBEDDING_MODEL`, and
`GEMINI_EMBEDDING_DIM` (768 by default; the model also supports 1536/3072).

Alternative providers, if you have those keys instead: `OPENAI_API_KEY` enables OpenAI
embeddings (and OpenAI answers with `GENERATION_PROVIDER=openai`); `ANTHROPIC_API_KEY` +
`GENERATION_PROVIDER=anthropic` enables Anthropic answers.

## Usage

**CLI**
```bash
rag ask "Which index makes vector search fast?"          # ingest sample corpus + answer
rag ask "..." --data ./my_docs --mode hybrid --rerank    # your own .md/.txt folder
rag eval                                                  # dense vs sparse vs hybrid vs +rerank
```

**API**
```bash
uv run uvicorn rag_assistant.api:app --reload
# POST /ask {"question": "...", "mode": "hybrid"}   POST /ingest {"doc_id","text"}
# GET  /eval     GET /health
```

**Library**
```python
from rag_assistant.config import load_settings
from rag_assistant.factory import build_pipeline

pipe = build_pipeline(load_settings())
pipe.ingest("notes", open("notes.md").read())
print(pipe.ask("...", mode="hybrid", rerank=True).text)
```

## Peek behind the curtain (`LLM_DEBUG`)

Curious what actually gets sent to (and comes back from) each model? Set `LLM_DEBUG=1` and every
embedder and answerer call — **including the offline fakes, so no API key needed** — prints a
plain request/response block to stderr (API keys are never logged). Two ways to turn it on:

```powershell
# 1) Per-session environment variable:
$env:LLM_DEBUG = "1"; uv run rag ask "Which index makes vector search fast?"
Remove-Item Env:LLM_DEBUG   # turn it off again
```

```ini
# 2) Or persistently, via a line in your project's .env file:
LLM_DEBUG=1
```

If both are present, the environment variable wins — handy for a one-off
`$env:LLM_DEBUG = "0"` override without editing `.env`.

## How it works (the parts interviewers ask about)

1. **Chunking** — recursive, structure-aware splitting with overlap, so a fact split across a
   boundary still lives in one chunk.
2. **Hybrid retrieval** — dense (cosine over embeddings) **and** sparse (BM25 lexical), fused with
   **Reciprocal Rank Fusion**. RRF combines by *rank*, so it doesn't matter that cosine and BM25 are
   on different scales. Hybrid beats either alone.
3. **Reranking (optional)** — a cross-encoder re-scores the top candidates by reading query + passage
   *together*. Expensive, so: retrieve broadly, rerank precisely.
4. **Grounded generation** — the model answers **only** from numbered contexts and cites them, or says
   it doesn't know.
5. **Evaluation** — recall@k and MRR over a labelled golden set, comparing every retrieval mode.

## Production backend (pgvector)

```bash
docker compose up -d                 # Postgres + pgvector
# then in .env: VECTOR_STORE=pgvector  DATABASE_URL=postgresql://rag:rag@localhost:5432/rag
uv sync --extra pgvector
```
The in-memory store is the default and is great to a few hundred thousand chunks; pgvector adds
persistence and HNSW-indexed search for production. The retriever code is identical either way.

## Testing

```bash
uv run ruff check . && uv run mypy . && uv run pytest
```
The suite runs the full pipeline offline — no keys, no database, no network — because every backend
implements a swappable protocol. CI runs lint + type-check + tests on every push.

## Serving & deployment

Production-serving features are built into the API (full guide:
[`docs/deployment.md`](docs/deployment.md)):
- **Semantic response cache** — paraphrased repeats skip retrieval + generation (`"cached": true`).
- **Rate limiting** — per-client sliding window (HTTP 429 over the limit).
- **`GET /metrics`** — request count, cache hit rate, cache size, rate-limit config.

```bash
docker build -t rag-knowledge-assistant .
docker run -p 8000:8000 --env-file .env rag-knowledge-assistant
```
Cloud: a [`render.yaml`](render.yaml) blueprint + a CI-gated [deploy workflow](.github/workflows/deploy.yml)
deploy the Docker service (no GPU needed). See [`docs/deployment.md`](docs/deployment.md).

## Future improvements
- Postgres full-text (`tsvector`) for sparse retrieval in the pgvector path (BM25 is in-memory today).
- Hosted reranker (Voyage/Cohere) as an API-based alternative to the local cross-encoder.
- Faithfulness / answer-quality evals (LLM-as-judge) — picked up in Milestone 4.
- Streaming answers and PDF/OCR ingestion.

## Learn more
- [`docs/code-walkthrough.md`](docs/code-walkthrough.md) — **new to RAG? start here** — a
  plain-English, file-by-file tour with a "where to find X" cheat sheet for every tunable knob
- [`docs/architecture.md`](docs/architecture.md) — design decisions & trade-offs
- [`docs/interview-questions.md`](docs/interview-questions.md) — RAG Q&A this project answers
- [`docs/lessons-learned.md`](docs/lessons-learned.md)

## License

[MIT](LICENSE) · Part of my [AI_Engineer](https://github.com/ArunRyzen/AI_Engineer) portfolio (Milestone 2).
