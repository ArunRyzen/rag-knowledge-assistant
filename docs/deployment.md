# Deployment & Serving

How this service goes from "runs locally" to "serving traffic," and the production-serving features
built into the API.

## Serving features (in the API)

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

## Local (Docker)

```bash
docker build -t rag-knowledge-assistant .
docker run -p 8000:8000 --env-file .env rag-knowledge-assistant
curl localhost:8000/health
```

## Cloud (Render — no GPU needed)

1. Push to GitHub (done).
2. Create a Render account, **New → Blueprint**, point it at this repo. Render reads
   [`render.yaml`](../render.yaml) and builds the Docker web service.
3. Set secrets in the dashboard (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) for live embeddings/answers;
   the default offline embedder works with none.
4. For persistence, add a Postgres instance, set `VECTOR_STORE=pgvector` + `DATABASE_URL`, and
   `uv sync --extra pgvector` in the image.

Fly.io / Railway are equivalent: they build the same `Dockerfile`; the start command is already the
`uvicorn` `CMD`.

## CI/CD

- [`ci.yml`](../.github/workflows/ci.yml) gates every push/PR on lint + types + tests.
- [`deploy.yml`](../.github/workflows/deploy.yml) triggers a Render deploy **after CI passes on
  main** — but only if the `RENDER_DEPLOY_HOOK_URL` secret is set, so it's safe to commit before any
  cloud account exists. Add the secret (Render → Settings → Deploy Hook) to turn it on.

## Scaling & cost notes (the "production ML is 80% infra" part)
- **Caching** is the highest-leverage cost lever for repeated queries; semantic caching extends it to
  paraphrases.
- **Model tiering** — cheaper embedding/generation models for high volume (see the cost table in
  `docs/architecture.md`).
- **Async + concurrency** — FastAPI is async; run multiple workers (`uvicorn --workers N`) behind a
  load balancer; make the vector store the shared state (pgvector).
- **Inference at scale** (conceptual) — self-hosted open models use **vLLM** (continuous batching +
  KV-cache) on GPUs; **Kubernetes** for orchestration. This service stays CPU/API-friendly by design.
- **Observability** — wire request traces + eval gates from
  [`llm-eval-kit`](https://github.com/Arunops700/llm-eval-kit) (Milestone 4).
