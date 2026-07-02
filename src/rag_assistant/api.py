"""FastAPI service with production-serving concerns: semantic caching + rate limiting + metrics.

One pipeline lives for the process lifetime (the in-memory store persists across requests). `/ask`
sits behind a per-client rate limiter and a semantic response cache, and `/metrics` exposes basic
operational counters. The endpoints stay thin — all logic is in the library.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from rag_assistant.cache import SemanticCache
from rag_assistant.config import load_settings
from rag_assistant.evaluation import GoldenItem, compare_modes
from rag_assistant.factory import build_embedder, build_pipeline
from rag_assistant.pipeline import RAGPipeline
from rag_assistant.ratelimit import RateLimiter
from rag_assistant.sample_data import GOLDEN, SAMPLE_DOCS

app = FastAPI(title="rag-knowledge-assistant", version="0.1.0")

# Serving controls — THE rate-limit knobs. Each client may make at most RATE_LIMIT_MAX
# requests per RATE_LIMIT_WINDOW_S seconds; beyond that /ask returns HTTP 429. In production
# these would be Redis-backed and configurable.
RATE_LIMIT_MAX = 60  # requests allowed per window, per client
RATE_LIMIT_WINDOW_S = 60.0  # window length in seconds
_limiter = RateLimiter(RATE_LIMIT_MAX, RATE_LIMIT_WINDOW_S)
_metrics = {"ask_requests": 0}


@lru_cache
def _pipeline() -> RAGPipeline:
    # @lru_cache on a zero-argument function = "build once, reuse forever": every request
    # shares one pipeline, so documents ingested via POST /ingest stay searchable.
    pipeline = build_pipeline(load_settings())
    for doc_id, text in SAMPLE_DOCS.items():
        pipeline.ingest(doc_id, text)
    return pipeline


@lru_cache
def _cache() -> SemanticCache:
    # Share the same embedder family the pipeline uses.
    return SemanticCache(build_embedder(load_settings()))


class IngestRequest(BaseModel):
    doc_id: str
    text: str = Field(min_length=1)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    mode: str = Field(default="hybrid")
    rerank: bool = Field(default=False)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict[str, int]:
    added = _pipeline().ingest(request.doc_id, request.text)
    return {"chunks_added": added}


@app.post("/ask")
def ask(request: AskRequest, http_request: Request) -> dict:
    # Order matters: rate limit FIRST (cheapest check), then cache, then the expensive
    # retrieve-and-generate path only when both let the request through.
    client = http_request.client.host if http_request.client else "unknown"
    if not _limiter.allow(client):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")

    _metrics["ask_requests"] += 1
    # Cache key folds in retrieval mode so different modes don't collide.
    key = f"{request.mode}|{request.rerank}|{request.question}"
    cached = _cache().get(key)
    if cached is not None:
        # Semantic-cache hit: a similar-enough question was answered before — reuse it.
        return {**cached, "cached": True}

    answer = _pipeline().ask(request.question, mode=request.mode, rerank=request.rerank)
    payload = answer.model_dump()
    _cache().put(key, payload)
    return {**payload, "cached": False}


@app.get("/metrics")
def metrics() -> dict:
    cache = _cache()
    return {
        "ask_requests": _metrics["ask_requests"],
        "cache_hits": cache.stats.hits,
        "cache_misses": cache.stats.misses,
        "cache_hit_rate": round(cache.stats.hit_rate, 3),
        "cache_size": cache.size,
        "rate_limit": {"max": RATE_LIMIT_MAX, "window_seconds": RATE_LIMIT_WINDOW_S},
    }


@app.get("/eval")
def evaluate(k: int = 5) -> list[dict]:
    dataset = [GoldenItem(**item) for item in GOLDEN]  # type: ignore[arg-type]
    metrics_list = compare_modes(_pipeline().retriever, dataset, k=k)
    return [
        {"mode": m.mode, "k": m.k, "recall_at_k": m.recall_at_k, "mrr": m.mrr, "n": m.n}
        for m in metrics_list
    ]
