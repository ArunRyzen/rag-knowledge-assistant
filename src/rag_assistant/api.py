"""FastAPI service.

One pipeline instance lives for the process lifetime (the in-memory store persists across
requests). The bundled sample corpus is ingested on startup so `/ask` works immediately; `POST
/ingest` adds more. The endpoints are thin — all logic is in the library.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag_assistant.config import load_settings
from rag_assistant.evaluation import GoldenItem, compare_modes
from rag_assistant.factory import build_pipeline
from rag_assistant.pipeline import RAGPipeline
from rag_assistant.sample_data import GOLDEN, SAMPLE_DOCS

app = FastAPI(title="rag-knowledge-assistant", version="0.1.0")


@lru_cache
def _pipeline() -> RAGPipeline:
    pipeline = build_pipeline(load_settings())
    for doc_id, text in SAMPLE_DOCS.items():
        pipeline.ingest(doc_id, text)
    return pipeline


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
def ask(request: AskRequest) -> dict:
    answer = _pipeline().ask(request.question, mode=request.mode, rerank=request.rerank)
    return answer.model_dump()


@app.get("/eval")
def evaluate(k: int = 5) -> list[dict]:
    dataset = [GoldenItem(**item) for item in GOLDEN]  # type: ignore[arg-type]
    metrics = compare_modes(_pipeline().retriever, dataset, k=k)
    return [
        {"mode": m.mode, "k": m.k, "recall_at_k": m.recall_at_k, "mrr": m.mrr, "n": m.n}
        for m in metrics
    ]
