"""HTTP surface: health, ask, ingest, eval — using the offline default pipeline (no keys)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from rag_assistant import api

client = TestClient(api.app)


def test_health() -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_returns_grounded_answer() -> None:
    resp = client.post("/ask", json={"question": "What does BM25 reward?", "mode": "hybrid"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"]
    assert body["contexts"]


def test_ingest_then_count() -> None:
    resp = client.post(
        "/ingest", json={"doc_id": "extra", "text": "a brand new document about cats"}
    )
    assert resp.status_code == 200
    assert resp.json()["chunks_added"] >= 1


def test_eval_reports_all_modes() -> None:
    rows = client.get("/eval").json()
    modes = {r["mode"] for r in rows}
    assert {"dense", "sparse", "hybrid", "hybrid+rerank"} <= modes


def test_ask_rejects_empty_question() -> None:
    assert client.post("/ask", json={"question": ""}).status_code == 422
