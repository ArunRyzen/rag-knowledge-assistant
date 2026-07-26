"""Embeddings: text → vector, via Gemini.

An "embedder" turns text into a list of numbers (a vector) so that similar texts end up with
similar vectors — that is what lets us search by meaning instead of exact words.

The `Embedder` protocol keeps the rest of the pipeline decoupled from the provider: the store and
retriever only ever see vectors, so adding another provider later means writing one class with an
`embed` method — nothing downstream changes. Tests use a small offline stub in `tests/conftest.py`
for exactly this reason.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol

from rag_assistant.debuglog import debug_enabled, log_block

if TYPE_CHECKING:
    from google import genai

# Gemini caps how many texts one embed_content call may carry; batch accordingly.
_BATCH_SIZE = 100


def _log_embed_request(label: str, model: str, texts: list[str]) -> None:
    # Debug tracing (LLM_DEBUG=1): what we're about to embed — counts and short previews only,
    # never raw vectors and never API keys.
    if not debug_enabled():
        return
    previews = {f"text_{i}": t[:80] for i, t in enumerate(texts[:5], start=1)}
    log_block(f"AI REQUEST ({label})", model=model, num_texts=len(texts), **previews)


def _log_embed_response(label: str, vectors: list[list[float]]) -> None:
    if not debug_enabled():
        return
    dim = len(vectors[0]) if vectors else 0
    log_block(f"AI RESPONSE ({label})", vectors=len(vectors), dimensions=dim)


class Embedder(Protocol):
    """Turns texts into fixed-dimension vectors."""

    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _l2_normalize(vec: list[float]) -> list[float]:
    # Scale the vector to length 1. After this, comparing two vectors with a dot product
    # gives cosine similarity directly — the standard "how alike are these?" measure.
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


class GeminiEmbedder:
    """Semantic embeddings via Google's Gemini API (model `gemini-embedding-001`).

    We ask the API to return vectors of exactly `dim` numbers (the model supports 768/1536/3072)
    and L2-normalize them ourselves, because Gemini only guarantees pre-normalized output at the
    full 3072 dimensions.
    """

    def __init__(self, *, model: str, dim: int, api_key: str | None = None) -> None:
        from google import genai

        self._client: genai.Client = genai.Client(api_key=api_key)
        self._model = model
        self.dim = dim

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        from google.genai import types

        response = self._client.models.embed_content(
            model=self._model,
            contents=texts,  # type: ignore[arg-type]  # SDK accepts a list of strings
            config=types.EmbedContentConfig(output_dimensionality=self.dim),
        )
        # Results come back in input order; normalize so cosine similarity behaves.
        return [_l2_normalize(list(item.values or [])) for item in response.embeddings or []]

    def embed(self, texts: list[str]) -> list[list[float]]:
        label = f"gemini/{self._model}"
        _log_embed_request(label, self._model, texts)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            vectors.extend(self._embed_batch(texts[start : start + _BATCH_SIZE]))
        _log_embed_response(label, vectors)
        return vectors
