"""Embedders behind one interface.

An "embedder" turns text into a list of numbers (a vector) so that similar texts end up with
similar vectors — that is what lets us search by meaning instead of exact words.

`HashingEmbedder` is a dependency-free, deterministic bag-of-words embedder: it hashes tokens
into a fixed-dimension vector and L2-normalizes. It is NOT semantic — but it is real lexical
similarity, needs no API key, and makes the whole pipeline runnable and testable offline. Use
`GeminiEmbedder` (or `OpenAIEmbedder`) for production-quality semantic embeddings.

Swapping embedders changes nothing downstream — the store and retriever only see vectors.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import TYPE_CHECKING, Protocol

from rag_assistant.debuglog import debug_enabled, log_block

if TYPE_CHECKING:
    from google import genai
    from openai import OpenAI

_TOKEN = re.compile(r"[a-z0-9]+")


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


class HashingEmbedder:
    """Deterministic, offline bag-of-words embedder (the zero-dependency default)."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        # Each word is hashed to one of `dim` buckets and we count occurrences — so two texts
        # sharing many words get similar vectors. Cheap, deterministic, and offline; the trade-off
        # is it only knows word overlap, not meaning ("car" and "automobile" look unrelated).
        vec = [0.0] * self.dim
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.md5(token.encode()).digest()  # noqa: S324 - not security-sensitive
            idx = int.from_bytes(digest[:4], "big") % self.dim
            vec[idx] += 1.0
        return _l2_normalize(vec)

    def embed(self, texts: list[str]) -> list[list[float]]:
        _log_embed_request("offline fake embedder", f"hashing-{self.dim}d", texts)
        vectors = [self._embed_one(t) for t in texts]
        _log_embed_response("offline fake embedder", vectors)
        return vectors


class GeminiEmbedder:
    """Semantic embeddings via Google's Gemini API (model `gemini-embedding-001`).

    This is the live path for anyone with a GEMINI_API_KEY. We ask the API to return vectors of
    exactly `dim` numbers (the model supports 768/1536/3072) and L2-normalize them ourselves,
    because Gemini only guarantees pre-normalized output at the full 3072 dimensions.
    """

    def __init__(self, *, model: str, dim: int, api_key: str | None = None) -> None:
        # Imported lazily so the offline default never needs the google-genai package at runtime.
        from google import genai

        self._client: genai.Client = genai.Client(api_key=api_key)
        self._model = model
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        from google.genai import types

        _log_embed_request(f"gemini/{self._model}", self._model, texts)
        response = self._client.models.embed_content(
            model=self._model,
            contents=texts,  # type: ignore[arg-type]  # SDK accepts a list of strings
            config=types.EmbedContentConfig(output_dimensionality=self.dim),
        )
        # Results come back in input order; normalize so cosine similarity behaves.
        vectors = [_l2_normalize(list(item.values or [])) for item in response.embeddings or []]
        _log_embed_response(f"gemini/{self._model}", vectors)
        return vectors


class OpenAIEmbedder:
    """Semantic embeddings via the OpenAI embeddings endpoint."""

    def __init__(self, *, model: str, dim: int, api_key: str | None = None) -> None:
        from openai import OpenAI

        self._client: OpenAI = OpenAI(api_key=api_key)
        self._model = model
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        _log_embed_request(f"openai/{self._model}", self._model, texts)
        response = self._client.embeddings.create(model=self._model, input=texts)
        # Results come back in input order; preserve it.
        vectors = [item.embedding for item in response.data]
        _log_embed_response(f"openai/{self._model}", vectors)
        return vectors
