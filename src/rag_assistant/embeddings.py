"""Embedders behind one interface.

`HashingEmbedder` is a dependency-free, deterministic bag-of-words embedder: it hashes tokens
into a fixed-dimension vector and L2-normalizes. It is NOT semantic — but it is real lexical
similarity, needs no API key, and makes the whole pipeline runnable and testable offline. Use
`OpenAIEmbedder` for production-quality semantic embeddings.

Swapping embedders changes nothing downstream — the store and retriever only see vectors.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from openai import OpenAI

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    """Turns texts into fixed-dimension vectors."""

    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


class HashingEmbedder:
    """Deterministic, offline bag-of-words embedder (the zero-dependency default)."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.md5(token.encode()).digest()  # noqa: S324 - not security-sensitive
            idx = int.from_bytes(digest[:4], "big") % self.dim
            vec[idx] += 1.0
        return _l2_normalize(vec)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class OpenAIEmbedder:
    """Semantic embeddings via the OpenAI embeddings endpoint."""

    def __init__(self, *, model: str, dim: int, api_key: str | None = None) -> None:
        from openai import OpenAI

        self._client: OpenAI = OpenAI(api_key=api_key)
        self._model = model
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self._model, input=texts)
        # Results come back in input order; preserve it.
        return [item.embedding for item in response.data]
