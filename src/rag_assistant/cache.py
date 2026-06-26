"""A semantic response cache.

An exact-match cache misses on paraphrases ("capital of France?" vs "what's France's capital?"). A
**semantic cache** embeds the query and returns a cached answer when a *similar* query is within a
similarity threshold — cutting cost and latency on the long tail of reworded repeats. Production
deployments back this with Redis + a vector index; this in-memory version has the same semantics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from rag_assistant.embeddings import Embedder


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class SemanticCache:
    """Caches answers keyed by query embedding; serves a hit when similarity ≥ threshold."""

    def __init__(self, embedder: Embedder, *, threshold: float = 0.97, max_size: int = 512) -> None:
        self._embedder = embedder
        self._threshold = threshold
        self._max_size = max_size
        self._entries: list[tuple[list[float], dict]] = []
        self.stats = CacheStats()

    def get(self, query: str) -> dict | None:
        if not self._entries:
            self.stats.misses += 1
            return None
        embedding = self._embedder.embed([query])[0]
        best_answer: dict | None = None
        best_sim = -1.0
        for cached_embedding, answer in self._entries:
            sim = _cosine(embedding, cached_embedding)
            if sim > best_sim:
                best_sim, best_answer = sim, answer
        if best_answer is not None and best_sim >= self._threshold:
            self.stats.hits += 1
            return best_answer
        self.stats.misses += 1
        return None

    def put(self, query: str, answer: dict) -> None:
        embedding = self._embedder.embed([query])[0]
        self._entries.append((embedding, answer))
        if len(self._entries) > self._max_size:
            self._entries.pop(0)  # simple FIFO eviction

    @property
    def size(self) -> int:
        return len(self._entries)
