"""The semantic cache: hits on similar queries, misses on unrelated ones, tracks stats."""

from __future__ import annotations

from rag_assistant.cache import SemanticCache
from rag_assistant.embeddings import HashingEmbedder


def _cache() -> SemanticCache:
    return SemanticCache(HashingEmbedder(dim=128), threshold=0.9)


def test_put_then_get_is_a_hit() -> None:
    cache = _cache()
    cache.put("what is mcp", {"text": "MCP is a protocol"})
    hit = cache.get("what is mcp")
    assert hit == {"text": "MCP is a protocol"}
    assert cache.stats.hits == 1


def test_reordered_query_hits_semantically() -> None:
    cache = _cache()
    cache.put("what is mcp", {"text": "answer"})
    # Same bag of words → high similarity → served from cache (the point of a semantic cache).
    assert cache.get("mcp what is") is not None


def test_unrelated_query_misses() -> None:
    cache = _cache()
    cache.put("what is mcp", {"text": "answer"})
    assert cache.get("how do birds fly") is None
    assert cache.stats.misses >= 1


def test_hit_rate() -> None:
    cache = _cache()
    cache.put("q", {"a": 1})
    cache.get("q")  # hit
    cache.get("zzz totally different words here")  # miss
    assert 0.0 < cache.stats.hit_rate < 1.0
