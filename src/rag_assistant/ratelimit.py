"""A simple in-memory rate limiter (fixed sliding window per client).

Rate limiting is table stakes for a public LLM endpoint — it caps cost and abuse. This is a
per-key sliding window; in production you'd back it with Redis so the limit is shared across
replicas. The interface is deliberately tiny: `allow(key) -> bool`.
"""

from __future__ import annotations

from time import monotonic


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._hits: dict[str, list[float]] = {}

    def allow(self, key: str) -> bool:
        """Record a request for `key`; return False if it exceeds the window limit."""
        now = monotonic()
        window_start = now - self._window
        recent = [t for t in self._hits.get(key, []) if t >= window_start]
        if len(recent) >= self._max:
            self._hits[key] = recent
            return False
        recent.append(now)
        self._hits[key] = recent
        return True
