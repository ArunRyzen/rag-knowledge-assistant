"""A simple in-memory rate limiter (fixed sliding window per client).

Rate limiting is table stakes for a public LLM endpoint — it caps cost and abuse. This is a
per-key sliding window; in production you'd back it with Redis so the limit is shared across
replicas. The interface is deliberately tiny: `allow(key) -> bool`.
"""

from __future__ import annotations

from time import monotonic


class RateLimiter:
    """Allows at most `max_requests` per `window_seconds` for each key (e.g. client IP).

    The actual numbers are NOT set here — the API passes them in from `RATE_LIMIT_MAX` and
    `RATE_LIMIT_WINDOW_S` at the top of api.py (default: 60 requests per 60 seconds).
    """

    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self._max = max_requests
        self._window = window_seconds
        # Per key, the timestamps of its recent requests.
        self._hits: dict[str, list[float]] = {}

    def allow(self, key: str) -> bool:
        """Record a request for `key`; return False if it exceeds the window limit."""
        # `monotonic` is a clock that never jumps backwards (unlike wall time on NTP sync),
        # which makes it the right choice for measuring durations.
        now = monotonic()
        window_start = now - self._window
        # Keep only the timestamps that still fall inside the sliding window.
        recent = [t for t in self._hits.get(key, []) if t >= window_start]
        if len(recent) >= self._max:
            # Over the limit: refuse, and do NOT record this attempt (rejected calls
            # shouldn't push the reset time further away).
            self._hits[key] = recent
            return False
        recent.append(now)
        self._hits[key] = recent
        return True
