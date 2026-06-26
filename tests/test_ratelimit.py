"""The rate limiter allows up to the limit per window, then denies."""

from __future__ import annotations

from rag_assistant.ratelimit import RateLimiter


def test_allows_up_to_limit_then_denies() -> None:
    limiter = RateLimiter(max_requests=2, window_seconds=60.0)
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is False  # third in the window


def test_limits_are_per_key() -> None:
    limiter = RateLimiter(max_requests=1, window_seconds=60.0)
    assert limiter.allow("a") is True
    assert limiter.allow("b") is True  # different client, own budget
    assert limiter.allow("a") is False
