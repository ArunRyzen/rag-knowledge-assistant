"""Opt-in debug tracing for every AI call — a learning aid, not a production logger.

Set the environment variable ``LLM_DEBUG=1`` **or** put ``LLM_DEBUG=1`` in a ``.env`` file in
your working directory, and every embedder and answerer call prints a plain-ASCII
request/response block to **stderr**, so you can watch exactly what goes into (and comes out of)
each model. The offline fakes are traced too, so this works with no API key. The real environment
variable always wins over the ``.env`` file; unset, empty, ``"0"``, or ``"false"`` (any casing)
leaves it off.

Ground rules baked in here:

- stderr only — stdout (the actual answer) stays clean and pipeable.
- API keys are never logged; call sites only pass prompts, previews, and counts.
- Long fields are truncated so a huge document can't flood your terminal.
"""

from __future__ import annotations

import functools
import os
import sys

# Fields longer than this are cut off with a "... [truncated]" marker.
_MAX_FIELD_CHARS = 2000

# Values of LLM_DEBUG that mean "off" (compared case-insensitively).
_FALSY = {"", "0", "false"}


def _is_truthy(value: str) -> bool:
    """Shared on/off rule: anything except "", "0", or "false" (any casing) means on."""
    return value.strip().lower() not in _FALSY


@functools.lru_cache(maxsize=1)
def debug_enabled() -> bool:
    """True when ``LLM_DEBUG`` is switched on via the environment or a local ``.env`` file.

    Precedence (highest first):

    1. The real environment variable ``LLM_DEBUG`` — if it is set *at all* (even to ``"0"``),
       its value alone decides, so an exported variable always beats the ``.env`` file.
    2. An ``LLM_DEBUG`` line in a ``.env`` file in the current working directory, read with
       ``python-dotenv`` (already installed as a dependency of ``pydantic-settings``).
    3. Neither source set → debugging is off.

    The result is cached with ``functools.lru_cache`` so we don't re-read ``.env`` from disk on
    every AI call; tests reset it with ``debug_enabled.cache_clear()``.
    """
    # 1) The real environment variable wins whenever it is present.
    env_value = os.environ.get("LLM_DEBUG")
    if env_value is not None:
        return _is_truthy(env_value)

    # 2) Fall back to `.env` in the current directory. python-dotenv is a transitive dependency
    #    (via pydantic-settings), but we import lazily and tolerate its absence: no library,
    #    no `.env` support — debugging simply stays off.
    try:
        from dotenv import dotenv_values
    except ImportError:
        return False

    file_value = dotenv_values(".env").get("LLM_DEBUG")
    if file_value is None:
        return False  # 3) Neither the env var nor the file mentions LLM_DEBUG.
    return _is_truthy(file_value)


def _truncate(value: str) -> str:
    if len(value) <= _MAX_FIELD_CHARS:
        return value
    return value[:_MAX_FIELD_CHARS] + "... [truncated]"


def log_block(title: str, **fields: object) -> None:
    """Print one ``=== title ===`` block with ``name: value`` lines to stderr.

    No-op unless :func:`debug_enabled`, so call sites can stay a single unconditional line.
    Example output::

        === AI REQUEST (gemini/gemini-2.5-flash) ===
        system: You are a precise question-answering assistant. ...
        user: Which index makes vector search fast?
        context: [1] pgvector adds an HNSW index...
        ============================================
    """
    if not debug_enabled():
        return
    header = f"=== {title} ==="
    lines = [header]
    for name, value in fields.items():
        lines.append(f"{name}: {_truncate(str(value))}")
    lines.append("=" * len(header))
    print("\n".join(lines), file=sys.stderr, flush=True)
