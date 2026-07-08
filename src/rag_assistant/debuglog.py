"""Opt-in debug tracing for every AI call — a learning aid, not a production logger.

Set the environment variable ``LLM_DEBUG=1`` and every embedder and answerer call prints a
plain-ASCII request/response block to **stderr**, so you can watch exactly what goes into (and
comes out of) each model. The offline fakes are traced too, so this works with no API key.
Unset, empty, ``"0"``, or ``"false"`` (any casing) leaves it off.

Ground rules baked in here:

- stderr only — stdout (the actual answer) stays clean and pipeable.
- API keys are never logged; call sites only pass prompts, previews, and counts.
- Long fields are truncated so a huge document can't flood your terminal.
"""

from __future__ import annotations

import os
import sys

# Fields longer than this are cut off with a "... [truncated]" marker.
_MAX_FIELD_CHARS = 2000

# Values of LLM_DEBUG that mean "off" (compared case-insensitively).
_FALSY = {"", "0", "false"}


def debug_enabled() -> bool:
    """True when the ``LLM_DEBUG`` env var is set to anything except "", "0", or "false"."""
    return os.environ.get("LLM_DEBUG", "").strip().lower() not in _FALSY


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
