"""Document chunking.

Why chunk at all? We can't hand a whole book to the retriever — we need bite-sized passages that
can be embedded, indexed, and returned individually. The chunk size itself is NOT set here: it
comes in as the `size` argument, from `chunk_size` in config.py (env var CHUNK_SIZE, default 800
characters).

Chunking is the single highest-leverage RAG decision: too big and retrieval is imprecise and
context-bloated; too small and you lose the context a passage needs to be meaningful. This is a
**recursive, structure-aware** splitter — it prefers to break on paragraph, then sentence, then
word boundaries, and adds overlap so a fact split across a boundary still survives in one chunk.
"""

from __future__ import annotations

import re

from rag_assistant.models import Chunk

# Separators tried in order of preference: paragraph → line → sentence → word.
_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_recursive(text: str, size: int, separators: list[str]) -> list[str]:
    """Greedily pack text into <= `size` pieces, breaking on the best available boundary."""
    if len(text) <= size:
        return [text]
    if not separators:
        # No boundary left — hard-split at the size limit.
        return [text[i : i + size] for i in range(0, len(text), size)]

    # Try the coarsest boundary first (paragraphs), keep the finer ones (`rest`) in reserve.
    sep, *rest = separators
    parts = text.split(sep)
    chunks: list[str] = []
    current = ""
    for part in parts:
        # Keep gluing parts onto the current chunk while it still fits under `size`.
        candidate = part if not current else current + sep + part
        if len(candidate) <= size:
            current = candidate
            continue
        # Adding this part would overflow — close off the chunk we have so far.
        if current:
            chunks.append(current)
        # A single part is itself too big — recurse with the next-finer separator.
        if len(part) > size:
            chunks.extend(_split_recursive(part, size, rest))
            current = ""
        else:
            current = part
    if current:
        chunks.append(current)
    return chunks


def chunk_document(*, doc_id: str, text: str, size: int, overlap: int) -> list[Chunk]:
    """Split a document into overlapping `Chunk`s.

    Overlap is applied by prefixing each chunk (after the first) with the tail of the previous
    one, so context that straddles a boundary appears in both neighbours.
    """
    # Normalize Windows/Mac line endings first so the "\n\n" separator matches everywhere.
    text = re.sub(r"\r\n?", "\n", text).strip()
    if not text:
        return []

    raw = _split_recursive(text, size, _SEPARATORS)
    chunks: list[Chunk] = []
    prev_tail = ""
    for i, body in enumerate(raw):
        # Prepend the last `overlap` characters of the previous chunk. If a sentence like
        # "the password is X" got cut right at the boundary, this repeat keeps it whole
        # in at least one chunk, so retrieval can still find it.
        content = (prev_tail + " " + body).strip() if prev_tail else body
        chunks.append(Chunk(id=f"{doc_id}::{i}", doc_id=doc_id, text=content, index=i))
        prev_tail = body[-overlap:] if overlap > 0 else ""
    return chunks
