"""Chunking: size bounds, overlap, and stable ids."""

from __future__ import annotations

from rag_assistant.chunking import chunk_document


def test_short_text_is_one_chunk() -> None:
    chunks = chunk_document(doc_id="d", text="just a short sentence.", size=400, overlap=50)
    assert len(chunks) == 1
    assert chunks[0].id == "d::0"
    assert chunks[0].doc_id == "d"


def test_long_text_splits_with_overlap() -> None:
    text = ". ".join(f"sentence number {i} with some words" for i in range(60))
    chunks = chunk_document(doc_id="d", text=text, size=200, overlap=40)

    assert len(chunks) > 1
    # ids are sequential and stable
    assert [c.index for c in chunks] == list(range(len(chunks)))
    # overlap: the tail of chunk 0 should appear at the start of chunk 1
    tail = chunks[0].text[-20:]
    assert tail.split()[-1] in chunks[1].text


def test_empty_text_yields_no_chunks() -> None:
    assert chunk_document(doc_id="d", text="   ", size=400, overlap=50) == []
