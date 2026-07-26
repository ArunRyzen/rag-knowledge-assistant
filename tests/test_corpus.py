"""Corpus loading: text files pass through, PDFs get their text extracted."""

from __future__ import annotations

from pathlib import Path

import pytest
from fpdf import FPDF

from rag_assistant.corpus import load_corpus
from rag_assistant.errors import IngestionError


def _write_pdf(path: Path, text: str) -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 6, text)
    pdf.output(str(path))


def test_loads_text_and_pdf_files(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("plain markdown content", encoding="utf-8")
    _write_pdf(tmp_path / "paper.pdf", "The quick brown fox appears inside a PDF page.")

    docs = dict(load_corpus(tmp_path))

    assert docs["notes"] == "plain markdown content"
    # PDF text is EXTRACTED, not read raw — the words come back as plain text.
    assert "quick brown fox" in docs["paper"]


def test_single_pdf_file(tmp_path: Path) -> None:
    _write_pdf(tmp_path / "single.pdf", "just one pdf")
    docs = load_corpus(tmp_path / "single.pdf")
    assert docs[0][0] == "single"
    assert "just one pdf" in docs[0][1]


def test_unknown_extensions_are_skipped(tmp_path: Path) -> None:
    (tmp_path / "keep.txt").write_text("kept", encoding="utf-8")
    (tmp_path / "skip.docx").write_text("ignored", encoding="utf-8")
    docs = dict(load_corpus(tmp_path))
    assert set(docs) == {"keep"}


def test_missing_folder_raises_helpful_error(tmp_path: Path) -> None:
    with pytest.raises(IngestionError, match="not found"):
        load_corpus(tmp_path / "nope")


def test_empty_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(IngestionError, match="No .md"):
        load_corpus(tmp_path)
