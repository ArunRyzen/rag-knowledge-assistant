"""Loading documents to ingest: the EXTRACTION step of the pipeline.

Everything downstream (chunking, embedding, retrieval) works on plain text — so this file's only
job is turning files into `(doc_id, text)` pairs. Markdown and .txt files ARE text already; PDFs
need extraction, because a PDF stores drawing instructions ("place these glyphs at these
coordinates"), not paragraphs. `pypdf` walks each page and reconstructs the text for us.

The default corpus is the `data/` folder in the project root — put your own documents there (or
point `--data` anywhere else) and they become the knowledge base. Doc id = filename stem.
"""

from __future__ import annotations

from pathlib import Path

from rag_assistant.errors import IngestionError

DEFAULT_DATA_DIR = Path("data")

_SUFFIXES = {".md", ".txt", ".pdf"}


def _extract_pdf(path: Path) -> str:
    from pypdf import PdfReader

    # One string per page, joined with blank lines so the chunker sees page breaks as
    # paragraph boundaries. Scanned/image-only PDFs yield empty text — that needs OCR,
    # which is out of scope here.
    reader = PdfReader(path)
    return "\n\n".join((page.extract_text() or "") for page in reader.pages).strip()


def _read(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _extract_pdf(path)
    return path.read_text(encoding="utf-8")


def load_corpus(path: Path | None) -> list[tuple[str, str]]:
    """Return [(doc_id, text), ...]. With no path, reads the project's `data/` folder."""
    root = path or DEFAULT_DATA_DIR
    if root.is_file():
        return [(root.stem, _read(root))]
    if not root.is_dir():
        raise IngestionError(
            f"Corpus folder '{root}' not found. Create it and add .md/.txt/.pdf documents, "
            "or pass --data pointing at your documents."
        )
    docs: list[tuple[str, str]] = []
    for file in sorted(root.glob("**/*")):
        if file.suffix.lower() in _SUFFIXES and file.is_file():
            docs.append((file.stem, _read(file)))
    if not docs:
        raise IngestionError(f"No .md/.txt/.pdf files found in '{root}'.")
    return docs
