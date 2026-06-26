"""Loading documents to ingest.

Reads `.md` / `.txt` files from a directory (doc id = filename stem), or falls back to the bundled
sample corpus. Kept separate from the CLI so it's reusable and testable.
"""

from __future__ import annotations

from pathlib import Path

from rag_assistant.sample_data import SAMPLE_DOCS


def load_corpus(path: Path | None) -> list[tuple[str, str]]:
    """Return [(doc_id, text), ...]. With no path, returns the bundled sample corpus."""
    if path is None:
        return list(SAMPLE_DOCS.items())
    if path.is_file():
        return [(path.stem, path.read_text(encoding="utf-8"))]
    docs: list[tuple[str, str]] = []
    for file in sorted(path.glob("**/*")):
        if file.suffix.lower() in {".md", ".txt"} and file.is_file():
            docs.append((file.stem, file.read_text(encoding="utf-8")))
    return docs
