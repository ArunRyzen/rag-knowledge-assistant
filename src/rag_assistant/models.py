"""Core domain types.

Pydantic models for the data that flows through the pipeline. Keeping these explicit (rather
than passing dicts around) is what makes the retrieval and eval code readable and type-checked.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A retrievable unit of text with provenance back to its source document."""

    id: str = Field(description="Stable unique id, e.g. '<doc_id>::<index>'.")
    doc_id: str = Field(description="Id of the source document.")
    text: str = Field(description="The chunk's text content.")
    index: int = Field(description="Position of this chunk within its document.")
    metadata: dict[str, str] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A chunk returned by retrieval, with the score that ranked it."""

    chunk: Chunk
    score: float = Field(description="Relevance score (higher is better; method-dependent).")
    source: str = Field(default="hybrid", description="Which retriever produced it.")


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    quote: str = Field(description="The supporting snippet from the chunk.")


class Answer(BaseModel):
    """A generated answer plus the chunks it was grounded in."""

    question: str
    text: str
    citations: list[Citation] = Field(default_factory=list)
    contexts: list[RetrievedChunk] = Field(default_factory=list)
