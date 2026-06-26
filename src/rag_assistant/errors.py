"""Domain exceptions, so callers can distinguish config/ingest/retrieval/generation failures."""

from __future__ import annotations


class RAGError(Exception):
    """Base class for all pipeline errors."""


class ConfigError(RAGError):
    """Missing or invalid configuration (e.g. no embeddings API key)."""


class IngestionError(RAGError):
    """A document could not be loaded, chunked, or embedded."""


class RetrievalError(RAGError):
    """Retrieval failed (store unavailable, dimension mismatch, etc.)."""


class GenerationError(RAGError):
    """The answer-generation step failed."""
