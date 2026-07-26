"""Domain exceptions, so callers can distinguish config failures from ingestion failures."""

from __future__ import annotations


class RAGError(Exception):
    """Base class for all pipeline errors."""


class ConfigError(RAGError):
    """Missing or invalid configuration (e.g. no API key, wrong vector store name)."""


class IngestionError(RAGError):
    """A document could not be loaded or extracted."""
