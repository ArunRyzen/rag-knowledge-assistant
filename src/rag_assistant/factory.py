"""Composition root: build a fully wired `RAGPipeline` from `Settings`.

All the "which implementation?" decisions live here, so the CLI and API just ask for a pipeline.
Provider preference: a GEMINI_API_KEY switches BOTH embeddings and answers to Gemini (one key,
whole live path). Sensible fallbacks keep the tool runnable with zero configuration: no embeddings
key → the offline hashing embedder; no generation key → the fake answerer.
"""

from __future__ import annotations

from rag_assistant.config import Settings
from rag_assistant.embeddings import Embedder, GeminiEmbedder, HashingEmbedder, OpenAIEmbedder
from rag_assistant.errors import ConfigError
from rag_assistant.generation import Answerer, FakeAnswerer, LLMAnswerer
from rag_assistant.pipeline import RAGPipeline
from rag_assistant.rerank import CrossEncoderReranker, NoopReranker, Reranker
from rag_assistant.vectorstore import InMemoryVectorStore, PgVectorStore, VectorStore


def build_embedder(settings: Settings) -> Embedder:
    # Gemini first: it's the one key most learners have, and it covers embeddings too.
    if settings.gemini_api_key:
        return GeminiEmbedder(
            model=settings.gemini_embedding_model,
            dim=settings.gemini_embedding_dim,
            api_key=settings.gemini_api_key,
        )
    if settings.openai_api_key:
        return OpenAIEmbedder(
            model=settings.embedding_model,
            dim=settings.embedding_dim,
            api_key=settings.openai_api_key,
        )
    # Offline default — real lexical similarity, no key required.
    return HashingEmbedder(dim=256)


def _embedding_dim(settings: Settings) -> int:
    """The vector width the store must match — depends on which embedder will be built."""
    if settings.gemini_api_key:
        return settings.gemini_embedding_dim
    if settings.openai_api_key:
        return settings.embedding_dim
    return 256  # HashingEmbedder default


def build_vector_store(settings: Settings) -> VectorStore:
    if settings.vector_store == "memory":
        return InMemoryVectorStore()
    if settings.vector_store == "pgvector":
        if not settings.database_url:
            raise ConfigError("DATABASE_URL is required when VECTOR_STORE=pgvector.")
        return PgVectorStore(database_url=settings.database_url, dim=_embedding_dim(settings))
    raise ConfigError(f"Unknown VECTOR_STORE '{settings.vector_store}' (memory|pgvector).")


def build_answerer(settings: Settings) -> Answerer:
    # "gemini" is the default provider, so setting GEMINI_API_KEY is all it takes to go live.
    if settings.generation_provider == "gemini" and settings.gemini_api_key:
        return LLMAnswerer(
            provider="gemini",
            model=settings.gemini_model,
            max_tokens=settings.max_tokens,
            api_key=settings.gemini_api_key,
        )
    if settings.generation_provider == "anthropic" and settings.anthropic_api_key:
        return LLMAnswerer(
            provider="anthropic",
            model=settings.anthropic_model,
            max_tokens=settings.max_tokens,
            api_key=settings.anthropic_api_key,
        )
    if settings.generation_provider == "openai" and settings.openai_api_key:
        return LLMAnswerer(
            provider="openai",
            model=settings.openai_model,
            max_tokens=settings.max_tokens,
            api_key=settings.openai_api_key,
        )
    # No generation key — still fully functional for retrieval + demos.
    return FakeAnswerer()


def build_reranker(settings: Settings) -> Reranker:
    return CrossEncoderReranker() if settings.use_reranker else NoopReranker()


def build_pipeline(settings: Settings) -> RAGPipeline:
    return RAGPipeline(
        embedder=build_embedder(settings),
        vector_store=build_vector_store(settings),
        answerer=build_answerer(settings),
        reranker=build_reranker(settings),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        candidate_k=settings.candidate_k,
        top_k=settings.top_k,
        rrf_k=settings.rrf_k,
    )
