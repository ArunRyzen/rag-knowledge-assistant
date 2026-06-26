"""Composition root: build a fully wired `RAGPipeline` from `Settings`.

All the "which implementation?" decisions live here, so the CLI and API just ask for a pipeline.
Sensible fallbacks keep the tool runnable with zero configuration: no embeddings key → the offline
hashing embedder; no generation key → the fake answerer.
"""

from __future__ import annotations

from rag_assistant.config import Settings
from rag_assistant.embeddings import Embedder, HashingEmbedder, OpenAIEmbedder
from rag_assistant.errors import ConfigError
from rag_assistant.generation import Answerer, FakeAnswerer, LLMAnswerer
from rag_assistant.pipeline import RAGPipeline
from rag_assistant.rerank import CrossEncoderReranker, NoopReranker, Reranker
from rag_assistant.vectorstore import InMemoryVectorStore, PgVectorStore, VectorStore


def build_embedder(settings: Settings) -> Embedder:
    if settings.openai_api_key:
        return OpenAIEmbedder(
            model=settings.embedding_model,
            dim=settings.embedding_dim,
            api_key=settings.openai_api_key,
        )
    # Offline default — real lexical similarity, no key required.
    return HashingEmbedder(dim=256)


def build_vector_store(settings: Settings) -> VectorStore:
    if settings.vector_store == "memory":
        return InMemoryVectorStore()
    if settings.vector_store == "pgvector":
        if not settings.database_url:
            raise ConfigError("DATABASE_URL is required when VECTOR_STORE=pgvector.")
        dim = settings.embedding_dim if settings.openai_api_key else 256
        return PgVectorStore(database_url=settings.database_url, dim=dim)
    raise ConfigError(f"Unknown VECTOR_STORE '{settings.vector_store}' (memory|pgvector).")


def build_answerer(settings: Settings) -> Answerer:
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
