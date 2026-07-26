"""Composition root: build a fully wired `RAGPipeline` from `Settings`.

All the "which implementation?" decisions live here, so the CLI and API just ask for a pipeline.
There is one real path — Gemini for embeddings + answers, memory or Pinecone for vectors — and
missing keys fail loudly with a ConfigError instead of silently degrading.
"""

from __future__ import annotations

from functools import partial

from rag_assistant.chat import GuardedChat
from rag_assistant.config import Settings
from rag_assistant.embeddings import Embedder, GeminiEmbedder
from rag_assistant.errors import ConfigError
from rag_assistant.generation import Answerer, GeminiAnswerer, generate_text
from rag_assistant.pipeline import RAGPipeline
from rag_assistant.rerank import CrossEncoderReranker, NoopReranker, Reranker
from rag_assistant.vectorstore import InMemoryVectorStore, PineconeVectorStore, VectorStore


def _require_gemini_key(settings: Settings) -> str:
    if not settings.gemini_api_key:
        raise ConfigError(
            "GEMINI_API_KEY is not set. Add it to your .env file (get a free key at "
            "https://aistudio.google.com/apikey)."
        )
    return settings.gemini_api_key


def build_embedder(settings: Settings) -> Embedder:
    return GeminiEmbedder(
        model=settings.gemini_embedding_model,
        dim=settings.gemini_embedding_dim,
        api_key=_require_gemini_key(settings),
    )


def build_vector_store(settings: Settings) -> VectorStore:
    if settings.vector_store == "memory":
        return InMemoryVectorStore()
    if settings.vector_store == "pinecone":
        if not settings.pinecone_api_key:
            raise ConfigError(
                "PINECONE_API_KEY is required when VECTOR_STORE=pinecone. Add it to your .env "
                "(from the API Keys page at https://app.pinecone.io)."
            )
        return PineconeVectorStore(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index,
            dim=settings.gemini_embedding_dim,  # index dimension must match the embedder
        )
    raise ConfigError(f"Unknown VECTOR_STORE '{settings.vector_store}' (memory|pinecone).")


def build_answerer(settings: Settings) -> Answerer:
    return GeminiAnswerer(
        model=settings.gemini_model,
        max_tokens=settings.max_tokens,
        api_key=_require_gemini_key(settings),
    )


def build_reranker(settings: Settings) -> Reranker:
    return CrossEncoderReranker() if settings.use_reranker else NoopReranker()


def build_chat(settings: Settings, pipeline: RAGPipeline) -> GuardedChat:
    """The guarded chatbot: same pipeline, plus the guard/condense/check agents on Gemini."""
    llm = partial(
        _agent_llm,
        model=settings.gemini_model,
        api_key=_require_gemini_key(settings),
    )
    return GuardedChat(pipeline=pipeline, llm=llm)


def _agent_llm(system: str, prompt: str, *, model: str, api_key: str) -> str:
    # Agent calls are short verdicts/rewrites — a small token cap keeps them fast and cheap.
    return generate_text(model=model, max_tokens=256, api_key=api_key, system=system, prompt=prompt)


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
