"""Configuration via environment / `.env`.

Defaults are chosen so the pipeline runs with the least friction: an in-memory vector store
(no database) and a small OpenAI embedding model. Swap to pgvector by setting VECTOR_STORE=pgvector.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Embeddings
    openai_api_key: str | None = Field(default=None)
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dim: int = Field(default=1536)  # dimension of the model above

    # Generation (answer synthesis)
    generation_provider: str = Field(default="anthropic")  # anthropic | openai
    anthropic_api_key: str | None = Field(default=None)
    anthropic_model: str = Field(default="claude-opus-4-8")
    openai_model: str = Field(default="gpt-4o")
    max_tokens: int = Field(default=1024)

    # Vector store: "memory" (zero-infra default) or "pgvector"
    vector_store: str = Field(default="memory")
    database_url: str | None = Field(default=None)  # required for pgvector

    # Chunking
    chunk_size: int = Field(default=800)  # target characters per chunk
    chunk_overlap: int = Field(default=120)

    # Retrieval
    top_k: int = Field(default=5)  # final contexts passed to the generator
    candidate_k: int = Field(default=20)  # candidates pulled before fusion/rerank
    use_reranker: bool = Field(default=False)
    rrf_k: int = Field(default=60)  # Reciprocal Rank Fusion constant


def load_settings() -> Settings:
    return Settings()
