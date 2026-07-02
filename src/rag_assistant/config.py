"""Configuration via environment / `.env`.

Every knob the pipeline exposes lives here as one `Settings` field. Pydantic reads each field
from an environment variable of the same name (upper-cased), falling back to a `.env` file, then
to the default written below. So `chunk_size` ⇐ env var `CHUNK_SIZE` ⇐ default 800.

Defaults are chosen so the pipeline runs with the least friction: an in-memory vector store
(no database) and no API keys required. Swap to pgvector by setting VECTOR_STORE=pgvector.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- API keys ---
    # If GEMINI_API_KEY is set, Gemini is used for BOTH embeddings and answers (it's the
    # provider most learners have — one free key covers everything). With no keys at all the
    # pipeline still runs, using the offline hashing embedder + fake answerer.
    gemini_api_key: str | None = Field(default=None)
    openai_api_key: str | None = Field(default=None)
    anthropic_api_key: str | None = Field(default=None)

    # Embeddings
    gemini_embedding_model: str = Field(default="gemini-embedding-001")
    gemini_embedding_dim: int = Field(default=768)  # gemini-embedding-001 supports 768/1536/3072
    embedding_model: str = Field(default="text-embedding-3-small")  # OpenAI path
    embedding_dim: int = Field(default=1536)  # dimension of the OpenAI model above

    # Generation (answer synthesis)
    generation_provider: str = Field(default="gemini")  # gemini | anthropic | openai
    gemini_model: str = Field(default="gemini-2.5-flash")
    anthropic_model: str = Field(default="claude-opus-4-8")
    openai_model: str = Field(default="gpt-4o")
    max_tokens: int = Field(default=1024)

    # Vector store: "memory" (zero-infra default) or "pgvector"
    vector_store: str = Field(default="memory")
    database_url: str | None = Field(default=None)  # required for pgvector

    # Chunking — THE chunk-size knob. Documents are cut into pieces of roughly this many
    # characters before indexing (see chunking.py). Halve it for more precise-but-fragmented
    # chunks; quadruple it for fewer, more contextual ones. Overlap repeats the tail of each
    # chunk at the start of the next so a fact on the border isn't lost.
    chunk_size: int = Field(default=800)  # target characters per chunk
    chunk_overlap: int = Field(default=120)  # characters shared between neighbouring chunks

    # Retrieval
    top_k: int = Field(default=5)  # final contexts passed to the generator
    candidate_k: int = Field(default=20)  # candidates pulled before fusion/rerank
    use_reranker: bool = Field(default=False)
    # RRF constant used when fusing dense + sparse rankings (see retrieval.py). Bigger values
    # flatten the difference between rank 1 and rank 10; smaller values favour the top ranks.
    rrf_k: int = Field(default=60)  # Reciprocal Rank Fusion constant


def load_settings() -> Settings:
    return Settings()
