"""Configuration via environment / `.env`.

Every knob the pipeline exposes lives here as one `Settings` field. Pydantic reads each field
from an environment variable of the same name (upper-cased), falling back to a `.env` file, then
to the default written below. So `chunk_size` ⇐ env var `CHUNK_SIZE` ⇐ default 800.

Two keys are required to run: GEMINI_API_KEY (embeddings + answers) and — when
VECTOR_STORE=pinecone — PINECONE_API_KEY. The factory raises a clear ConfigError if they are
missing, so a typo in `.env` fails loudly instead of silently degrading.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Gemini (required) ---
    # One key covers both halves of the pipeline: semantic embeddings AND answer synthesis.
    gemini_api_key: str | None = Field(default=None)
    gemini_embedding_model: str = Field(default="gemini-embedding-001")
    gemini_embedding_dim: int = Field(default=768)  # gemini-embedding-001 supports 768/1536/3072
    gemini_model: str = Field(default="gemini-2.5-flash")
    max_tokens: int = Field(default=1024)

    # --- Vector store: "memory" (in-process, gone when the process exits) or "pinecone" ---
    # Pinecone is the persistent path: ingest once with `rag ingest`, query any time after.
    vector_store: str = Field(default="memory")
    pinecone_api_key: str | None = Field(default=None)
    pinecone_index: str = Field(default="rag-assistant")

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
