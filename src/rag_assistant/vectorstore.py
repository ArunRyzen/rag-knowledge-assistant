"""Vector stores behind one interface.

`InMemoryVectorStore` (numpy cosine) lives entirely in-process: instant, free, but gone when the
process exits. `PineconeVectorStore` is the persistent backend — a managed, serverless vector
database: ingest once, query from any process afterwards.

Both satisfy the same `VectorStore` protocol, so the retriever is oblivious to which one runs.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Protocol

import numpy as np

from rag_assistant.errors import ConfigError
from rag_assistant.models import Chunk, RetrievedChunk

if TYPE_CHECKING:
    from pinecone import Pinecone


class VectorStore(Protocol):
    """Stores chunk embeddings and answers nearest-neighbour queries."""

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    def search(self, query_embedding: list[float], k: int) -> list[RetrievedChunk]: ...

    def __len__(self) -> int: ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class InMemoryVectorStore:
    """Cosine-similarity search over an in-memory matrix. Good to a few hundred thousand chunks."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        new = _normalize(np.asarray(embeddings, dtype=np.float32))
        self._matrix = new if self._matrix is None else np.vstack([self._matrix, new])
        self._chunks.extend(chunks)

    def search(self, query_embedding: list[float], k: int) -> list[RetrievedChunk]:
        if self._matrix is None or not self._chunks:
            return []
        query = _normalize(np.asarray([query_embedding], dtype=np.float32))[0]
        # Cosine similarity == dot product on normalized vectors.
        scores = self._matrix @ query
        top = np.argsort(-scores)[:k]
        return [
            RetrievedChunk(chunk=self._chunks[i], score=float(scores[i]), source="dense")
            for i in top
        ]

    def __len__(self) -> int:
        return len(self._chunks)


class PineconeVectorStore:
    """Pinecone serverless backend: a managed vector database in the cloud.

    Each chunk becomes one Pinecone *record*: the embedding as the vector, plus the chunk's text
    and provenance stored as metadata so search results can be reconstructed into `Chunk`s
    without any second datastore. Records are upserted by chunk id, so re-ingesting the same
    document overwrites cleanly instead of duplicating.

    Two real-world constraints worth knowing (both great interview talking points):
    - The index dimension is FIXED at creation and must match the embedder (768 for the default
      Gemini setup). We check and fail loudly on mismatch.
    - Upserts are eventually consistent — a record may take a few seconds to become searchable.
      That's why ingestion is a separate, run-once `rag ingest` step.
    """

    _UPSERT_BATCH = 100

    def __init__(self, *, api_key: str, index_name: str, dim: int) -> None:
        from pinecone import Pinecone

        self._pc: Pinecone = Pinecone(api_key=api_key)
        self._dim = dim
        self._ensure_index(index_name)
        self._index = self._pc.Index(index_name)

    def _ensure_index(self, name: str) -> None:
        """Create the index if missing (serverless, cosine); verify the dimension if present."""
        from pinecone import ServerlessSpec

        if not self._pc.has_index(name):
            self._pc.create_index(
                name=name,
                dimension=self._dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            # A fresh index takes a moment to come up; wait until Pinecone reports it ready.
            while not self._pc.describe_index(name).status.ready:
                time.sleep(1)
            return
        existing_dim = self._pc.describe_index(name).dimension
        if existing_dim != self._dim:
            raise ConfigError(
                f"Pinecone index '{name}' has dimension {existing_dim}, but the embedder "
                f"produces {self._dim}-dim vectors. Delete the index (or use another name) "
                "and re-run `rag ingest`."
            )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        records = [
            {
                "id": c.id,
                "values": e,
                "metadata": {"doc_id": c.doc_id, "index": c.index, "text": c.text},
            }
            for c, e in zip(chunks, embeddings, strict=True)
        ]
        # Batched upserts: Pinecone caps request sizes, and batching keeps memory flat.
        for start in range(0, len(records), self._UPSERT_BATCH):
            self._index.upsert(vectors=records[start : start + self._UPSERT_BATCH])

    def search(self, query_embedding: list[float], k: int) -> list[RetrievedChunk]:
        response = self._index.query(vector=query_embedding, top_k=k, include_metadata=True)
        results: list[RetrievedChunk] = []
        for match in response.matches:
            meta = match.metadata or {}
            chunk = Chunk(
                id=match.id,
                doc_id=str(meta.get("doc_id", "")),
                text=str(meta.get("text", "")),
                index=int(meta.get("index", 0)),
            )
            results.append(RetrievedChunk(chunk=chunk, score=float(match.score), source="dense"))
        return results

    def __len__(self) -> int:
        return int(self._index.describe_index_stats().total_vector_count)
