"""Vector stores behind one interface.

`InMemoryVectorStore` (numpy cosine) is the zero-infra default — perfect for development, tests,
and small corpora. `PgVectorStore` is the production backend: Postgres + the pgvector extension,
which keeps your vectors next to your relational data and scales with an HNSW index.

Both satisfy the same `VectorStore` protocol, so the retriever is oblivious to which one runs.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from rag_assistant.models import Chunk, RetrievedChunk


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


class PgVectorStore:
    """Postgres + pgvector backend. Requires the `pgvector` optional dependency and a database.

    Schema is created on first use. Search uses cosine distance (`<=>`); for large corpora add an
    HNSW index: `CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);`.
    """

    def __init__(self, *, database_url: str, dim: int) -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        self._conn = psycopg.connect(database_url, autocommit=True)
        register_vector(self._conn)
        self._dim = dim
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding vector({dim})
            )
            """
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        with self._conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (id, doc_id, idx, text, embedding) "
                "VALUES (%s,%s,%s,%s,%s) "
                "ON CONFLICT (id) DO UPDATE SET "
                "text = EXCLUDED.text, embedding = EXCLUDED.embedding",
                [
                    (c.id, c.doc_id, c.index, c.text, e)
                    for c, e in zip(chunks, embeddings, strict=True)
                ],
            )

    def search(self, query_embedding: list[float], k: int) -> list[RetrievedChunk]:
        rows = self._conn.execute(
            "SELECT id, doc_id, idx, text, 1 - (embedding <=> %s::vector) AS score "
            "FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s",
            (query_embedding, query_embedding, k),
        ).fetchall()
        return [
            RetrievedChunk(
                chunk=Chunk(id=r[0], doc_id=r[1], index=r[2], text=r[3]),
                score=float(r[4]),
                source="dense",
            )
            for r in rows
        ]

    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return int(row[0]) if row else 0
