"""The RAG pipeline: ingest → retrieve → generate.

This is the composition of every other module. It owns ingestion (chunk → embed → index into both
the dense store and the BM25 index) and answering (retrieve → generate). It depends only on the
abstractions (`Embedder`, `VectorStore`, `Answerer`), so any backend mix works.
"""

from __future__ import annotations

from rag_assistant.chunking import chunk_document
from rag_assistant.embeddings import Embedder
from rag_assistant.generation import Answerer
from rag_assistant.models import Answer, RetrievedChunk
from rag_assistant.rerank import Reranker
from rag_assistant.retrieval import Retriever
from rag_assistant.sparse import BM25Index
from rag_assistant.vectorstore import VectorStore


class RAGPipeline:
    """End-to-end retrieval-augmented generation."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        vector_store: VectorStore,
        answerer: Answerer,
        bm25: BM25Index | None = None,
        reranker: Reranker | None = None,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        candidate_k: int = 20,
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> None:
        self._embedder = embedder
        self._store = vector_store
        self._bm25 = bm25 or BM25Index()
        self._answerer = answerer
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._retriever = Retriever(
            vector_store=vector_store,
            bm25=self._bm25,
            embedder=embedder,
            reranker=reranker,
            candidate_k=candidate_k,
            top_k=top_k,
            rrf_k=rrf_k,
        )

    @property
    def retriever(self) -> Retriever:
        """Exposed so the eval harness can probe retrieval directly."""
        return self._retriever

    @property
    def vector_count(self) -> int:
        """How many vectors the dense store currently holds (for sanity checks / status)."""
        return len(self._store)

    def reset_store(self) -> None:
        """Wipe the dense store. Use before re-ingesting a CHANGED corpus, so chunks from
        removed documents don't linger (upserts only overwrite matching ids)."""
        self._store.clear()

    def ingest(self, doc_id: str, text: str, *, dense: bool = True) -> int:
        """Chunk, embed, and index one document. Returns the number of chunks added.

        Every chunk is indexed TWICE — once as a vector (for meaning-based search) and once
        in BM25 (for exact-word search). That dual indexing is what makes hybrid possible.

        `dense=False` skips the embed-and-store half and only rebuilds the in-memory BM25
        index. Used when the vector store is persistent (Pinecone): the vectors are already
        there from `rag ingest`, but BM25 lives in this process and must be rebuilt each run —
        chunking is deterministic, so both sides see identical chunks.
        """
        chunks = chunk_document(
            doc_id=doc_id, text=text, size=self._chunk_size, overlap=self._chunk_overlap
        )
        if not chunks:
            return 0
        if dense:
            embeddings = self._embedder.embed([c.text for c in chunks])
            self._store.add(chunks, embeddings)
        self._bm25.add(chunks)
        return len(chunks)

    def retrieve(
        self, question: str, *, mode: str = "hybrid", k: int | None = None, rerank: bool = False
    ) -> list[RetrievedChunk]:
        return self._retriever.retrieve(question, mode=mode, k=k, rerank=rerank)

    def ask(
        self, question: str, *, mode: str = "hybrid", k: int | None = None, rerank: bool = False
    ) -> Answer:
        contexts = self.retrieve(question, mode=mode, k=k, rerank=rerank)
        return self._answerer.answer(question, contexts)
