"""rag-knowledge-assistant: a production-shaped Retrieval-Augmented Generation pipeline.

Public surface: build a `RAGPipeline`, ingest documents, ask questions, get cited answers —
and evaluate retrieval quality with a real harness. The interesting parts are hybrid retrieval
(dense + BM25 fused with Reciprocal Rank Fusion), optional reranking, and measurable evaluation.
"""

from rag_assistant.models import Answer, Chunk, RetrievedChunk
from rag_assistant.pipeline import RAGPipeline

__version__ = "0.1.0"

__all__ = ["RAGPipeline", "Answer", "Chunk", "RetrievedChunk", "__version__"]
