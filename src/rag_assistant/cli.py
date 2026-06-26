"""Command-line interface.

Commands are self-contained (ingest-then-act in one process) so the zero-infra in-memory store
works without persistence between runs. For a persistent corpus, use the pgvector backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from rag_assistant.config import load_settings
from rag_assistant.corpus import load_corpus
from rag_assistant.evaluation import GoldenItem, compare_modes
from rag_assistant.factory import build_pipeline
from rag_assistant.pipeline import RAGPipeline
from rag_assistant.sample_data import GOLDEN

app = typer.Typer(help="RAG knowledge assistant: ingest, ask, and evaluate.", no_args_is_help=True)


def _ingest_corpus(pipeline: RAGPipeline, data: Path | None) -> int:
    total = 0
    for doc_id, text in load_corpus(data):
        total += pipeline.ingest(doc_id, text)
    return total


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="The question to answer.")],
    data: Annotated[Path | None, typer.Option(help="Folder/file of docs (.md/.txt).")] = None,
    mode: Annotated[str, typer.Option(help="dense | sparse | hybrid")] = "hybrid",
    rerank: Annotated[
        bool, typer.Option(help="Apply the reranker (needs the rerank extra).")
    ] = False,
) -> None:
    """Ingest the corpus (bundled sample by default), then answer the question."""
    pipeline = build_pipeline(load_settings())
    n = _ingest_corpus(pipeline, data)
    typer.echo(f"Ingested {n} chunks.", err=True)

    result = pipeline.ask(question, mode=mode, rerank=rerank)
    typer.echo(result.text)
    if result.contexts:
        typer.echo("\nSources:", err=True)
        for i, ctx in enumerate(result.contexts, start=1):
            typer.echo(f"  [{i}] {ctx.chunk.doc_id} (score={ctx.score:.3f})", err=True)


@app.command(name="eval")
def evaluate(
    data: Annotated[
        Path | None, typer.Option(help="Docs folder; defaults to sample corpus.")
    ] = None,
    k: Annotated[int, typer.Option(help="Cut-off for recall@k / MRR.")] = 5,
) -> None:
    """Compare retrieval modes (dense / sparse / hybrid / +rerank) on the golden set."""
    pipeline = build_pipeline(load_settings())
    n = _ingest_corpus(pipeline, data)
    typer.echo(f"Ingested {n} chunks. Evaluating on {len(GOLDEN)} golden questions...\n", err=True)

    dataset = [GoldenItem(**item) for item in GOLDEN]  # type: ignore[arg-type]
    for metrics in compare_modes(pipeline.retriever, dataset, k=k):
        typer.echo(metrics.as_row())


if __name__ == "__main__":
    app()
