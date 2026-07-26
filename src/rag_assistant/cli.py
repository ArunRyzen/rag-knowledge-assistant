"""Command-line interface.

The flow mirrors production RAG systems: `rag ingest` is the run-once (or run-on-change) step
that chunks, embeds, and upserts documents into the vector store; `rag ask` and `rag eval` are
the query-time commands.

One asymmetry to understand: with VECTOR_STORE=pinecone the dense vectors persist in the cloud,
but the BM25 index is in-memory — so ask/eval re-chunk the corpus locally (no API calls) to
rebuild BM25, while dense search hits the vectors already in Pinecone. With VECTOR_STORE=memory
nothing persists, so every command ingests fully (embedding calls included) each run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from rag_assistant.config import load_settings
from rag_assistant.corpus import load_corpus
from rag_assistant.evaluation import GoldenItem, compare_modes
from rag_assistant.factory import build_chat, build_pipeline
from rag_assistant.golden import GOLDEN
from rag_assistant.pipeline import RAGPipeline

app = typer.Typer(help="RAG knowledge assistant: ingest, ask, and evaluate.", no_args_is_help=True)


def _ingest_corpus(pipeline: RAGPipeline, data: Path | None, *, dense: bool) -> int:
    total = 0
    for doc_id, text in load_corpus(data):
        total += pipeline.ingest(doc_id, text, dense=dense)
    return total


def _prepare(data: Path | None) -> RAGPipeline:
    """Build the pipeline and make it queryable.

    Persistent store (pinecone): rebuild only the local BM25 side; dense vectors are already
    in the index from `rag ingest`. In-memory store: full ingest, embeddings included.
    """
    settings = load_settings()
    pipeline = build_pipeline(settings)
    persistent = settings.vector_store == "pinecone"
    n = _ingest_corpus(pipeline, data, dense=not persistent)
    if persistent and pipeline.vector_count == 0:
        typer.echo(
            "Warning: the Pinecone index is empty — dense/hybrid search will find nothing. "
            "Run `rag ingest` first.",
            err=True,
        )
    typer.echo(f"Prepared {n} chunks.", err=True)
    return pipeline


@app.command()
def ingest(
    data: Annotated[
        Path | None, typer.Option(help="Folder/file of docs (.md/.txt/.pdf).")
    ] = None,
    reset: Annotated[
        bool, typer.Option(help="Wipe the vector store first (use when the corpus CHANGED).")
    ] = False,
) -> None:
    """Chunk, embed, and index the corpus (the `data/` folder by default) into the vector store.

    With VECTOR_STORE=pinecone this persists — run it once, then `rag ask` freely. Freshly
    upserted records can take a few seconds to become searchable (eventual consistency).
    Re-running upserts by chunk id (safe); pass --reset when documents were REMOVED or renamed.
    """
    settings = load_settings()
    pipeline = build_pipeline(settings)
    if reset:
        pipeline.reset_store()
        typer.echo("Cleared the vector store.", err=True)
    n = _ingest_corpus(pipeline, data, dense=True)
    typer.echo(f"Ingested {n} chunks into the '{settings.vector_store}' store.")
    if settings.vector_store == "pinecone":
        typer.echo(
            f"Pinecone index '{settings.pinecone_index}' now holds ~{pipeline.vector_count} "
            "vectors (fresh upserts can take a few seconds to appear)."
        )


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="The question to answer.")],
    data: Annotated[Path | None, typer.Option(help="Folder/file of docs (.md/.txt).")] = None,
    mode: Annotated[str, typer.Option(help="dense | sparse | hybrid")] = "hybrid",
    rerank: Annotated[
        bool, typer.Option(help="Apply the reranker (needs the rerank extra).")
    ] = False,
) -> None:
    """Answer a question over the corpus (the `data/` folder by default)."""
    pipeline = _prepare(data)
    result = pipeline.ask(question, mode=mode, rerank=rerank)
    typer.echo(result.text)
    if result.contexts:
        typer.echo("\nSources:", err=True)
        for i, ctx in enumerate(result.contexts, start=1):
            typer.echo(f"  [{i}] {ctx.chunk.doc_id} (score={ctx.score:.3f})", err=True)


@app.command()
def chat(
    data: Annotated[Path | None, typer.Option(help="Folder/file of docs (.md/.txt/.pdf).")] = None,
    once: Annotated[
        str | None, typer.Option(help="Ask a single message non-interactively and exit.")
    ] = None,
) -> None:
    """Multi-turn guarded chatbot: input guardrail → condense → RAG answer → grounding check."""
    settings = load_settings()
    pipeline = build_pipeline(settings)
    persistent = settings.vector_store == "pinecone"
    _ingest_corpus(pipeline, data, dense=not persistent)
    session = build_chat(settings, pipeline)

    def _respond(message: str) -> None:
        # Each turn makes several model calls (guard + answer + check, sometimes web), and the
        # Gemini FREE tier allows only ~5/minute — so rate limits are normal, not fatal.
        try:
            turn = session.turn(message)
        except Exception as exc:  # noqa: BLE001 - surface any provider error politely
            if "RESOURCE_EXHAUSTED" in str(exc) or "429" in str(exc):
                typer.echo(
                    "bot> The free Gemini quota is briefly exhausted (each chat turn uses "
                    "several model calls). Wait a minute and ask again.",
                )
                return
            raise
        typer.echo(f"bot> {turn.reply}")
        if turn.source == "book" and turn.answer:
            docs = ", ".join(sorted({c.doc_id for c in turn.answer.citations}))
            typer.echo(f"     (book sources: {docs})", err=True)
        elif turn.source == "web":
            typer.echo("     [book had no answer — web search agent replied]", err=True)
        if not turn.allowed:
            typer.echo("     [input guardrail refused this message]", err=True)
        elif turn.grounded is False:
            typer.echo("     [grounding checker vetoed the book answer]", err=True)

    if once is not None:
        _respond(once)
        return

    typer.echo("Guarded chat — type a question, or 'exit' to quit.", err=True)
    while True:
        try:
            message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not message:
            continue
        if message.lower() in {"exit", "quit"}:
            break
        _respond(message)


@app.command(name="eval")
def evaluate(
    data: Annotated[
        Path | None, typer.Option(help="Docs folder; defaults to the `data/` folder.")
    ] = None,
    k: Annotated[int, typer.Option(help="Cut-off for recall@k / MRR.")] = 5,
) -> None:
    """Compare retrieval modes (dense / sparse / hybrid / +rerank) on the golden set."""
    pipeline = _prepare(data)
    typer.echo(f"Evaluating on {len(GOLDEN)} golden questions...\n", err=True)
    dataset = [GoldenItem(**item) for item in GOLDEN]  # type: ignore[arg-type]
    for metrics in compare_modes(pipeline.retriever, dataset, k=k):
        typer.echo(metrics.as_row())


if __name__ == "__main__":
    app()
