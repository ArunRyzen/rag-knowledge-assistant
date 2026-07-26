"""Answer generation grounded in retrieved context, via Gemini.

The generator's job is narrow and safety-critical: answer **only** from the supplied contexts and
cite them, or say it doesn't know. That instruction (plus passing numbered contexts) is what turns
retrieval into a trustworthy, attributable answer instead of a confident hallucination.

Citations are extracted from the `[n]` markers the model writes, so `Answer.citations` reflects
the passages the model *actually used* — not merely everything retrieval handed it.
"""

from __future__ import annotations

import re
from typing import Protocol

from rag_assistant.debuglog import debug_enabled, log_block
from rag_assistant.models import Answer, Citation, RetrievedChunk

# The system prompt. The "ONLY ... provided" and "say you don't know" clauses are the
# anti-hallucination guardrails; the citation clause makes answers checkable.
_SYSTEM = (
    "You are a precise question-answering assistant. Answer ONLY using the numbered context "
    "passages provided. If the answer is not in the context, say you don't know. Be concise and "
    "cite the passage numbers you used, e.g. [1], [2]."
)

_CITATION_MARKER = re.compile(r"\[(\d+)\]")


def _format_contexts(contexts: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{i}] (doc: {c.chunk.doc_id}) {c.chunk.text}" for i, c in enumerate(contexts, start=1)
    )


def _build_prompt(question: str, contexts: list[RetrievedChunk]) -> str:
    return f"Context passages:\n{_format_contexts(contexts)}\n\nQuestion: {question}"


def extract_citations(text: str, contexts: list[RetrievedChunk]) -> list[Citation]:
    """Turn the `[n]` markers in the answer into `Citation`s for the passages they point at.

    Only passages the model referenced become citations; out-of-range markers (a hallucinated
    `[9]` when 5 passages were sent) are ignored. If the model cited nothing, fall back to
    citing every context so provenance is never silently empty.
    """
    cited_numbers = {int(m) for m in _CITATION_MARKER.findall(text)}
    valid = [n for n in sorted(cited_numbers) if 1 <= n <= len(contexts)]
    chosen = [contexts[n - 1] for n in valid] if valid else contexts
    return [
        Citation(chunk_id=c.chunk.id, doc_id=c.chunk.doc_id, quote=c.chunk.text[:160])
        for c in chosen
    ]


def _log_answer_request(
    label: str, system: str, question: str, contexts: list[RetrievedChunk]
) -> None:
    # Debug tracing (LLM_DEBUG=1): the exact system prompt, question, and context previews the
    # answerer sees. API keys are never logged.
    if not debug_enabled():
        return
    context = "\n".join(f"[{i}] {c.chunk.text[:200]}..." for i, c in enumerate(contexts, start=1))
    log_block(f"AI REQUEST ({label})", system=system, user=question, context=context)


def _log_answer_response(label: str, text: str) -> None:
    if not debug_enabled():
        return
    log_block(f"AI RESPONSE ({label})", text=text)


class Answerer(Protocol):
    def answer(self, question: str, contexts: list[RetrievedChunk]) -> Answer: ...


def generate_text(
    *, model: str, max_tokens: int, api_key: str | None, system: str, prompt: str
) -> str:
    """One Gemini text-generation call. Shared by the answerer and the chat agents (chat.py)."""
    # The system instruction and token cap ride along in a config object.
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=0,  # deterministic-ish: we want grounded answers, not creativity
        ),
    )
    return response.text or ""


class GeminiAnswerer:
    """Grounded, cited answer synthesis via the Gemini API."""

    def __init__(self, *, model: str, max_tokens: int, api_key: str | None = None) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._api_key = api_key

    def answer(self, question: str, contexts: list[RetrievedChunk]) -> Answer:
        if not contexts:
            return Answer(question=question, text="I don't know — no relevant context found.")
        label = f"gemini/{self._model}"
        _log_answer_request(label, _SYSTEM, question, contexts)
        text = generate_text(
            model=self._model,
            max_tokens=self._max_tokens,
            api_key=self._api_key,
            system=_SYSTEM,
            prompt=_build_prompt(question, contexts),
        )
        _log_answer_response(label, text)
        return Answer(
            question=question,
            text=text,
            citations=extract_citations(text, contexts),
            contexts=contexts,
        )
