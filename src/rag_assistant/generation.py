"""Answer generation grounded in retrieved context.

The generator's job is narrow and safety-critical: answer **only** from the supplied contexts and
cite them, or say it doesn't know. That instruction (plus passing numbered contexts) is what turns
retrieval into a trustworthy, attributable answer instead of a confident hallucination.

`FakeAnswerer` makes the pipeline testable with no model call; `LLMAnswerer` calls Gemini,
Anthropic, or OpenAI for real synthesis.
"""

from __future__ import annotations

from typing import Protocol

from rag_assistant.debuglog import debug_enabled, log_block
from rag_assistant.errors import GenerationError
from rag_assistant.models import Answer, Citation, RetrievedChunk

# The system prompt every live provider gets. The "ONLY ... provided" and "say you don't know"
# clauses are the anti-hallucination guardrails; the citation clause makes answers checkable.
_SYSTEM = (
    "You are a precise question-answering assistant. Answer ONLY using the numbered context "
    "passages provided. If the answer is not in the context, say you don't know. Be concise and "
    "cite the passage numbers you used, e.g. [1], [2]."
)


def _format_contexts(contexts: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{i}] (doc: {c.chunk.doc_id}) {c.chunk.text}" for i, c in enumerate(contexts, start=1)
    )


def _build_prompt(question: str, contexts: list[RetrievedChunk]) -> str:
    return f"Context passages:\n{_format_contexts(contexts)}\n\nQuestion: {question}"


def _citations(contexts: list[RetrievedChunk]) -> list[Citation]:
    return [
        Citation(chunk_id=c.chunk.id, doc_id=c.chunk.doc_id, quote=c.chunk.text[:160])
        for c in contexts
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


class FakeAnswerer:
    """Deterministic answerer for tests and offline demos — no network."""

    def answer(self, question: str, contexts: list[RetrievedChunk]) -> Answer:
        _log_answer_request("offline fake answerer", _SYSTEM, question, contexts)
        if not contexts:
            text = "I don't know — no relevant context found."
            _log_answer_response("offline fake answerer", text)
            return Answer(question=question, text=text)
        top = contexts[0].chunk
        text = f"Based on {len(contexts)} passage(s), see doc '{top.doc_id}'. [1]"
        _log_answer_response("offline fake answerer", text)
        return Answer(
            question=question, text=text, citations=_citations(contexts), contexts=contexts
        )


class LLMAnswerer:
    """Real synthesis via Gemini, Anthropic, or OpenAI.

    All three providers get the exact same system instruction and prompt — only the SDK call
    differs — so answers stay grounded and cited no matter which key you have.
    """

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        max_tokens: int,
        api_key: str | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._max_tokens = max_tokens
        self._api_key = api_key

    def _generate(self, system: str, prompt: str) -> str:
        if self._provider == "gemini":
            # Google's SDK: the system instruction and token cap ride along in a config object.
            from google import genai
            from google.genai import types

            gclient = genai.Client(api_key=self._api_key)
            response = gclient.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=self._max_tokens,
                    temperature=0,  # deterministic-ish: we want grounded answers, not creativity
                ),
            )
            return response.text or ""
        if self._provider == "anthropic":
            from anthropic import Anthropic

            aclient = Anthropic(api_key=self._api_key)
            resp = aclient.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in resp.content if b.type == "text")
        if self._provider == "openai":
            from openai import OpenAI

            oclient = OpenAI(api_key=self._api_key)
            completion = oclient.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=0,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            return completion.choices[0].message.content or ""
        raise GenerationError(f"Unknown generation provider '{self._provider}'.")

    def answer(self, question: str, contexts: list[RetrievedChunk]) -> Answer:
        if not contexts:
            return Answer(question=question, text="I don't know — no relevant context found.")
        label = f"{self._provider}/{self._model}"
        _log_answer_request(label, _SYSTEM, question, contexts)
        text = self._generate(_SYSTEM, _build_prompt(question, contexts))
        _log_answer_response(label, text)
        return Answer(
            question=question, text=text, citations=_citations(contexts), contexts=contexts
        )
