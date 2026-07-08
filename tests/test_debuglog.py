"""LLM_DEBUG tracing: silent by default, request/response blocks on stderr when enabled.

Everything here runs through the offline fakes (hashing embedder + fake answerer), proving the
debug feature needs no API key.
"""

from __future__ import annotations

import pytest

from rag_assistant.debuglog import debug_enabled, log_block
from tests.conftest import make_pipeline


def test_debug_enabled_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_DEBUG", raising=False)
    assert not debug_enabled()
    for falsy in ("", "0", "false", "FALSE", "False"):
        monkeypatch.setenv("LLM_DEBUG", falsy)
        assert not debug_enabled()
    for truthy in ("1", "true", "yes", "on"):
        monkeypatch.setenv("LLM_DEBUG", truthy)
        assert debug_enabled()


def test_silent_when_unset(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("LLM_DEBUG", raising=False)
    pipeline = make_pipeline()  # ingestion embeds every chunk
    answer = pipeline.ask("How are two ranked lists combined?", mode="hybrid")
    assert answer.text
    captured = capsys.readouterr()
    assert "AI REQUEST" not in captured.err
    assert "AI RESPONSE" not in captured.err
    assert captured.err == ""


def test_blocks_printed_to_stderr_when_set(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LLM_DEBUG", "1")
    pipeline = make_pipeline()
    answer = pipeline.ask("How are two ranked lists combined?", mode="hybrid")
    assert answer.text
    captured = capsys.readouterr()
    # Request/response pairs land on stderr, never stdout.
    assert "=== AI REQUEST" in captured.err
    assert "=== AI RESPONSE" in captured.err
    assert captured.out == ""
    # Both offline fakes are labelled, so learners can tell what ran with no key.
    assert "offline fake embedder" in captured.err
    assert "offline fake answerer" in captured.err
    # The answerer request shows the grounding prompt and the question.
    assert "system: You are a precise question-answering assistant." in captured.err
    assert "user: How are two ranked lists combined?" in captured.err
    # The embedder response reports shape only — raw vectors are never dumped.
    assert "dimensions: 128" in captured.err


def test_long_fields_are_truncated(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LLM_DEBUG", "1")
    log_block("AI REQUEST (test)", body="x" * 5000)
    err = capsys.readouterr().err
    assert "... [truncated]" in err
    assert "x" * 2001 not in err


def test_log_block_is_noop_when_disabled(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LLM_DEBUG", "0")
    log_block("AI REQUEST (test)", user="hello")
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""
