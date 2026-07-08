"""LLM_DEBUG tracing: silent by default, request/response blocks on stderr when enabled.

Everything here runs through the offline fakes (hashing embedder + fake answerer), proving the
debug feature needs no API key.

Two hermeticity rules keep these tests honest no matter what the developer's machine looks like:

- ``debug_enabled()`` caches its answer (``functools.lru_cache``), so every test that changes
  ``LLM_DEBUG`` calls ``debug_enabled.cache_clear()`` afterwards.
- ``debug_enabled()`` also reads ``.env`` from the current directory, so tests that need the
  "nothing is set" state chdir into an empty ``tmp_path`` first.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_assistant.debuglog import debug_enabled, log_block
from tests.conftest import make_pipeline


def test_debug_enabled_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)  # no `.env` here, so "unset" really means unset
    monkeypatch.delenv("LLM_DEBUG", raising=False)
    debug_enabled.cache_clear()
    assert not debug_enabled()
    for falsy in ("", "0", "false", "FALSE", "False"):
        monkeypatch.setenv("LLM_DEBUG", falsy)
        debug_enabled.cache_clear()
        assert not debug_enabled()
    for truthy in ("1", "true", "yes", "on"):
        monkeypatch.setenv("LLM_DEBUG", truthy)
        debug_enabled.cache_clear()
        assert debug_enabled()


def test_env_file_enables_debug(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """With no env var at all, an ``LLM_DEBUG=1`` line in `.env` switches tracing on."""
    (tmp_path / ".env").write_text("LLM_DEBUG=1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_DEBUG", raising=False)
    debug_enabled.cache_clear()
    assert debug_enabled()


def test_env_var_beats_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A real environment variable — even a falsy one — always overrides the `.env` file."""
    (tmp_path / ".env").write_text("LLM_DEBUG=1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_DEBUG", "0")
    debug_enabled.cache_clear()
    assert not debug_enabled()
    # And the reverse: env var "1" wins over a falsy file value.
    (tmp_path / ".env").write_text("LLM_DEBUG=0\n", encoding="utf-8")
    monkeypatch.setenv("LLM_DEBUG", "1")
    debug_enabled.cache_clear()
    assert debug_enabled()


def test_disabled_when_neither_source_set(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No env var and no `.env` file (or one without LLM_DEBUG) means tracing stays off."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_DEBUG", raising=False)
    debug_enabled.cache_clear()
    assert not debug_enabled()
    # A `.env` that never mentions LLM_DEBUG changes nothing.
    (tmp_path / ".env").write_text("GEMINI_API_KEY=\n", encoding="utf-8")
    debug_enabled.cache_clear()
    assert not debug_enabled()


def test_silent_when_unset(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)  # keep a developer's real `.env` out of the picture
    monkeypatch.delenv("LLM_DEBUG", raising=False)
    debug_enabled.cache_clear()
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
    debug_enabled.cache_clear()
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
    debug_enabled.cache_clear()
    log_block("AI REQUEST (test)", body="x" * 5000)
    err = capsys.readouterr().err
    assert "... [truncated]" in err
    assert "x" * 2001 not in err


def test_log_block_is_noop_when_disabled(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LLM_DEBUG", "0")
    debug_enabled.cache_clear()
    log_block("AI REQUEST (test)", user="hello")
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""
