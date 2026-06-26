# Contributing

## Setup
```bash
uv sync --extra dev
uv run pre-commit install
```

## Checks (CI enforces these)
```bash
uv run ruff check .
uv run ruff format .
uv run mypy .
uv run pytest
```

## Conventions
- Type hints everywhere; mypy clean. Tests for new logic; fakes over real network.
- New backend (embedder / store / reranker / answerer): implement the protocol and wire it in
  `factory.py` — touch nothing else.
- Secrets via `.env` (never committed); update `.env.example` when adding a variable.
- Conventional-commit messages (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).
