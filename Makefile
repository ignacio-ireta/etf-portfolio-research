.PHONY: sync test lint format check run repro handoff-bundle

sync:
	uv sync --group dev

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

check:
	uv run ruff check . --fix
	uv run ruff format .
	uv run pytest

run:
	uv run etf-portfolio

repro:
	uv run dvc repro

handoff-bundle:
	uv run python scripts/generate_handoff_bundle.py
