.PHONY: sync test lint format check run handoff-bundle

sync:
	uv sync --group dev

test:
	uv run pytest -q

lint:
	uv run ruff check .

format:
	uv run ruff format .

check:
	uv run ruff check . --fix
	uv run ruff format .
	uv run pytest

run:
	uv run etf-portfolio run-all --config configs/base.yaml

handoff-bundle:
	uv run python scripts/generate_handoff_bundle.py
