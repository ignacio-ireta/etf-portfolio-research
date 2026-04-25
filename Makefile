.PHONY: help sync test lint format check run handoff-bundle

help:
	@echo "Available targets:"
	@echo "  sync           Install dependencies using uv"
	@echo "  test           Run tests using pytest"
	@echo "  lint           Run ruff check"
	@echo "  format         Run ruff format"
	@echo "  check          Run ruff check --fix, ruff format, and pytest"
	@echo "  run            Run the full pipeline using configs/base.yaml"
	@echo "  handoff-bundle Generate the handoff archive and manifest"

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
