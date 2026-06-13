.PHONY: help sync test test-fast lint format typecheck coverage check precommit run handoff-bundle

help:
	@echo "Available targets:"
	@echo "  sync           Install dependencies using uv"
	@echo "  test           Run the full test suite using pytest"
	@echo "  test-fast      Run tests excluding the slow marker"
	@echo "  lint           Run ruff check"
	@echo "  format         Run ruff format"
	@echo "  typecheck      Run mypy on src (lenient)"
	@echo "  coverage       Run pytest with coverage measurement"
	@echo "  check          Run ruff check --fix, ruff format, mypy (advisory), and pytest"
	@echo "  precommit      Run all pre-commit hooks on all files"
	@echo "  run            Run the full pipeline using configs/base.yaml"
	@echo "  handoff-bundle Generate the handoff archive and manifest"

sync:
	uv sync --group dev

test:
	uv run pytest -q

test-fast:
	uv run pytest -q -m "not slow"

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy src

coverage:
	uv run pytest --cov=etf_portfolio --cov-report=term-missing

check:
	uv run ruff check . --fix
	uv run ruff format .
	uv run mypy src || true  # advisory (non-blocking), consistent with CI; see standards M. Use `make typecheck` to enforce.
	uv run pytest

precommit:
	uv run pre-commit run --all-files

run:
	uv run etf-portfolio run-all --config configs/base.yaml

handoff-bundle:
	uv run python scripts/generate_handoff_bundle.py
