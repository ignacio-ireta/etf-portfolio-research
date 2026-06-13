# Contributing

Thanks for working on **etf-portfolio-research**. This project follows the
engineering standards in [`docs/engineering_standards.md`](docs/engineering_standards.md).

## Development setup

```bash
uv sync --group dev            # install everything (test, lint, typecheck, security)
uv run pre-commit install      # install git hooks (optional but recommended)
```

The core pipeline is local-first; only `ingest` reaches the network. ML
experiment tracking is optional: `uv sync --extra ml`.

## Workflow

- `main` is the integration branch; create a feature branch for any change.
- Keep commits small and focused, with conventional-commit-style messages:
  - Good: `feat(pipeline): add --resume skip-unchanged`, `fix(report): preserve sheet order`,
    `test(golden): snapshot backtest metrics`, `docs(cli): document --log-level`
  - Avoid: `updates`, `final`, `stuff`, `fix`
- Open a pull request for non-trivial changes; CI must be green.
- Never commit secrets. Copy `.env.example` to `.env` (git-ignored) for local keys.

## Quality gates

Run everything locally before pushing:

```bash
make check        # ruff check --fix, ruff format, mypy, pytest
# or individually:
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
uv run pre-commit run --all-files
```

Expectations:

- **Tests:** new behavior comes with tests. Put fast checks in `tests/unit/`;
  reuse fixtures from `tests/conftest.py`. Mark slow pipeline tests `slow`. See
  [`docs/testing.md`](docs/testing.md).
- **Types:** mypy runs in lenient mode; keep new modules clean and don't add to
  the baseline. Don't silence errors with blanket ignores.
- **Lint/format:** ruff must pass; the formatter is authoritative.
- **Determinism:** outputs must stay reproducible (see
  [`docs/output_contract.md`](docs/output_contract.md)). If you intentionally
  change the metrics contract, regenerate the golden file
  (`UPDATE_GOLDEN=1 uv run pytest tests/golden`) and review the diff.

## Documentation & changelog

Update the relevant `docs/` page and add an entry under `## [Unreleased]` in
`CHANGELOG.md` for user-visible changes. Bump `OUTPUT_SCHEMA_VERSION` and note it
if you change the output contract.

## Releases

Update `__version__` in `src/etf_portfolio/__init__.py`, move the `Unreleased`
changelog section under the new version, and tag `vX.Y.Z`.
