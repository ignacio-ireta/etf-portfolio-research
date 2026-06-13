# Testing

How the test suite is organized and run. See `docs/engineering_standards.md`
sections K and L for the strategy behind it.

## Running tests

```bash
uv run pytest                      # full suite
uv run pytest -m "not slow"        # skip slow pipeline/e2e/golden tests
uv run pytest tests/unit           # unit tests only
uv run pytest tests/integration    # integration tests
uv run pytest tests/e2e            # end-to-end CLI tests
uv run pytest --cov=etf_portfolio  # with coverage
make test                          # full suite
make test-fast                     # -m "not slow"
make coverage                      # coverage with missing-line report
```

## Test layout

| Directory | Kind | What it covers |
| --- | --- | --- |
| `tests/unit/` | Unit | Individual functions/classes: returns, optimizer, metrics, constraints, config, errors, atomic writes, pipeline state/runner, CLI exit codes |
| `tests/integration/` | Integration | Multi-component flows on fixtures (e.g. resume re-runs only the changed stage) |
| `tests/e2e/` | End-to-end | Drives `cli.main([...])` like a user (run-all, artifacts, state, resume) |
| `tests/golden/` | Golden | Snapshots the metrics output contract |
| `tests/regression/` | Regression | No-lookahead backtest + ML leakage guards |

## Markers

Registered in `pyproject.toml`: `slow`, `integration`, `e2e`, `golden`,
`property`. The pipeline/e2e/golden tests are marked `slow` (they run the real
backtest + figure export). Use `-m "not slow"` for a fast inner loop; CI runs
everything.

## Fixtures

Shared fixtures live in `tests/conftest.py`:

- `FakePriceProvider` / `make_synthetic_prices()` — deterministic, offline price data.
- `tiny_config` — a valid `AppConfig` for the small VTI/BND/IAU + VT universe.
- `tiny_project` — a ready-to-run project (config + metadata + git repo, working
  directory changed, provider patched) so `cli.main([...])` runs the full pipeline
  with no network.
- An autouse fixture detaches per-run log file handlers so they don't leak.

New tests should reuse these rather than redefining their own provider/helpers.

## Property / fuzz tests

`tests/unit/test_property_invariants.py` uses Hypothesis to check that malformed
data raises a domain error (not a crash), unicode tickers are handled, and
optimizer output stays long-only and fully invested across many inputs.

## Golden files

The golden snapshot (`tests/golden/backtest_metrics_optimized_strategy.json`) is
the optimized-strategy metrics for the fixed synthetic input. After an
**intentional** output change, regenerate it:

```bash
UPDATE_GOLDEN=1 uv run pytest tests/golden
```

Review the diff before committing. The configured strategy is equal-weight, so
the snapshot is platform-stable; a small tolerance absorbs last-bit float noise.

## External dependencies

Tests are fully offline — no market-data network calls (the provider is faked)
and no MLflow backend. `kaleido` renders figures headlessly; if image export is
unavailable, the pipeline falls back to a placeholder PNG and logs a warning, so
report tests still pass.

## CI

GitHub Actions runs the whole suite on Python 3.11–3.13 with coverage. mypy and
pip-audit run as advisory (non-blocking) steps. Nothing is skipped in CI; the
`slow` marker is only for the local fast loop.
