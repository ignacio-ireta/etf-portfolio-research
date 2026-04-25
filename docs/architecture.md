# Architecture

This project is a code-driven ETF research pipeline. The system of record is
the CLI, configuration, source code, tests, generated artifacts, and run
records; notebooks are exploratory only.

## Entry Points

- `uv sync --group dev` creates the development environment from `pyproject.toml`
  and the authoritative `uv.lock`.
- `uv run etf-portfolio run-all --config configs/base.yaml` runs the full
  pipeline.
- `make run` is a convenience wrapper for the same `run-all` command.
- `uv run python scripts/generate_handoff_bundle.py` builds the handoff archive.

The package entrypoint is declared in `pyproject.toml` as
`etf-portfolio = "etf_portfolio.cli:main"`. Command parsing and stage
orchestration live in `src/etf_portfolio/cli.py`.

## Pipeline Flow

1. `ingest` reads the configured universe and downloads adjusted ETF prices.
   The default provider is `yfinance`.
2. `validate` checks price coverage and quality, then writes validated prices.
3. `features` converts validated prices into return series.
4. `optimize` estimates expected return and covariance, applies configured
   constraints, and writes optimizer outputs.
5. `backtest` runs the walk-forward strategy, benchmarks, transaction-cost
   assumptions, metrics, reports, figures, workbooks, and run record.
6. `report` can regenerate reporting artifacts from existing pipeline outputs.

`run-all` executes these stages in order using one run id so logs, artifacts,
metrics, and run records can be tied back to the same execution.

## Configuration

Base configuration is `configs/base.yaml`. Additional universe and constraint
examples live under `configs/`.

Important configuration areas:

- `universe` selects tradable tickers and metadata filters.
- `optimization` selects the objective, risk model, expected-return estimator,
  and optimizer constraints.
- `rebalance` defines the backtest cadence and realized-holdings behavior.
- `benchmark` defines primary and secondary benchmark comparisons.
- `report` controls generated report content and assumptions.

The active optimizer objective is `optimization.active_objective`; it is used by
`optimize`, `backtest`, and `run-all`.

## Source Layout

- `src/etf_portfolio/data/`: price ingestion, provider adapters, validation, and
  schemas.
- `src/etf_portfolio/features/`: return and estimator utilities.
- `src/etf_portfolio/optimization/`: objectives, constraints, efficient
  frontier, and optimizer implementation.
- `src/etf_portfolio/backtesting/`: walk-forward engine, rebalancing, costs,
  attribution, and metrics.
- `src/etf_portfolio/risk/`: exposure, drawdown, stress, and attribution
  analysis.
- `src/etf_portfolio/reporting/`: Plotly figures, Excel tables, HTML report, and
  report-bundle generation.
- `src/etf_portfolio/ml/`: experimental ML dataset, training, evaluation,
  registry, and governance gates.
- `src/etf_portfolio/tracking.py`: run-record and artifact hashing helpers.

## Data And Artifacts

Generated data:

- `data/raw/prices.parquet`: provider output from ingestion.
- `data/processed/prices_validated.parquet`: validated price matrix.
- `data/processed/returns.parquet`: aligned return matrix used by optimizer and
  backtest stages.

Generated reporting artifacts:

- `reports/html/latest_report.html`
- `reports/html/backtest_report.html`
- `reports/html/frontier.html`
- `reports/excel/optimized_portfolios.xlsx`
- `reports/excel/portfolio_results.xlsx`
- `reports/figures/*.png`
- `reports/metrics/backtest_metrics.json`
- `reports/runs/*.json`

Run records hash output artifacts and capture the effective configuration,
source commit, data version, and generated file locations.

## Handoff Bundle Policy

`scripts/generate_handoff_bundle.py` creates `handoff_bundle.zip` and
`handoff/included_files.txt`.

The bundle includes:

- source, tests, docs, configs, `Makefile`, `pyproject.toml`, and `uv.lock`
- `data/metadata/etf_universe.csv`
- `data/processed/*.parquet`
- latest metrics, reports, figures, workbooks, and run records
- handoff command logs under `handoff/*.txt`
- the bundle-generation script

The bundle excludes `data/raw/prices.parquet`, virtual environments, caches,
MLflow runs, and obvious secret paths. Processed parquet files are included
because they are small and allow recipients to inspect or rerun report/backtest
stages without an immediate data-provider call.

## Reproducibility Policy

`uv.lock` is authoritative. The supported Python range is 3.11 through 3.13;
`pyproject.toml` caps support below 3.14, and CI tests 3.11, 3.12, and 3.13.

The repository does not currently define a DVC workflow. Reproduction uses the
CLI and Makefile workflows documented in the README and runbook.
