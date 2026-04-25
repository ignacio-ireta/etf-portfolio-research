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

- `src/etf_portfolio/data/`: price ingestion, provider adapters (yfinance, tiingo), validation, and schemas.
- `src/etf_portfolio/features/`: return calculations and statistical estimators (mean, covariance).
- `src/etf_portfolio/optimization/`: objectives (Sharpe, variance, etc.), constraints (asset class, ticker bounds), efficient frontier, and the SLSQP optimizer integration.
- `src/etf_portfolio/backtesting/`: walk-forward engine, contribution-only rebalancing, transaction costs, and portfolio metrics.
- `src/etf_portfolio/risk/`: exposure analysis, drawdown, stress testing, and risk attribution.
- `src/etf_portfolio/reporting/`: Plotly figures, Excel tables, HTML report generation, and the report-bundle utility.
- `src/etf_portfolio/ml/`: experimental ML dataset creation, training, evaluation, registry, and governance gates.
- `src/etf_portfolio/tracking.py`: utility for run-record persistence and artifact hashing.

## Data and Artifacts

- `data/raw/`: raw data from providers (excluded from handoff).
- `data/processed/`: validated and aligned data used for research (included in handoff).
- `reports/html/`: interactive research reports.
- `reports/excel/`: structured workbooks for weights and backtest results.
- `reports/figures/`: static PNG plots for the reports.
- `reports/metrics/`: aggregated summary metrics in JSON format.
- `reports/runs/`: detailed records of every pipeline execution.

## Handoff Bundle Policy

`scripts/generate_handoff_bundle.py` creates `handoff_bundle.zip` and `handoff/included_files.txt`.

The bundle intentionally includes:
- source, tests, docs, configs, `Makefile`, `pyproject.toml`, and `uv.lock`
- `data/metadata/etf_universe.csv`
- `data/processed/*.parquet`
- latest metrics, reports, figures, workbooks, and run records
- handoff command logs under `handoff/*.txt`
- the bundle-generation script

The bundle excludes `data/raw/prices.parquet`, virtual environments, caches, MLflow runs, and obvious secret paths.

## Reproducibility Policy

- **Authoritative Lockfile**: `uv.lock` is checked into source control and is the single source of truth for the environment.
- **Python Versioning**: The project supports Python 3.11 through 3.13. The CI environment tests all three versions to ensure compatibility.
- **DVC (Data Version Control)**: Currently, the project does not use DVC for data tracking. Reproducibility is maintained via the code-driven pipeline and the CLI.
- **Artifact Hashing**: Each run record in `reports/runs/` captures hashes of the input configuration and output artifacts.
