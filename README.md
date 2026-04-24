# etf-portfolio-research

Reproducible ETF portfolio research pipeline for universe definition, data ingestion, optimization, walk-forward backtesting, reporting, and tracked experiments.

This project is a research tool, not financial advice. It is designed to make ETF portfolio assumptions visible, reproducible, and easier to challenge.

## Start Here

If you are new to the project or to portfolio analytics, start with the beginner learning path:

1. [Start Here](docs/start_here.md)
2. [Glossary](docs/glossary.md)
3. [Metric Dictionary](docs/metric_dictionary.md)
4. [Trust And Safety](docs/trust_and_safety.md)
5. [Guided Examples](docs/guided_examples.md)
6. [How To Read The Report](docs/how_to_read_the_report.md)
7. [Interpretation Guide](docs/interpretation_guide.md)
8. [Assumptions And Limitations](docs/assumptions_and_limitations.md)

The main generated output to read first is:

```text
reports/html/latest_report.html
```

## Quick Start

This project uses `uv` for environment and dependency management.

```bash
uv sync --group dev
```

Run the full pipeline:

```bash
uv run etf-portfolio run-all --config configs/base.yaml
```

Optional DVC reproduction:

```bash
uv run dvc repro
```

## CLI Reference

CLI entrypoint:

```bash
uv run etf-portfolio ingest --config configs/base.yaml
uv run etf-portfolio validate --config configs/base.yaml
uv run etf-portfolio features --config configs/base.yaml
uv run etf-portfolio optimize --config configs/base.yaml
uv run etf-portfolio backtest --config configs/base.yaml
uv run etf-portfolio report --config configs/base.yaml
uv run etf-portfolio run-all --config configs/base.yaml
```

Make targets:

```bash
make sync
make lint
make test
make run
make repro
```

## Pipeline Outputs

Core outputs produced by the code-driven pipeline:

- `data/raw/prices.parquet`
- `data/processed/prices_validated.parquet`
- `data/processed/returns.parquet`
- `reports/html/latest_report.html`
- `reports/excel/optimized_portfolios.xlsx`
- `reports/excel/portfolio_results.xlsx`
- `reports/figures/*.png`
- `reports/metrics/backtest_metrics.json`
- `reports/runs/*.json`

## Example Output

Expected report bundle after `run-all` or `backtest`:

```text
reports/html/latest_report.html
reports/excel/optimized_portfolios.xlsx
reports/excel/portfolio_results.xlsx
reports/figures/efficient_frontier.png
reports/figures/cumulative_returns.png
reports/figures/drawdown.png
reports/metrics/backtest_metrics.json
```

The HTML report is generated from pipeline outputs, not notebook state.

## Project Assumptions

- Base configuration is in [configs/base.yaml](configs/base.yaml).
- Universe is ETF-based and currently USD-denominated.
- Prototype price provider is `yfinance`; this is acceptable for research, not production-grade market data.
- Expected returns are currently historical means.
- Risk model is currently `sample` or `ledoit_wolf`, depending on config.
- `optimization.active_objective` is the single optimizer objective the pipeline uses for `optimize`, `backtest`, and `run-all`. Run records in `reports/runs/*.json` persist the effective `optimization_method`.
- Backtests are walk-forward and use only trailing data before each rebalance date.
- `backtest.start_date` and `backtest.end_date` are applied to aligned asset and benchmark return series before eligible rebalance dates are generated.
- In `contribution_only` mode, optimizer targets always honor configured caps. Realized holdings use `rebalance.realized_constraint_policy`: `report_drift` keeps HODL/no-sell behavior and reports drift violations, while `enforce_hard` permits sell-based fallback rebalances when realized caps are breached.
- The report frontier uses the same constraint bundle as optimization and walk-forward backtesting: asset classes, asset-class bounds, ticker bounds, bond floor, and expense-ratio inputs are passed through consistently.
- Backtest reporting separates optimizer targets from realized holdings. The Excel bundle includes `optimizer_target_portfolio` and `latest_realized_portfolio`, plus realized-constraint warnings when contribution-only drift breaches configured caps.
- Transaction costs and slippage are modeled, but execution is still simplified relative to live trading.
- Reporting is code-generated; notebooks are research interfaces only.
- Metrics JSON is written as strict JSON: non-finite values are sanitized, NumPy scalars are unwrapped, pandas timestamps are serialized, and output is written with `allow_nan=False`.
- ML exists in the repo, but governance gates decide whether a model is eligible for portfolio use.

## Documentation

- [Start Here](docs/start_here.md)
- [Glossary](docs/glossary.md)
- [Metric Dictionary](docs/metric_dictionary.md)
- [Trust And Safety](docs/trust_and_safety.md)
- [Guided Examples](docs/guided_examples.md)
- [How To Read The Report](docs/how_to_read_the_report.md)
- [Interpretation Guide](docs/interpretation_guide.md)
- [Assumptions And Limitations](docs/assumptions_and_limitations.md)
- [Methodology](docs/methodology.md)
- [Data Dictionary](docs/data_dictionary.md)
- [Runbook](docs/runbook.md)
- [Model Card](docs/model_card.md)
- [Research Log](docs/research_log.md)
