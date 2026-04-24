# Runbook

## How To Reproduce Results

Install dependencies:

```bash
uv sync --group dev
```

Run the full pipeline:

```bash
uv run etf-portfolio run-all --config configs/base.yaml
```

Or reproduce with DVC:

```bash
uv run dvc repro
```

Primary generated outputs:

- `reports/html/latest_report.html`
- `reports/excel/optimized_portfolios.xlsx`
- `reports/excel/portfolio_results.xlsx`
- `reports/metrics/backtest_metrics.json`
- `reports/runs/*.json`

For reproducible beginner walkthroughs using the current generated artifacts, see [Guided Examples](guided_examples.md).

Current workbook/report layout:

- `optimized_portfolios.xlsx` captures the latest optimizer output for the active objective.
- `portfolio_results.xlsx` includes separate `optimizer_target_portfolio` and `latest_realized_portfolio` sheets.
- `portfolio_results.xlsx` may also include `optimizer_target_history` and `realized_constraint_warnings` when the backtest generated them.
- `reports/runs/*.json` records the effective `optimization_method`; verify this when changing `optimization.active_objective`.

## How To Refresh Data

Refresh only ingestion and downstream stages:

```bash
uv run etf-portfolio ingest --config configs/base.yaml
uv run etf-portfolio validate --config configs/base.yaml
uv run etf-portfolio features --config configs/base.yaml
uv run etf-portfolio backtest --config configs/base.yaml
```

If you want the entire pipeline rebuilt from stage dependencies, use:

```bash
uv run dvc repro
```

## How To Debug Failed Optimization

Check structured logs first. Failures should be emitted with reason, stage, `run_id`, and optimizer status.

Common checks:

- confirm `data/processed/returns.parquet` exists and has enough history
- confirm benchmark ticker exists in the ingested data
- confirm `default_max_weight_per_etf` is feasible for the universe size
- confirm ticker and asset-class bounds do not conflict
- confirm minimum bond exposure is satisfiable by the current universe
- confirm no asset column is entirely null after alignment
- confirm `backtest.start_date` and `backtest.end_date` leave enough trailing history for the configured lookback window
- confirm the report frontier is feasible under the same constraint bundle passed to optimization and backtest

Useful commands:

```bash
uv run etf-portfolio optimize --config configs/base.yaml
uv run etf-portfolio backtest --config configs/base.yaml
uv run pytest tests/unit/test_optimizer.py tests/unit/test_walk_forward_backtest.py
```

## How To Add ETFs

1. Add the ETF row to `data/metadata/etf_universe.csv`.
2. Include at least the schema-required metadata columns.
3. If needed, update `configs/base.yaml` universe and constraint bounds.
4. Re-run ingestion and validation.
5. Re-run backtest/report generation.

When adding ETFs, check:

- inception date is correct
- expense ratio is stored as a decimal, not a percentage string
- asset class and region are consistent with existing taxonomy
- benchmark overlap is still valid
- new ETF does not make constraints infeasible

## Operational Notes

- Use notebooks for exploration only, not for the system of record.
- Use run records in `reports/runs` to tie outputs back to config, code, and data versions.
- When `optimization.active_objective` changes, regenerate the full pipeline and confirm the new `optimization_method` in the latest run record before trusting report outputs.
- Treat `yfinance` as a prototype provider. If a result matters, validate it against a second source.
- `backtest.start_date` and `backtest.end_date` filter both asset and benchmark return series before rebalance dates are generated, so moving the start date changes the first eligible rebalance date.
- `rebalance.mode: contribution_only` now separates optimizer-target constraints from realized-holdings behavior via `rebalance.realized_constraint_policy`.
- Use `report_drift` for tax-aware/HODL operation: realized cap breaches are surfaced in report warnings instead of treated as silent compliance.
- Use `enforce_hard` when realized caps must remain hard: contribution-only will allow sell-based fallback rebalances when a realized ticker/class/bond cap is breached.
- `rebalance.fallback_sell_allowed` plus `rebalance.fallback.sell_allowed_if_absolute_drift_exceeds` remains available for threshold-based sell fallbacks unrelated to hard-cap enforcement.
- Metrics artifacts are strict JSON and should parse cleanly with `json.loads`; literal `NaN` or `Infinity` in metrics files indicates a regression.
