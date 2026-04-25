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

This repository does not currently define a DVC workflow. Reproduce results
with the CLI command above or with `make run`.

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

## Run Tracking Policy

`tracking.require_git_commit` controls whether tracked outputs may be written
without a git commit:

- `tracking.require_git_commit: true` is strict mode and is the default. Backtest
  and ML runs must execute from a git repository with at least one commit. If no
  commit can be resolved, the command fails before writing metrics JSON that
  points at a run record.
- Strict-mode metrics and run records include `provenance_status: "tracked"` and
  a 40-character `git_commit_hash`.
- `tracking.require_git_commit: false` is preview mode for non-git zips or
  throwaway local experiments. The run is allowed to complete even when no commit
  exists.
- Preview-mode metrics and run records include
  `provenance_status: "untracked_preview"`. If no commit exists, the run record
  stores `git_commit_hash: null`.
- Preview outputs are not reproducible artifacts. Do not use them as the system
  of record; rerun in strict mode from a committed git checkout before relying on
  the results.

## How To Refresh Data

Refresh only ingestion and downstream stages:

```bash
uv run etf-portfolio ingest --config configs/base.yaml
uv run etf-portfolio validate --config configs/base.yaml
uv run etf-portfolio features --config configs/base.yaml
uv run etf-portfolio backtest --config configs/base.yaml
```

If you want the entire pipeline rebuilt from the configured provider and local
configuration, use:

```bash
uv run etf-portfolio run-all --config configs/base.yaml
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

## Troubleshooting

### Git Tracking Issues
- **Error: "Run failed: tracking.require_git_commit is true but no git commit hash found"**
  - This happens if you are running in a directory that is not a git repository or has no commits.
  - **Fix:** Initialize a git repo (`git init`), add files (`git add .`), and commit (`git commit -m "initial"`).
  - **Alternative:** For quick tests, set `tracking.require_git_commit: false` in your config, but note that outputs will be labeled as "untracked_preview".
- **Git State is Dirty:**
  - The system records the commit hash but does not verify if the working directory is clean. For absolute reproducibility, always commit your config changes before running the pipeline.

### yfinance / Provider Issues
- **"No data found for symbol XYZ" or "Ticker not found"**
  - yfinance sometimes fails for certain tickers or during temporary API outages.
  - **Check:** Verify the ticker on Yahoo Finance web.
  - **Check:** Try `uv run python -c "import yfinance as yf; print(yf.Ticker('SPY').history(period='1d'))"` to test your local connection.
  - **Fix:** If a ticker is consistently failing, it may have been delisted or requires a different provider.
- **Rate Limiting:**
  - If you are ingesting a large universe, you may hit rate limits. The pipeline includes basic retry logic, but for very large universes, consider a professional data provider.

### Image Export Issues
- **Figure saves as a 1x1 empty PNG (Placeholder):**
  - This happens if `kaleido` (the image export engine) fails or is not installed correctly.
  - **Check:** Look for `report_figure_export_fallback` warnings in the logs.
  - **Fix:** Ensure `kaleido` is in your environment (`uv sync`). If it still fails, you can still view the interactive charts in the generated HTML report.
- **Charts are missing in the HTML report:**
  - Ensure `get_plotlyjs()` is working and JavaScript is enabled in your browser. The HTML report is self-contained and should work offline.

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
