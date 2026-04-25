# Guided Examples

These walkthroughs show how to reproduce, modify, and interpret the project using the generated artifacts already present in `reports/`.

The examples use this artifact snapshot as the baseline:

- run record: `reports/runs/backtest_run-all-20260424T175044Z-3777911e.json`
- HTML report: `reports/html/latest_report.html`
- Excel workbook: `reports/excel/portfolio_results.xlsx`
- metrics JSON: `reports/metrics/backtest_metrics.json`
- figures: `reports/figures/*.png`

If you rerun the pipeline, the run ID, dates, prices, and metrics may change. That is expected. Treat the numbers below as a concrete reading guide for the current checked-in artifact bundle, not as permanent investment facts.

This project is a research tool, not financial advice.

Before interpreting the examples, read [Trust And Safety](trust_and_safety.md) to avoid common false conclusions about Sharpe ratios, backtests, historical mean returns, costs, and contribution-only drift.

## Example 1: Default Run

### Goal

Reproduce the standard research pipeline and know which artifacts to inspect first.

### Reproduce

Install dependencies:

```bash
uv sync --group dev
```

Run the full pipeline:

```bash
uv run etf-portfolio run-all --config configs/base.yaml
```

The default config uses:

- ETF universe: `VTI`, `VEA`, `VWO`, `BND`, `IEI`, `TIP`, `IAU`, `VNQ`, `GSG`, `QQQ`, `TLT`, `REMX`
- primary benchmark: `VT`
- optimizer objective: `max_sharpe`
- risk model: `ledoit_wolf`
- rebalance mode: `contribution_only`
- contribution amount: `1500.0`
- backtest start date: `2012-01-01`

Because the walk-forward backtest needs trailing lookback data, the current run record reports an effective backtest start date of `2015-01-30`.

### Inspect The Artifacts

Start here:

```text
reports/html/latest_report.html
```

Then inspect the machine-readable files:

```text
reports/metrics/backtest_metrics.json
reports/runs/backtest_run-all-20260424T175044Z-3777911e.json
reports/excel/portfolio_results.xlsx
```

Use the run record to confirm what produced the report:

- `run_id`: `run-all-20260424T141121Z-1a1090c0`
- `optimization_method`: `max_sharpe`
- `risk_model`: `ledoit_wolf`
- `benchmark`: `VT`
- `start_date`: `2015-01-30`
- `end_date`: `2026-04-24`

### Read The Result

The current metrics JSON reports these optimized-strategy values:

- `CAGR`: `0.102113`, meaning about `10.21%` annualized historical growth.
- `Annualized Volatility`: `0.121614`, meaning about `12.16%` annualized fluctuation.
- `Max Drawdown`: `-0.250168`, meaning the worst peak-to-trough decline was about `-25.02%`.
- `Sharpe Ratio`: `0.613851`, a risk-adjusted return estimate after the configured risk-free rate.
- `Beta`: `0.669599`, lower than the selected benchmark ETF in this run.
- `Turnover`: `0.022086`, meaning average rebalance turnover was about `2.21%`.

Compare those against `Selected Benchmark ETF` in the same JSON:

- benchmark `CAGR`: `0.108676`
- benchmark `Annualized Volatility`: `0.171229`
- benchmark `Max Drawdown`: `-0.342362`
- benchmark `Sharpe Ratio`: `0.513463`

One reasonable reading is: in this historical run, the optimized strategy had lower CAGR than `VT`, but also lower volatility, lower drawdown, and a higher Sharpe ratio. That does not prove it is better. It means the selected configuration produced a smoother historical path under the modeled assumptions.

## Example 2: Changing ETFs

### Goal

Change the ETF universe without losing track of what changed or which artifact should be compared.

### Start From The Current Universe

The current universe is defined in `configs/base.yaml` and explained in `data/metadata/etf_universe.csv`.

The current generated workbook also records the ETF universe in:

```text
reports/excel/portfolio_results.xlsx
```

Use the `etf_universe` sheet to see ticker names, asset classes, regions, currencies, expense ratios, benchmark indexes, and inception dates.

The latest realized portfolio in the current workbook is:

- `VTI`: `41.62%`
- `QQQ`: `18.24%`
- `IAU`: `10.91%`
- `BND`: `7.34%`
- `IEI`: `4.64%`
- `VEA`: `4.29%`
- `TLT`: `3.90%`
- `VNQ`: `3.00%`
- `TIP`: `3.00%`
- `GSG`: `2.18%`
- `REMX`: `0.82%`
- `VWO`: `0.05%`

The latest asset-class exposure is:

- equity: `65.02%`
- fixed income: `18.88%`
- commodity: `13.10%`
- real estate: `3.00%`

These values give you a baseline before changing the universe.

### Make A Safe Config Copy

Do not overwrite the baseline config while experimenting:

```bash
cp configs/base.yaml configs/my_universe.yaml
```

Edit `configs/my_universe.yaml`.

For example, to remove the thematic `REMX` satellite from the investable universe, remove `REMX` from:

```yaml
universe:
  tickers:
    - VTI
    - VEA
    - VWO
    - BND
    - IEI
    - TIP
    - IAU
    - VNQ
    - GSG
    - QQQ
    - TLT
```

Then also remove or adjust the matching `constraints.ticker_bounds.REMX` entry. Bounds for tickers outside `universe.tickers` are rejected during config validation so stale constraints cannot be silently ignored.

If you add a new ETF instead of only removing one, also update `data/metadata/etf_universe.csv` with complete metadata. At minimum, the new ticker needs consistent asset class, region, currency, expense ratio, benchmark index, and inception date data.

### Rerun

If all required price data is already present, you can rerun downstream stages:

```bash
uv run etf-portfolio backtest --config configs/my_universe.yaml
```

If you added a ticker that is not already in `data/raw/prices.parquet`, run the full pipeline so ingestion can fetch the new data:

```bash
uv run etf-portfolio run-all --config configs/my_universe.yaml
```

### Compare

Compare the new outputs against the baseline artifacts:

- `reports/runs/*.json`: confirm the new run used the intended config, objective, risk model, and universe.
- `reports/html/latest_report.html`: review the ETF universe, data coverage, reader guide, metric dictionary, portfolio weights, and benchmark comparison.
- `reports/excel/portfolio_results.xlsx`: compare `latest_realized_portfolio`, `optimizer_target_portfolio`, `metrics`, `asset_class_exposure`, `region_exposure`, and `realized_constraint_warnings`.
- `reports/metrics/backtest_metrics.json`: compare the top-level optimized strategy metrics against the old values listed in Example 1.

A universe change is meaningful only if you can explain why the resulting metrics changed. If removing one ETF improves a metric but increases concentration, worsens drawdown, or breaks the intended exposure profile, the change may not be an improvement.

## Example 3: Interpreting A Backtest

### Goal

Read the current backtest as a structured historical experiment.

### Step 1: Confirm The Experiment Setup

Open:

```text
reports/runs/backtest_run-all-20260424T175044Z-3777911e.json
```

Check:

- `benchmark`: `VT`
- `optimization_method`: `max_sharpe`
- `risk_model`: `ledoit_wolf`
- `expected_return_estimator`: `historical_mean`
- `start_date`: `2015-01-30`
- `end_date`: `2026-04-24`
- `data_version.exists`: `true`

This tells you what historical experiment was run. Do this before reading performance metrics.

### Step 2: Read Return And Risk Together

Open:

```text
reports/metrics/backtest_metrics.json
```

For the optimized strategy, read these together:

- `CAGR`: historical annualized growth.
- `Annualized Volatility`: how much the return path fluctuated.
- `Max Drawdown`: worst historical decline from a prior high.
- `Sharpe Ratio` and `Sortino Ratio`: return adjusted by broad risk and downside risk.
- `Calmar Ratio`: return relative to max drawdown.

Do not read `CAGR` alone. In the current artifact, the selected benchmark ETF has a higher CAGR than the optimized strategy, but the optimized strategy has lower drawdown and volatility. That is a tradeoff, not a simple win or loss.

### Step 3: Check Benchmark-Relative Metrics

Use:

- `Beta`: how sensitive the strategy was to the selected benchmark.
- `Alpha`: historical return not explained by benchmark beta under the model.
- `Tracking Error`: how differently the strategy moved from the benchmark.
- `Information Ratio`: excess return per unit of tracking error.

In the current optimized-strategy metrics:

- `Beta`: `0.669599`
- `Alpha`: `0.019432`
- `Tracking Error`: `0.069605`
- `Information Ratio`: `-0.190605`

This combination says the strategy had lower benchmark sensitivity and positive model-estimated alpha, but did not deliver attractive excess return per unit of tracking error in this run.

### Step 4: Inspect Portfolio Behavior

Open:

```text
reports/excel/portfolio_results.xlsx
```

Use these sheets:

- `latest_realized_portfolio`: what the simulated portfolio actually held at the end.
- `optimizer_target_portfolio`: what the optimizer wanted at the latest rebalance.
- `weights_history`: realized weights through time.
- `optimizer_target_history`: target weights through time.
- `rebalance_summary`: turnover, transactions, contribution behavior, and constraint warnings by rebalance date.
- `realized_constraint_warnings`: any drift beyond configured realized constraints.

In the current artifact, the latest realized portfolio differs from the latest optimizer target. That is expected under `contribution_only` rebalancing: new cash moves the portfolio toward target weights, but existing holdings are not automatically sold unless the configured policy allows it.

### Step 5: Use Figures For Pattern Recognition

Use these generated figures:

- `reports/figures/cumulative_returns.png`: growth path.
- `reports/figures/drawdown.png`: depth and length of losses.
- `reports/figures/portfolio_weights.png`: concentration and drift through time.
- `reports/figures/benchmark_comparison.png`: portfolio versus benchmark path.
- `reports/figures/rolling_volatility.png`: whether risk was stable or regime-dependent.
- `reports/figures/rolling_sharpe.png`: whether risk-adjusted performance was persistent.
- `reports/figures/stress_periods.png`: behavior during named stress windows.

Charts are not proof. Use them to find patterns worth checking in the tables and JSON.

### Step 6: End With Limitations

Before using the result, read:

```text
docs/assumptions_and_limitations.md
```

The backtest is conditional on the ETF universe, data source, historical period, objective, constraints, benchmark, costs, and rebalancing assumptions. If those assumptions do not match your question, the output may not answer your question.
