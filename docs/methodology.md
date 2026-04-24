# Methodology

## Return Estimation

Current expected return estimator:

- `historical_mean`

Implemented return functions in `src/etf_portfolio/features/returns.py`:

- simple returns
- log returns
- annualized return
- annualized volatility
- cumulative returns
- drawdown series
- max drawdown

Assumption:

- expected returns are backward-looking and estimated from the trailing window available at each rebalance date

## Covariance Estimation

Implemented in `src/etf_portfolio/features/estimators.py`.

Supported covariance methods:

- `sample`
- `ledoit_wolf`

The active method is controlled by `optimization.risk_model` in the config.

## Optimization Objectives

Implemented optimizer surface:

- equal weight
- inverse volatility
- minimum variance
- maximum Sharpe
- target volatility
- target return
- risk parity

The config selects the optimizer objective through `optimization.active_objective`.

Current pipeline behavior:

- `optimization.active_objective` is the authoritative objective selector used by `optimize`, `backtest`, and `run-all`
- run artifacts in `reports/runs/*.json` persist that effective `optimization_method` for auditability

## Constraints

Implemented v1 constraints:

- weights sum to 1
- long-only bounds
- max ETF weight
- ticker-level min/max bounds
- asset-class exposure bounds
- minimum bond exposure
- weighted expense ratio cap
- turnover-aware backtest accounting through transaction costs

Ticker bound behavior:

- `optimization.default_max_weight_per_etf` and optimizer `min_weight` define default per-asset bounds
- `constraints.ticker_bounds` overrides those defaults per ticker when provided
- feasibility is checked before solve using the effective per-ticker bounds:
  - `sum(min_bounds) <= weight_sum`
  - `sum(max_bounds) >= weight_sum`

Important limitation:

- the optimizer and backtest engine support richer constraints than the earliest scaffold docs implied, but constraint effectiveness still depends on clean ETF metadata and sensible config bounds
- the report frontier is now built under the same constraint bundle as the optimizer and walk-forward backtest, so infeasibility in one surface should be investigated as a shared-constraint problem rather than a reporting-only discrepancy

## Backtest Design

Walk-forward backtesting is implemented in `src/etf_portfolio/backtesting/engine.py`.

At each rebalance date:

1. use only trailing return history strictly before the rebalance date
2. estimate expected returns and covariance from that trailing window
3. optimize weights under the configured constraints
4. decide how optimizer targets are translated into realized holdings based on the rebalance mode and realized-constraint policy
5. apply turnover-based transaction costs and slippage assumptions
6. hold through the next rebalance window
7. record realized returns, optimizer targets, applied weights, trades, and diagnostics

Benchmarks currently supported in reporting/backtesting:

- selected benchmark ETF
- configured secondary benchmark mixes
- equal-weight universe
- inverse-volatility baseline
- previous optimized strategy

Backtest windowing and realized holdings:

- `backtest.start_date` and `backtest.end_date` are applied to aligned asset and benchmark return series before rebalance dates are generated
- `WalkForwardBacktestResult` stores optimizer `target_weights` separately from realized `applied_weights`
- realized dollar trades are persisted as `trades_dollars`
- contribution-only runs persist `realized_constraint_violations` so drift beyond configured ticker, asset-class, or bond-floor constraints is explicit
- with `rebalance.realized_constraint_policy: enforce_hard`, contribution-only may perform a sell-based fallback rebalance when realized hard caps are breached; with `report_drift`, those breaches are reported as soft drift warnings instead

## Risk Metrics

Canonical metric explanations live in [Metric Dictionary](metric_dictionary.md).

Implemented portfolio and backtest metrics include:

- CAGR
- annualized volatility
- Sharpe ratio
- Sortino ratio
- maximum drawdown
- Calmar ratio
- beta
- alpha
- tracking error
- information ratio
- turnover
- average number of holdings
- largest position
- Herfindahl concentration index
- worst month
- worst quarter
- best month

Rolling and exposure views in the report include:

- ETF weights
- asset-class exposure
- region exposure
- currency exposure
- weighted expense ratio
- rolling volatility
- rolling Sharpe
- rolling correlation to benchmark
- stress-period returns

Artifact conventions:

- HTML and Excel reports distinguish the latest realized portfolio from the optimizer target portfolio
- the workbook uses `latest_realized_portfolio` and `optimizer_target_portfolio` sheets, with warning tables for realized constraint drift when present
- metrics JSON artifacts are sanitized to strict JSON before write, so downstream consumers can parse them with standard JSON tooling

## Known Limitations

- `yfinance` is the initial free data source and should be treated as research-grade, not production-grade.
- Benchmark analytics are only as clean as the overlapping history between benchmark and asset series.
- The metadata validation schema is stricter than the raw CSV storage layout; ingestion currently selects only the schema-required columns.
- Backtests are end-of-period and do not model intraday execution, taxes, or market impact.
- FX-adjusted returns, MXN/NOK reporting layers, UCITS comparisons, and tax-aware domicile analysis are not implemented yet.
- ML exists, but portfolio use should remain gated by out-of-sample evidence and governance approval.
