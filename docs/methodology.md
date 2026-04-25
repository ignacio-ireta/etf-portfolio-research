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

Missing-data policy:

- observed prices must be strictly positive; zero and negative observed prices fail validation before returns are created
- requested ticker columns must be returned by the configured provider; omitted provider columns fail ingestion instead of producing partial universes
- simple returns preserve missing prices by default and call pandas with `fill_method=None`; a missing middle price produces missing returns around that gap instead of an implicit flat return and catch-up return
- forward-filling prices before return calculation is only allowed when a caller explicitly requests the `forward_fill` missing-price policy
- suspicious-jump detection also uses `fill_method=None`, so jumps are measured only across adjacent observed prices and are not created by implicit gap filling

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
- `target_return` and `target_volatility` remain optimizer/frontier APIs, not run-config objectives; `optimization.target_return` and `optimization.target_volatility` keys are rejected in config files
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
- ticker bounds must refer to configured/available tickers; stale ticker bounds fail validation instead of being ignored
- asset-class bounds must refer to metadata asset classes available to the optimization universe
- feasibility is checked before solve using the effective per-ticker bounds and the combined linear constraint system, including weight sum, ticker bounds, asset-class bounds, bond exposure, and expense cap

Important limitation:

- the optimizer and backtest engine support richer constraints than the earliest scaffold docs implied, but constraint effectiveness still depends on clean ETF metadata and sensible config bounds
- the report frontier is now built under the same constraint bundle as the optimizer and walk-forward backtest, so infeasibility in one surface should be investigated as a shared-constraint problem rather than a reporting-only discrepancy

## Benchmark Fairness

To ensure backtest comparisons are defensible, optimized benchmarks (such as Equal-Weight, Inverse-Volatility, or Min-Variance) are subject to the same operational constraints as the main strategy:

- **Shared Rebalance Mode:** If the main strategy uses `contribution_only`, optimized benchmarks also use `contribution_only`. This prevents benchmarks from having an "unfair" advantage of being able to sell to maintain risk targets if the user cannot.
- **Shared Contribution Amount:** Any periodic external contribution added to the main strategy is also added to optimized benchmarks.
- **Shared Operational Parameters:** Initial capital, transaction costs, slippage, and rebalance frequency are aligned across all optimized portfolios and benchmarks.

Constant-weight benchmarks (like 60/40 or single ETFs) remain as theoretical ideal baselines and do not model transaction costs or drift between rebalance dates unless otherwise noted.

## Execution Timing Assumptions

The backtest engine assumes **End-of-Day Execution**:

1. **Information Cutoff:** At the close of `rebalance_date` (T), the optimizer calculates new target weights using data available up to and including T.
2. **Execution:** Trades are assumed to execute at the close of T.
3. **Transaction Costs:** Costs and slippage for those trades are subtracted from the *first day* of the new period (T+1).
4. **Realized Returns:**
   - The return on `rebalance_date` (T) is realized using the **old** weights (from the previous period).
   - The returns starting from the next day (T+1) are realized using the **new** weights (applied at T close).

This approach avoids lookahead bias by ensuring the weights applied to a return were determined strictly before that return was realized.

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

Sharpe, Sortino, and Calmar ratios are reported as finite values. When the
relevant denominator is effectively zero, the ratio is reported as `0.0`
rather than `NaN` or infinity.

Turnover is defined as gross traded-weight turnover: the sum of absolute
weight changes across assets. Portfolio summary turnover averages that gross
change across rebalance-to-rebalance transitions and excludes the initial
allocation from cash.

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
