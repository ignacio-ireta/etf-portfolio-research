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

To ensure backtest comparisons are defensible, optimized benchmarks (such as Equal-Weight, Inverse-Volatility, or Min-Variance) are subject to the same operational implementation as the main strategy:

- **Shared Rebalance Mode:** If the main strategy uses `contribution_only`, optimized benchmarks also use `contribution_only`. This prevents benchmarks from having an "unfair" advantage of being able to sell to maintain risk targets if the user cannot.
- **Shared Contribution Amount:** Any periodic external contribution added to the main strategy is also added to optimized benchmarks.
- **Shared Operational Parameters:** Initial capital, transaction costs, slippage, constraints, realized constraint policy, and rebalance frequency are aligned across all optimized portfolios and optimized benchmark objectives.

The chosen project policy is therefore to compare contribution-only implementation against contribution-only optimized benchmarks, not against full-rebalance theoretical optimized benchmarks.

Constant-weight benchmarks (like 60/40, configured secondary allocation benchmarks, or single ETFs) remain theoretical return-series baselines. They do not model contribution-only drift, strategy transaction costs, or rebalance-date implementation mechanics unless a future config explicitly adds those assumptions.

## Execution Timing Assumptions

The backtest engine uses conservative rebalance-date semantics:

1. **Information Cutoff:** For a `rebalance_date` (T), the optimizer calculates new target weights using returns strictly before T. The return interval labeled T is not used to set weights for T.
2. **Realization On T:** The return labeled T is realized by the previous period's weights.
3. **Execution:** Trades are assumed after the T return interval, so the newly applied weights affect returns strictly after T.
4. **Transaction Costs:** Costs and slippage for those trades are subtracted from the first post-rebalance return date.
5. **Realized Returns:** The portfolio return series for a newly selected weight vector starts strictly after that weight vector's `rebalance_date`.

This approach avoids lookahead bias by ensuring the weights applied to a return were determined before that return was realized.

## Rebalancing Guarantees

Tolerance-band rebalancing is constraint-safe for the configured drift bands. When a ticker or asset-class band is breached, the engine solves a constrained projection to the nearest long-only portfolio that sums to 1.0 and satisfies:

- every ticker weight within `target_weight +/- rebalance.tolerance_bands.per_ticker_abs_drift`
- every available asset-class exposure within `target_class_weight +/- rebalance.tolerance_bands.per_asset_class_abs_drift`

Contribution-only rebalancing has a different guarantee. It never sells unless an explicit sell fallback or hard-constraint policy is triggered. New cash is first routed toward under-weight positions. If the contribution is larger than all under-weight gaps, the surplus is invested proportionally to target weights, so a previously over-weight ticker may still receive part of that surplus. Under `realized_constraint_policy: report_drift`, realized holdings can remain outside configured caps and are reported as drift warnings rather than forcibly sold.

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
- ML is disabled by default. When explicitly enabled, the persisted research model is trained on the chronological train split and labeled `train_split`; the final test window remains held out for evaluation and governance. Portfolio use remains gated by out-of-sample evidence and governance approval.
