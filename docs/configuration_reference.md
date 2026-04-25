# Configuration Reference

The CLI intentionally exposes a small flag surface. Most runtime behavior is
controlled by YAML config keys, so treat each key below like a long-form
terminal flag.

The base file is [configs/base.yaml](../configs/base.yaml). For experiments,
copy it first and run against the copy:

```bash
cp configs/base.yaml configs/my_experiment.yaml
uv run etf-portfolio run-all --config configs/my_experiment.yaml
```

## CLI Commands And Flags

All commands require `--config`.

| Command | Purpose | Flags |
| --- | --- | --- |
| `ingest` | Download raw prices for the configured universe, primary benchmark, and secondary benchmark tickers. | `--config PATH` |
| `validate` | Validate raw prices and write cleaned prices. | `--config PATH` |
| `features` | Build periodic returns from validated prices. | `--config PATH` |
| `optimize` | Run the configured optimizer objective and write optimizer outputs. | `--config PATH` |
| `backtest` | Run walk-forward backtest and report generation from existing pipeline inputs. | `--config PATH`, `--lookback-periods N` |
| `report` | Regenerate the report bundle through the backtest/report path. | `--config PATH`, `--lookback-periods N` |
| `run-all` | Run ingest, validate, features, optimize, and report/backtest. | `--config PATH`, `--lookback-periods N` |
| `ml` | Train and evaluate configured ML research models. | `--config PATH` |

`--lookback-periods` is the trailing return history used for each walk-forward
optimization window. The default is `756`, roughly three trading years.

Examples:

```bash
uv run etf-portfolio backtest --config configs/base.yaml --lookback-periods 504
uv run etf-portfolio report --config configs/my_experiment.yaml --lookback-periods 252
uv run etf-portfolio run-all --config configs/my_experiment.yaml --lookback-periods 1008
```

## YAML Parameters

### `project`

| Key | Type | Accepted values | Effect |
| --- | --- | --- | --- |
| `project.name` | string | non-empty | Used in reports and run metadata. |
| `project.base_currency` | string | 3-letter currency code | Labels capital and cash figures. |

Example:

```yaml
project:
  name: my_global_etf_test
  base_currency: USD
```

### `universe`

| Key | Type | Accepted values | Effect |
| --- | --- | --- | --- |
| `universe.tickers` | list of strings | unique, non-empty tickers | ETF universe optimized and backtested. Each ticker must exist in `data/metadata/etf_universe.csv`. |

When removing a ticker, also remove any matching `constraints.ticker_bounds`
entry.

Example:

```yaml
universe:
  tickers:
    - VTI
    - VEA
    - VWO
    - BND
    - IAU
```

### `benchmark`

| Key | Type | Accepted values | Effect |
| --- | --- | --- | --- |
| `benchmark.primary` | string | non-empty ticker | Primary external benchmark return series. |
| `benchmark.secondary.<name>` | mapping | ticker weights summing to `1.0` | Additional theoretical allocation benchmarks in the report. |

Example:

```yaml
benchmark:
  primary: VT
  secondary:
    global_60_40:
      VT: 0.60
      BND: 0.40
    us_80_20:
      VTI: 0.80
      BND: 0.20
```

### `data`

| Key | Type | Accepted values | Effect |
| --- | --- | --- | --- |
| `data.provider` | string | `yfinance`, `tiingo` | Primary market data provider. `tiingo` requires `TIINGO_API_KEY`. |
| `data.start_date` | date string | `YYYY-MM-DD` | First requested price date. |
| `data.end_date` | date string or `null` | `YYYY-MM-DD`, `null` | Last requested price date. `null` uses latest provider data. |
| `data.price_field` | string | non-empty | Recorded price field assumption. Current providers return adjusted closes. |
| `data.cross_check.enabled` | boolean | `true`, `false` | Enables provider cross-checking during ingestion. |
| `data.cross_check.provider` | string or `null` | `yfinance`, `tiingo`, `null` | Secondary provider. Required when cross-checking is enabled. |
| `data.cross_check.max_relative_divergence` | number | `>= 0.0` | Maximum accepted relative price divergence. |
| `data.cross_check.min_overlap_observations` | integer | `>= 1` | Minimum overlapping dates needed for cross-checking. |

Example:

```yaml
data:
  provider: yfinance
  start_date: "2015-01-01"
  end_date: null
  price_field: adjusted_close
  cross_check:
    enabled: true
    provider: tiingo
    max_relative_divergence: 0.005
    min_overlap_observations: 20
```

### `investor_profile`

| Key | Type | Accepted values | Effect |
| --- | --- | --- | --- |
| `investor_profile.horizon_years` | integer | `> 0` | Documents the intended investment horizon. |
| `investor_profile.objective` | string | non-empty | Documents the investment objective. |
| `investor_profile.tax_preference` | string | non-empty | Documents tax preference assumptions. |

Example:

```yaml
investor_profile:
  horizon_years: 20
  objective: long_term_accumulation
  tax_preference: minimize_realized_gains
```

### `optimization`

| Key | Type | Accepted values | Effect |
| --- | --- | --- | --- |
| `optimization.long_only` | boolean | `true`, `false` | Long-only assumption. The pipeline is built around long-only portfolios. |
| `optimization.default_max_weight_per_etf` | number | `> 0.0`, `<= 1.0` | Default per-ticker cap unless overridden by ticker bounds. |
| `optimization.risk_model` | string | `sample`, `ledoit_wolf` | Covariance estimator. |
| `optimization.expected_return_estimator` | string | `historical_mean` | Expected return estimator. |
| `optimization.active_objective` | string | `equal_weight`, `inverse_volatility`, `min_variance`, `max_sharpe`, `risk_parity` | Main optimizer objective for optimize, backtest, report, and run records. |
| `optimization.benchmark_objectives` | list of strings | same objective values as above, unique, excluding active objective | Additional optimized benchmark strategies. |

`optimization.target_return` and `optimization.target_volatility` are rejected
in run config files. Use optimizer/frontier APIs for targeted efficient-frontier
experiments.

Example:

```yaml
optimization:
  long_only: true
  default_max_weight_per_etf: 0.20
  risk_model: ledoit_wolf
  expected_return_estimator: historical_mean
  active_objective: min_variance
  benchmark_objectives:
    - equal_weight
    - inverse_volatility
    - max_sharpe
```

### `constraints`

| Key | Type | Accepted values | Effect |
| --- | --- | --- | --- |
| `constraints.asset_class_bounds.<asset_class>.min` | number | `0.0` to `1.0` | Minimum allocation for that metadata asset class. |
| `constraints.asset_class_bounds.<asset_class>.max` | number | `0.0` to `1.0`, `>= min` | Maximum allocation for that metadata asset class. |
| `constraints.ticker_bounds.<ticker>.min` | number | `0.0` to `1.0` | Minimum allocation for one universe ticker. |
| `constraints.ticker_bounds.<ticker>.max` | number | `0.0` to `1.0`, `>= min` | Maximum allocation for one universe ticker. |

Ticker bounds may only reference tickers in `universe.tickers`. The combined
default and ticker-specific maximum capacity must be feasible for a portfolio
that sums to `1.0`.

Example:

```yaml
constraints:
  asset_class_bounds:
    equity:
      min: 0.50
      max: 0.85
    fixed_income:
      min: 0.15
      max: 0.45
  ticker_bounds:
    VTI:
      min: 0.15
      max: 0.50
    BND:
      min: 0.00
      max: 0.40
```

### `rebalance`

| Key | Type | Accepted values | Effect |
| --- | --- | --- | --- |
| `rebalance.mode` | string | `contribution_only`, `full_rebalance`, `tolerance_band` | Rebalancing implementation style. |
| `rebalance.frequency` | string | `daily`, `weekly`, `monthly`, `quarterly`, `yearly` | Rebalance schedule. |
| `rebalance.realized_constraint_policy` | string | `report_drift`, `enforce_hard` | Whether contribution-only realized cap breaches are reported or forcibly corrected. |
| `rebalance.fallback_sell_allowed` | boolean | `true`, `false` | Allows sell-based fallback when the configured fallback threshold is breached. |
| `rebalance.fallback.sell_allowed_if_absolute_drift_exceeds` | number | `> 0.0`, `<= 1.0` | Absolute drift threshold for sell fallback. Required when fallback selling is enabled. |
| `rebalance.contribution_amount` | number | `>= 0.0`; must be `> 0.0` for `contribution_only` | Cash contributed at each rebalance. |
| `rebalance.tolerance_bands.per_ticker_abs_drift` | number | `0.0` to `1.0` | Per-ticker absolute weight drift band. |
| `rebalance.tolerance_bands.per_asset_class_abs_drift` | number | `0.0` to `1.0` | Per-asset-class absolute weight drift band. |

Examples:

```yaml
rebalance:
  mode: contribution_only
  frequency: monthly
  realized_constraint_policy: report_drift
  fallback_sell_allowed: false
  contribution_amount: 1500.0
```

```yaml
rebalance:
  mode: tolerance_band
  frequency: quarterly
  realized_constraint_policy: enforce_hard
  fallback_sell_allowed: true
  fallback:
    sell_allowed_if_absolute_drift_exceeds: 0.10
  contribution_amount: 0.0
  tolerance_bands:
    per_ticker_abs_drift: 0.04
    per_asset_class_abs_drift: 0.08
```

### `backtest`

| Key | Type | Accepted values | Effect |
| --- | --- | --- | --- |
| `backtest.start_date` | date string or `null` | `YYYY-MM-DD`, `null` | First eligible backtest return date after alignment. |
| `backtest.end_date` | date string or `null` | `YYYY-MM-DD`, `null`; must be after start date | Last eligible backtest return date after alignment. |
| `backtest.initial_capital` | number | `> 0.0` | Starting capital used to convert weights and contributions into trades. |

Example:

```yaml
backtest:
  start_date: "2018-01-01"
  end_date: "2025-12-31"
  initial_capital: 250000.0
```

### `costs`

| Key | Type | Accepted values | Effect |
| --- | --- | --- | --- |
| `costs.transaction_cost_bps` | number | `>= 0.0` | Transaction cost assumption in basis points. |
| `costs.slippage_bps` | number | `>= 0.0` | Slippage assumption in basis points. |

Example:

```yaml
costs:
  transaction_cost_bps: 3
  slippage_bps: 2
```

### `risk_free`

| Key | Type | Accepted values | Effect |
| --- | --- | --- | --- |
| `risk_free.source` | string | `constant` | Risk-free rate source. |
| `risk_free.value` | number | `0.0` to `1.0` | Annualized risk-free rate used by Sharpe, Sortino, alpha, and max-Sharpe optimization. |

Example:

```yaml
risk_free:
  source: constant
  value: 0.04
```

### `tracking`

| Key | Type | Accepted values | Effect |
| --- | --- | --- | --- |
| `tracking.artifact_dir` | string | non-empty path | Directory for run records. |
| `tracking.require_git_commit` | boolean | `true`, `false` | Strict mode requires a resolvable git commit before writing tracked backtest or ML outputs. |

Example for a local preview zip or scratch run:

```yaml
tracking:
  artifact_dir: reports/runs
  require_git_commit: false
```

### `ml`

ML is disabled by default and produces research artifacts only.

| Key | Type | Accepted values | Effect |
| --- | --- | --- | --- |
| `ml.enabled` | boolean | `true`, `false` | Enables `uv run etf-portfolio ml`. |
| `ml.task` | string | `regression`, `classification` | Model task. Classification is only valid with `beat_benchmark`. |
| `ml.target` | string | `forward_return`, `forward_volatility`, `forward_drawdown`, `beat_benchmark` | Prediction target. Non-`beat_benchmark` targets require regression. |
| `ml.horizon_periods` | integer | `> 0` | Forward target horizon in return periods. |
| `ml.models` | list of strings | `historical_mean`, `ridge`, `random_forest`; unique, non-empty | Candidate model set. |
| `ml.features.lag_periods` | list of integers | positive integers | Lagged return windows. |
| `ml.features.momentum_periods` | list of integers | positive integers | Momentum windows. |
| `ml.features.volatility_windows` | list of integers | positive integers | Realized volatility windows. |
| `ml.features.drawdown_windows` | list of integers | positive integers | Drawdown windows. |
| `ml.features.correlation_windows` | list of integers | positive integers | Benchmark correlation windows. |
| `ml.features.moving_average_windows` | list of integers | positive integers | Moving-average windows. |
| `ml.validation.train_window_periods` | integer | `> 0`, `>= min_train_periods` | Walk-forward training window. |
| `ml.validation.test_window_periods` | integer | `> 0` | Walk-forward test window. |
| `ml.validation.step_periods` | integer | `> 0` | Step between validation folds. |
| `ml.validation.min_train_periods` | integer | `> 0` | Minimum observations required for training. |
| `ml.validation.embargo_periods` | integer | `>= 0` | Gap between training and test samples. |
| `ml.tracking.enable_mlflow` | boolean | `true`, `false` | Enables MLflow logging when available. |
| `ml.tracking.experiment_name` | string | non-empty | MLflow experiment name. |
| `ml.tracking.artifact_dir` | string | non-empty path | ML artifact directory. |
| `ml.tracking.dataset_version` | string | non-empty | Dataset version label or path. |
| `ml.tracking.feature_version` | string | non-empty | Feature version label. |
| `ml.governance.minimum_fold_win_rate` | number | `0.0` to `1.0` | Minimum share of folds that must pass. |
| `ml.governance.minimum_folds_for_stability` | integer | `>= 1` | Minimum folds needed for stability checks. |
| `ml.governance.require_baseline_outperformance` | boolean | `true`, `false` | Requires model outperformance versus baseline. |
| `ml.governance.require_leakage_checks` | boolean | `true`, `false` | Requires leakage checks before accepting results. |

Regression example:

```yaml
ml:
  enabled: true
  task: regression
  target: forward_return
  horizon_periods: 21
  models:
    - ridge
    - random_forest
  features:
    lag_periods: [1, 5, 21]
    momentum_periods: [21, 63, 126]
    volatility_windows: [21, 63]
    drawdown_windows: [63]
    correlation_windows: [63]
    moving_average_windows: [21, 63]
  validation:
    train_window_periods: 252
    test_window_periods: 21
    step_periods: 21
    min_train_periods: 252
    embargo_periods: 0
```

Classification example:

```yaml
ml:
  enabled: true
  task: classification
  target: beat_benchmark
  horizon_periods: 21
```

Run:

```bash
uv run etf-portfolio ml --config configs/my_experiment.yaml
```

## Common Experiment Patterns

Change the optimizer objective:

```yaml
optimization:
  active_objective: risk_parity
```

Use a shorter walk-forward lookback without editing YAML:

```bash
uv run etf-portfolio backtest --config configs/my_experiment.yaml --lookback-periods 252
```

Move from contribution-only behavior to full periodic rebalancing:

```yaml
rebalance:
  mode: full_rebalance
  frequency: quarterly
  contribution_amount: 0.0
```

Make realized constraints hard in contribution-only mode:

```yaml
rebalance:
  mode: contribution_only
  realized_constraint_policy: enforce_hard
  contribution_amount: 1500.0
```

Add a secondary benchmark:

```yaml
benchmark:
  secondary:
    us_stock_bond_70_30:
      VTI: 0.70
      BND: 0.30
```

Limit the backtest window:

```yaml
backtest:
  start_date: "2020-01-01"
  end_date: null
```
