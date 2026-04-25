from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from etf_portfolio.config import (
    CrossCheckConfig,
    MLConfig,
    RiskFreeConfig,
    config_to_dict,
    load_config,
    load_config_files,
)


def test_load_config_reads_revised_base_yaml() -> None:
    config = load_config("configs/base.yaml")

    assert config.project.name == "etf_portfolio_research"
    assert config.project.base_currency == "USD"
    assert config.universe.tickers == [
        "VTI",
        "VEA",
        "VWO",
        "BND",
        "IEI",
        "TIP",
        "IAU",
        "VNQ",
        "GSG",
        "QQQ",
        "TLT",
        "REMX",
    ]
    assert config.benchmark.primary == "VT"
    assert config.benchmark.secondary["global_60_40"].allocations == {"VT": 0.6, "BND": 0.4}
    assert config.optimization.active_objective == "max_sharpe"
    assert config.optimization.default_max_weight_per_etf == 0.25
    assert config.optimization.benchmark_objectives == [
        "equal_weight",
        "inverse_volatility",
        "min_variance",
        "risk_parity",
    ]
    assert config.optimization.risk_model == "ledoit_wolf"
    assert config.optimization.expected_return_estimator == "historical_mean"
    assert set(config.constraints.asset_class_bounds) == {
        "equity",
        "fixed_income",
        "commodity",
        "real_estate",
    }
    assert config.constraints.ticker_bounds["REMX"].max == 0.05
    assert config.rebalance.realized_constraint_policy == "report_drift"
    assert config.rebalance.fallback_sell_allowed is False
    assert config.rebalance.fallback is not None
    assert config.costs.transaction_cost_bps == 2
    assert config.tracking.artifact_dir == "reports/runs"
    assert config.tracking.require_git_commit is True
    assert config.ml.enabled is False
    assert config.ml.target == "forward_return"
    assert config.ml.models == ["historical_mean", "ridge", "random_forest"]
    assert config.ml.governance.minimum_fold_win_rate == 0.6


def test_ml_config_is_disabled_by_default() -> None:
    assert MLConfig().enabled is False


def test_load_config_files_deep_merges_overlays(tmp_path: Path) -> None:
    base_config = tmp_path / "base.yaml"
    base_config.write_text(
        """
project:
  name: etf_portfolio_research
  base_currency: USD
universe:
  tickers:
    - VTI
    - VEA
    - BND
    - TIP
    - IAU
benchmark:
  primary: VT
  secondary:
    global_60_40:
      VT: 0.60
      BND: 0.40
data:
  provider: yfinance
  start_date: "2011-01-01"
  end_date: null
  price_field: adjusted_close
investor_profile:
  horizon_years: 35
  objective: long_term_accumulation
  tax_preference: minimize_realized_gains
optimization:
  long_only: true
  default_max_weight_per_etf: 0.25
  active_objective: max_sharpe
  benchmark_objectives:
    - equal_weight
constraints:
  asset_class_bounds:
    fixed_income:
      min: 0.10
      max: 0.50
  ticker_bounds:
    BND:
      min: 0.00
      max: 0.70
rebalance:
  mode: contribution_only
  frequency: monthly
  fallback_sell_allowed: false
  fallback:
    sell_allowed_if_absolute_drift_exceeds: 0.10
  contribution_amount: 250.0
costs:
  transaction_cost_bps: 2
  slippage_bps: 1
ml:
  enabled: true
  task: regression
  target: forward_return
  horizon_periods: 21
  models:
    - historical_mean
    - ridge
  validation:
    train_window_periods: 252
    test_window_periods: 21
    step_periods: 21
    min_train_periods: 252
""".strip(),
        encoding="utf-8",
    )
    overlay_config = tmp_path / "overlay.yaml"
    overlay_config.write_text(
        """
benchmark:
  secondary:
    simple_global_baseline:
      VTI: 0.55
      BND: 0.45
optimization:
  default_max_weight_per_etf: 0.2
constraints:
  ticker_bounds:
    VTI:
      min: 0.00
      max: 0.30
    IAU:
      min: 0.00
      max: 0.10
rebalance:
  fallback_sell_allowed: false
  fallback: null
ml:
  target: forward_volatility
""".strip(),
        encoding="utf-8",
    )

    config = load_config_files(base_config, overlay_config)

    assert config.benchmark.primary == "VT"
    assert config.benchmark.secondary["global_60_40"].allocations == {"VT": 0.6, "BND": 0.4}
    assert config.benchmark.secondary["simple_global_baseline"].allocations == {
        "VTI": 0.55,
        "BND": 0.45,
    }
    assert config.optimization.default_max_weight_per_etf == 0.2
    assert config.optimization.active_objective == "max_sharpe"
    assert config.optimization.benchmark_objectives == ["equal_weight"]
    assert config.constraints.asset_class_bounds["fixed_income"].max == 0.5
    assert set(config.constraints.ticker_bounds) == {"VTI", "IAU"}
    assert config.constraints.ticker_bounds["VTI"].max == 0.30
    assert config.constraints.ticker_bounds["IAU"].max == 0.10
    assert config.rebalance.fallback_sell_allowed is False
    assert config.rebalance.fallback is None
    assert config.ml.target == "forward_volatility"


def test_load_config_rejects_infeasible_max_weight_for_universe(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        """
project:
  name: etf_portfolio_research
  base_currency: USD
universe:
  tickers:
    - VTI
    - BND
    - TIP
benchmark:
  primary: VT
data:
  provider: yfinance
  start_date: "2011-01-01"
  end_date: null
  price_field: adjusted_close
investor_profile:
  horizon_years: 35
  objective: long_term_accumulation
  tax_preference: minimize_realized_gains
optimization:
  long_only: true
  default_max_weight_per_etf: 0.25
  active_objective: equal_weight
  benchmark_objectives:
    - min_variance
constraints:
  asset_class_bounds: {}
rebalance:
  mode: contribution_only
  frequency: monthly
  fallback_sell_allowed: false
  fallback:
    sell_allowed_if_absolute_drift_exceeds: 0.10
  contribution_amount: 100.0
costs:
  transaction_cost_bps: 2
  slippage_bps: 1
ml:
  enabled: true
  task: regression
  target: forward_return
  horizon_periods: 21
  models:
    - historical_mean
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "optimization.default_max_weight_per_etf and constraints.ticker_bounds "
            "are infeasible for the configured universe size"
        ),
    ):
        load_config(config_path)


def test_load_config_uses_ticker_bounds_for_default_cap_feasibility(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "valid_with_ticker_override.yaml"
    config_path.write_text(
        """
project:
  name: etf_portfolio_research
  base_currency: USD
universe:
  tickers:
    - VTI
    - BND
    - TIP
benchmark:
  primary: VT
data:
  provider: yfinance
  start_date: "2011-01-01"
  end_date: null
  price_field: adjusted_close
investor_profile:
  horizon_years: 35
  objective: long_term_accumulation
  tax_preference: minimize_realized_gains
optimization:
  long_only: true
  default_max_weight_per_etf: 0.25
  active_objective: equal_weight
  benchmark_objectives:
    - min_variance
constraints:
  asset_class_bounds: {}
  ticker_bounds:
    VTI:
      min: 0.00
      max: 0.60
rebalance:
  mode: contribution_only
  frequency: monthly
  fallback_sell_allowed: false
  fallback:
    sell_allowed_if_absolute_drift_exceeds: 0.10
  contribution_amount: 100.0
costs:
  transaction_cost_bps: 2
  slippage_bps: 1
ml:
  enabled: true
  task: regression
  target: forward_return
  horizon_periods: 21
  models:
    - historical_mean
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.optimization.default_max_weight_per_etf == 0.25
    assert config.constraints.ticker_bounds["VTI"].max == 0.60


def test_load_config_rejects_ticker_bounds_outside_universe(tmp_path: Path) -> None:
    config_path = tmp_path / "unknown_ticker_bound.yaml"
    config_text = Path("configs/base.yaml").read_text(encoding="utf-8")
    config_path.write_text(
        config_text.replace(
            "  ticker_bounds:\n",
            "  ticker_bounds:\n    MISSING:\n      min: 0.00\n      max: 0.10\n",
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="constraints.ticker_bounds contains tickers"):
        load_config(config_path)


def test_load_config_rejects_target_return_and_target_volatility_keys(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "unsupported_target_objective.yaml"
    config_text = Path("configs/base.yaml").read_text(encoding="utf-8")
    config_path.write_text(
        config_text.replace(
            "  active_objective: max_sharpe\n",
            "  target_return: 0.08\n  target_volatility: 0.12\n  active_objective: max_sharpe\n",
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not supported in run config files"):
        load_config(config_path)


def test_load_config_rejects_invalid_benchmark_mix_weights(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_mix.yaml"
    config_path.write_text(
        """
project:
  name: etf_portfolio_research
  base_currency: USD
universe:
  tickers:
    - VTI
    - BND
    - TIP
    - IAU
benchmark:
  primary: VT
  secondary:
    broken_mix:
      VT: 0.70
      BND: 0.20
data:
  provider: yfinance
  start_date: "2011-01-01"
  end_date: null
  price_field: adjusted_close
investor_profile:
  horizon_years: 35
  objective: long_term_accumulation
  tax_preference: minimize_realized_gains
optimization:
  long_only: true
  default_max_weight_per_etf: 0.25
  active_objective: equal_weight
  benchmark_objectives:
    - min_variance
constraints:
  asset_class_bounds: {}
rebalance:
  mode: contribution_only
  frequency: monthly
  fallback_sell_allowed: false
  fallback:
    sell_allowed_if_absolute_drift_exceeds: 0.10
  contribution_amount: 100.0
costs:
  transaction_cost_bps: 2
  slippage_bps: 1
ml:
  enabled: true
  task: regression
  target: forward_return
  horizon_periods: 21
  models:
    - historical_mean
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Benchmark mix weights must sum to 1.0"):
        load_config(config_path)


def test_config_to_dict_round_trips_revised_benchmark_and_rebalance() -> None:
    config = load_config("configs/base.yaml")
    payload = config_to_dict(config)

    assert payload["benchmark"]["primary"] == "VT"
    assert payload["benchmark"]["secondary"]["global_60_40"]["allocations"] == {
        "VT": 0.6,
        "BND": 0.4,
    }
    assert payload["rebalance"]["mode"] == "contribution_only"
    assert payload["rebalance"]["realized_constraint_policy"] == "report_drift"
    assert payload["tracking"]["artifact_dir"] == "reports/runs"
    assert payload["optimization"]["active_objective"] == "max_sharpe"
    assert payload["optimization"]["default_max_weight_per_etf"] == 0.25
    assert "max_weight_per_etf" not in payload["optimization"]
    assert payload["optimization"]["benchmark_objectives"] == [
        "equal_weight",
        "inverse_volatility",
        "min_variance",
        "risk_parity",
    ]
    assert payload["ml"]["enabled"] is False
    assert payload["ml"]["target"] == "forward_return"


def test_load_config_rejects_incompatible_ml_task_and_target(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_ml.yaml"
    config_path.write_text(
        """
project:
  name: etf_portfolio_research
  base_currency: USD
universe:
  tickers:
    - VTI
    - BND
    - TIP
    - IAU
benchmark:
  primary: VT
data:
  provider: yfinance
  start_date: "2011-01-01"
  end_date: null
  price_field: adjusted_close
investor_profile:
  horizon_years: 35
  objective: long_term_accumulation
  tax_preference: minimize_realized_gains
optimization:
  long_only: true
  default_max_weight_per_etf: 0.25
  active_objective: equal_weight
  benchmark_objectives:
    - min_variance
constraints:
  asset_class_bounds: {}
rebalance:
  mode: contribution_only
  frequency: monthly
  fallback_sell_allowed: false
  fallback:
    sell_allowed_if_absolute_drift_exceeds: 0.10
  contribution_amount: 100.0
costs:
  transaction_cost_bps: 2
  slippage_bps: 1
ml:
  enabled: true
  task: regression
  target: beat_benchmark
  horizon_periods: 21
  models:
    - historical_mean
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="ml.target='beat_benchmark' requires ml.task='classification'",
    ):
        load_config(config_path)


def test_base_yaml_loads_risk_free_rate() -> None:
    config = load_config("configs/base.yaml")

    assert config.risk_free.source == "constant"
    assert config.risk_free.value == pytest.approx(0.03)


def test_base_yaml_loads_cross_check_defaults() -> None:
    config = load_config("configs/base.yaml")

    assert config.data.cross_check.enabled is False
    assert config.data.cross_check.provider == "tiingo"
    assert config.data.cross_check.max_relative_divergence == pytest.approx(0.005)
    assert config.data.cross_check.min_overlap_observations == 20


def test_cross_check_config_requires_provider_when_enabled() -> None:
    with pytest.raises(ValidationError):
        CrossCheckConfig(enabled=True, provider=None)


def test_base_yaml_loads_rebalance_extensions_and_backtest_config() -> None:
    config = load_config("configs/base.yaml")

    assert config.rebalance.contribution_amount == pytest.approx(1500.0)
    assert config.rebalance.realized_constraint_policy == "report_drift"
    assert config.rebalance.tolerance_bands.per_ticker_abs_drift == pytest.approx(0.05)
    assert config.rebalance.tolerance_bands.per_asset_class_abs_drift == pytest.approx(0.10)
    assert config.backtest.initial_capital == pytest.approx(100_000.0)
    assert config.backtest.start_date is not None
    assert config.backtest.start_date.isoformat() == "2012-01-01"
    assert config.backtest.end_date is None


def test_load_config_rejects_duplicate_benchmark_objectives(tmp_path: Path) -> None:
    config_path = _write_minimal_config(
        tmp_path,
        optimization_block="""
optimization:
  long_only: true
  default_max_weight_per_etf: 0.30
  active_objective: max_sharpe
  benchmark_objectives:
    - equal_weight
    - equal_weight
""",
    )

    with pytest.raises(ValidationError):
        load_config(config_path)


def test_load_config_rejects_active_objective_in_benchmark_objectives(tmp_path: Path) -> None:
    config_path = _write_minimal_config(
        tmp_path,
        optimization_block="""
optimization:
  long_only: true
  default_max_weight_per_etf: 0.30
  active_objective: max_sharpe
  benchmark_objectives:
    - equal_weight
    - max_sharpe
""",
    )

    with pytest.raises(
        ValidationError,
        match="optimization.active_objective must not also appear",
    ):
        load_config(config_path)


def test_rebalance_config_rejects_negative_contribution_amount(tmp_path: Path) -> None:
    config_path = _write_minimal_config(
        tmp_path,
        rebalance_block="""
rebalance:
  mode: contribution_only
  frequency: monthly
  fallback_sell_allowed: false
  fallback:
    sell_allowed_if_absolute_drift_exceeds: 0.10
  contribution_amount: -100.0
""",
    )

    with pytest.raises(ValidationError):
        load_config(config_path)


def test_rebalance_config_rejects_contribution_only_with_zero_contribution(tmp_path: Path) -> None:
    config_path = _write_minimal_config(
        tmp_path,
        rebalance_block="""
rebalance:
  mode: contribution_only
  frequency: monthly
  fallback_sell_allowed: false
  fallback:
    sell_allowed_if_absolute_drift_exceeds: 0.10
  contribution_amount: 0.0
""",
    )

    with pytest.raises(ValueError, match="contribution_amount must be > 0"):
        load_config(config_path)


def test_rebalance_config_rejects_negative_tolerance_bands(tmp_path: Path) -> None:
    config_path = _write_minimal_config(
        tmp_path,
        rebalance_block="""
rebalance:
  mode: tolerance_band
  frequency: monthly
  fallback_sell_allowed: false
  contribution_amount: 0.0
  tolerance_bands:
    per_ticker_abs_drift: -0.01
    per_asset_class_abs_drift: 0.10
""",
    )

    with pytest.raises(ValidationError):
        load_config(config_path)


def test_rebalance_config_accepts_tolerance_band_mode(tmp_path: Path) -> None:
    config_path = _write_minimal_config(
        tmp_path,
        rebalance_block="""
rebalance:
  mode: tolerance_band
  frequency: monthly
  fallback_sell_allowed: false
  contribution_amount: 0.0
  tolerance_bands:
    per_ticker_abs_drift: 0.05
    per_asset_class_abs_drift: 0.10
""",
    )

    config = load_config(config_path)

    assert config.rebalance.mode == "tolerance_band"
    assert config.rebalance.tolerance_bands.per_ticker_abs_drift == pytest.approx(0.05)


def test_rebalance_config_rejects_enabled_fallback_sell_without_threshold(
    tmp_path: Path,
) -> None:
    config_path = _write_minimal_config(
        tmp_path,
        rebalance_block="""
rebalance:
  mode: contribution_only
  frequency: monthly
  fallback_sell_allowed: true
  fallback: null
  contribution_amount: 100.0
""",
    )

    with pytest.raises(
        ValueError,
        match="rebalance.fallback is required when rebalance.fallback_sell_allowed is true",
    ):
        load_config(config_path)


def test_backtest_config_rejects_inverted_dates(tmp_path: Path) -> None:
    config_path = _write_minimal_config(
        tmp_path,
        backtest_block="""
backtest:
  start_date: "2020-01-01"
  end_date: "2019-01-01"
  initial_capital: 25000.0
""",
    )

    with pytest.raises(ValueError, match="end_date must be strictly after"):
        load_config(config_path)


def test_backtest_config_rejects_non_positive_initial_capital(tmp_path: Path) -> None:
    config_path = _write_minimal_config(
        tmp_path,
        backtest_block="""
backtest:
  initial_capital: 0.0
""",
    )

    with pytest.raises(ValidationError):
        load_config(config_path)


def _write_minimal_config(
    tmp_path: Path,
    *,
    optimization_block: str | None = None,
    rebalance_block: str | None = None,
    backtest_block: str | None = None,
) -> Path:
    optimization = (
        optimization_block
        or """
optimization:
  long_only: true
  default_max_weight_per_etf: 0.30
  active_objective: equal_weight
  benchmark_objectives:
    - min_variance
"""
    )
    rebalance = (
        rebalance_block
        or """
rebalance:
  mode: contribution_only
  frequency: monthly
  fallback_sell_allowed: false
  fallback:
    sell_allowed_if_absolute_drift_exceeds: 0.10
  contribution_amount: 100.0
"""
    )
    backtest = backtest_block or ""

    contents = f"""
project:
  name: etf_portfolio_research
  base_currency: USD
universe:
  tickers:
    - VTI
    - BND
    - TIP
    - IAU
benchmark:
  primary: VT
data:
  provider: yfinance
  start_date: "2011-01-01"
  end_date: null
  price_field: adjusted_close
investor_profile:
  horizon_years: 35
  objective: long_term_accumulation
  tax_preference: minimize_realized_gains
{optimization.strip()}
constraints:
  asset_class_bounds: {{}}
{rebalance.strip()}
{backtest.strip()}
costs:
  transaction_cost_bps: 2
  slippage_bps: 1
ml:
  enabled: true
  task: regression
  target: forward_return
  horizon_periods: 21
  models:
    - historical_mean
"""
    config_path = tmp_path / "minimal.yaml"
    config_path.write_text(contents.strip(), encoding="utf-8")
    return config_path


def test_risk_free_config_rejects_negative_value() -> None:
    with pytest.raises(ValidationError):
        RiskFreeConfig(source="constant", value=-0.01)


def test_risk_free_config_rejects_value_above_one() -> None:
    with pytest.raises(ValidationError):
        RiskFreeConfig(source="constant", value=1.5)


def test_risk_free_config_rejects_unknown_source() -> None:
    with pytest.raises(ValidationError):
        RiskFreeConfig(source="treasury_bill", value=0.04)  # type: ignore[arg-type]


def test_risk_free_config_defaults_to_zero_when_omitted(tmp_path: Path) -> None:
    config_path = tmp_path / "no_rf.yaml"
    config_path.write_text(
        """
project:
  name: etf_portfolio_research
  base_currency: USD
universe:
  tickers:
    - VTI
    - BND
    - TIP
    - IAU
benchmark:
  primary: VT
data:
  provider: yfinance
  start_date: "2011-01-01"
  end_date: null
  price_field: adjusted_close
investor_profile:
  horizon_years: 35
  objective: long_term_accumulation
  tax_preference: minimize_realized_gains
optimization:
  long_only: true
  default_max_weight_per_etf: 0.25
  active_objective: equal_weight
  benchmark_objectives:
    - min_variance
constraints:
  asset_class_bounds: {}
rebalance:
  mode: contribution_only
  frequency: monthly
  fallback_sell_allowed: false
  fallback:
    sell_allowed_if_absolute_drift_exceeds: 0.10
  contribution_amount: 100.0
costs:
  transaction_cost_bps: 2
  slippage_bps: 1
ml:
  enabled: true
  task: regression
  target: forward_return
  horizon_periods: 21
  models:
    - historical_mean
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.risk_free.source == "constant"
    assert config.risk_free.value == 0.0
