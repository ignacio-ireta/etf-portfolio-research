from __future__ import annotations

import pandas as pd
import pytest

from etf_portfolio.backtesting.engine import run_walk_forward_backtest
from etf_portfolio.cli import _build_benchmark_suite, _report_assumptions
from etf_portfolio.config import AppConfig


def make_dummy_returns() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {"ETF1": [0.01] * 10, "ETF2": [0.02] * 10, "BENCH": [0.015] * 10}, index=dates
    )


def test_rebalance_date_return_uses_previous_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    returns = make_dummy_returns()
    rebalance_dates = [returns.index[3], returns.index[6]]

    # Force a specific weight change
    weights_sequence = [
        pd.Series({"ETF1": 1.0, "ETF2": 0.0, "BENCH": 0.0}),
        pd.Series({"ETF1": 0.0, "ETF2": 1.0, "BENCH": 0.0}),
    ]

    idx = 0

    def fake_optimize(*args, **kwargs):
        nonlocal idx
        res = weights_sequence[idx]
        idx += 1
        return res

    monkeypatch.setattr("etf_portfolio.backtesting.engine.optimize_portfolio", fake_optimize)

    result = run_walk_forward_backtest(
        returns,
        rebalance_dates=rebalance_dates,
        lookback_periods=2,
        optimization_method="equal_weight",
    )

    # R1 = index[3]. Weights at R1 set to ETF1=1.0, but the R1 return is
    # not included in this new segment. The first realized return for those
    # weights is strictly after R1.
    assert result.portfolio_returns.index[0] == returns.index[4]

    # R2 = index[6]. Return at R2 remains in the previous holding period and
    # is realized with the old ETF1=1.0 weights.
    assert result.portfolio_returns.loc[returns.index[6]] == pytest.approx(0.01)

    # The first return strictly after R2 is realized with the new ETF2=1.0 weights.
    assert result.portfolio_returns.loc[returns.index[7]] == pytest.approx(0.02)


def test_benchmark_suite_aligns_rebalance_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    # This tests that _build_benchmark_suite passes the rebalance mode to run_walk_forward_backtest
    returns = make_dummy_returns()
    asset_returns = returns[["ETF1", "ETF2"]]

    config_dict = {
        "project": {"name": "Test", "base_currency": "USD"},
        "universe": {"tickers": ["ETF1", "ETF2"]},
        "benchmark": {"primary": "BENCH", "secondary": {}},
        "data": {"provider": "yfinance", "start_date": "2020-01-01", "price_field": "adj_close"},
        "investor_profile": {"horizon_years": 10, "objective": "Growth", "tax_preference": "None"},
        "optimization": {
            "active_objective": "equal_weight",
            "benchmark_objectives": ["min_variance"],
            "risk_model": "sample",
            "expected_return_estimator": "historical_mean",
            "default_max_weight_per_etf": 1.0,
        },
        "constraints": {"asset_class_bounds": {}, "ticker_bounds": {}},
        "rebalance": {
            "mode": "contribution_only",
            "frequency": "monthly",
            "contribution_amount": 1000.0,
            "realized_constraint_policy": "report_drift",
        },
        "backtest": {"initial_capital": 100000.0},
        "costs": {"transaction_cost_bps": 10, "slippage_bps": 5},
    }
    config = AppConfig.model_validate(config_dict)

    captured_kwargs: list[dict[str, object]] = []

    def fake_run_backtest(*args, **kwargs):
        captured_kwargs.append(kwargs.copy())
        # Return a dummy result
        from etf_portfolio.backtesting.engine import WalkForwardBacktestResult

        return WalkForwardBacktestResult(
            portfolio_returns=pd.Series([0.01], index=[returns.index[-1]]),
            target_weights=pd.DataFrame([{"ETF1": 0.5}], index=[returns.index[0]]),
            applied_weights=pd.DataFrame([{"ETF1": 0.5}], index=[returns.index[0]]),
            rebalance_summary=pd.DataFrame([{"rebalanced": True}], index=[returns.index[0]]),
        )

    monkeypatch.setattr("etf_portfolio.cli.run_walk_forward_backtest", fake_run_backtest)

    _build_benchmark_suite(
        returns,
        asset_returns,
        config=config,
        rebalance_dates=pd.DatetimeIndex([returns.index[5]]),
        lookback_periods=1,
        transaction_cost_rate=0.0015,
    )

    # Should have called for 'min_variance' and 'Previous Optimized Strategy'
    assert len(captured_kwargs) == 2
    assert [kwargs["optimization_method"] for kwargs in captured_kwargs] == [
        "min_volatility",
        "equal_weight",
    ]
    assert all(kwargs["rebalance_mode"] == "contribution_only" for kwargs in captured_kwargs)
    assert all(kwargs["contribution_amount"] == 1000.0 for kwargs in captured_kwargs)
    assert all(kwargs["initial_capital"] == 100000.0 for kwargs in captured_kwargs)
    assert all(
        kwargs["transaction_cost_rate"] == pytest.approx(0.0015) for kwargs in captured_kwargs
    )
    assert all(kwargs["realized_constraint_policy"] == "report_drift" for kwargs in captured_kwargs)
    assert all(kwargs["covariance_method"] == "sample" for kwargs in captured_kwargs)
    assert all(kwargs["expected_return_method"] == "historical_mean" for kwargs in captured_kwargs)


def test_report_assumptions_explain_benchmark_and_rebalance_semantics() -> None:
    config = AppConfig.model_validate(
        {
            "project": {"name": "Test", "base_currency": "USD"},
            "universe": {"tickers": ["ETF1", "ETF2"]},
            "benchmark": {"primary": "BENCH", "secondary": {}},
            "data": {
                "provider": "yfinance",
                "start_date": "2020-01-01",
                "price_field": "adj_close",
            },
            "investor_profile": {
                "horizon_years": 10,
                "objective": "Growth",
                "tax_preference": "None",
            },
            "optimization": {
                "active_objective": "equal_weight",
                "benchmark_objectives": ["min_variance"],
                "risk_model": "sample",
                "expected_return_estimator": "historical_mean",
                "default_max_weight_per_etf": 1.0,
            },
            "constraints": {"asset_class_bounds": {}, "ticker_bounds": {}},
            "rebalance": {
                "mode": "contribution_only",
                "frequency": "monthly",
                "contribution_amount": 1000.0,
                "realized_constraint_policy": "report_drift",
            },
            "backtest": {"initial_capital": 100000.0},
            "costs": {"transaction_cost_bps": 10, "slippage_bps": 5},
        }
    )

    assumptions = _report_assumptions(config, "equal_weight")

    assert "returns strictly before each rebalance date" in assumptions["rebalance_execution"]
    assert "return labeled with the rebalance date remains" in assumptions["rebalance_execution"]
    assert "same rebalance mode (contribution_only)" in assumptions["benchmark_fairness"]
    assert "contribution amount (1000.00)" in assumptions["benchmark_fairness"]
    assert "not simulated with contribution-only drift" in assumptions["benchmark_return_series"]
