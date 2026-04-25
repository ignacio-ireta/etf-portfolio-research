from __future__ import annotations

import pandas as pd
import pytest

from etf_portfolio.backtesting.engine import run_walk_forward_backtest
from etf_portfolio.cli import _build_benchmark_suite
from etf_portfolio.config import AppConfig


def make_dummy_returns() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {"ETF1": [0.01] * 10, "ETF2": [0.02] * 10, "BENCH": [0.015] * 10}, index=dates
    )


def test_rebalance_date_return_uses_previous_weights() -> None:
    returns = make_dummy_returns()
    rebalance_dates = [returns.index[2], returns.index[5]]

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

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("etf_portfolio.backtesting.engine.optimize_portfolio", fake_optimize)

    result = run_walk_forward_backtest(
        returns,
        rebalance_dates=rebalance_dates,
        lookback_periods=1,
        optimization_method="equal_weight",
    )
    monkeypatch.undo()

    # R1 = index[2]. Weights at R1 set to ETF1=1.0.
    # But R1 return should NOT be included in this segment's returns.
    # The first return in result.portfolio_returns should be index[3].

    assert result.portfolio_returns.index[0] == returns.index[3]

    # R2 = index[5]. Weights at R2 set to ETF2=1.0.
    # Return at index[5] should be realized with OLD weights (ETF1=1.0).
    # ETF1 return at index[5] is 0.01.

    assert result.portfolio_returns.loc[returns.index[5]] == pytest.approx(0.01)

    # Return at index[6] should be realized with NEW weights (ETF2=1.0).
    # ETF2 return at index[6] is 0.02.
    assert result.portfolio_returns.loc[returns.index[6]] == pytest.approx(0.02)


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

    captured_modes = []

    def fake_run_backtest(*args, **kwargs):
        captured_modes.append(kwargs.get("rebalance_mode"))
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
    assert len(captured_modes) >= 2
    assert all(mode == "contribution_only" for mode in captured_modes)
