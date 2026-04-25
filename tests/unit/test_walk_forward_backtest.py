from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf_portfolio.backtesting.engine import run_walk_forward_backtest


def make_asset_returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AAA": [0.010, 0.020, 0.030, 0.040, 0.050, 0.060],
            "BBB": [0.000, 0.010, 0.000, 0.010, 0.000, 0.010],
            "CCC": [0.020, 0.000, 0.010, -0.010, 0.020, 0.000],
        },
        index=pd.date_range("2024-01-31", periods=6, freq="ME"),
    )


def test_walk_forward_uses_only_prior_data() -> None:
    returns = make_asset_returns()
    rebalance_dates = returns.index[3:]

    result = run_walk_forward_backtest(
        returns,
        rebalance_dates=rebalance_dates,
        lookback_periods=3,
        optimization_method="equal_weight",
    )

    assert not result.rebalance_summary.empty
    assert (result.rebalance_summary["train_end"] < result.rebalance_summary.index).all()
    assert (result.rebalance_summary["observation_count"] == 3).all()
    assert (result.rebalance_summary["turnover"] >= 0.0).all()
    assert (result.rebalance_summary["largest_position"] <= 1.0).all()


def test_walk_forward_applies_transaction_costs_on_rebalance() -> None:
    returns = make_asset_returns()
    rebalance_dates = returns.index[3:]
    target = pd.Series({"AAA": 0.8, "BBB": 0.2, "CCC": 0.0}, dtype=float)

    def fake_optimize_portfolio(*args, **kwargs):
        return target.copy()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "etf_portfolio.backtesting.engine.optimize_portfolio", fake_optimize_portfolio
    )

    result = run_walk_forward_backtest(
        returns,
        rebalance_dates=rebalance_dates,
        lookback_periods=3,
        optimization_method="equal_weight",
        max_weight=0.8,
        transaction_cost_rate=0.01,
    )
    monkeypatch.undo()

    first_rebalance_date = result.rebalance_summary.index[0]
    # The first return after rebalance is the one where costs are applied
    first_return_date = returns.index[returns.index > first_rebalance_date][0]

    first_weights = result.weights.loc[first_rebalance_date]
    gross_return = returns.loc[first_return_date].dot(first_weights)
    net_return = result.portfolio_returns.loc[first_return_date]
    summary = result.rebalance_summary.loc[first_rebalance_date]

    assert summary["transaction_cost_dollars"] == pytest.approx(0.01)
    assert summary["transaction_cost_rate"] == pytest.approx(0.01)
    assert summary["transaction_cost"] == pytest.approx(0.01)
    assert net_return == pytest.approx(gross_return - 0.01)


def test_walk_forward_requires_enough_history_before_rebalance() -> None:
    returns = make_asset_returns()

    with pytest.raises(ValueError, match="produced no rebalance periods"):
        run_walk_forward_backtest(
            returns,
            rebalance_dates=returns.index[:2],
            lookback_periods=3,
            optimization_method="equal_weight",
        )


def test_walk_forward_previous_weights_lag_avoids_same_period_optimized_weights() -> None:
    returns = make_asset_returns()
    rebalance_dates = returns.index[3:]

    current = run_walk_forward_backtest(
        returns,
        rebalance_dates=rebalance_dates,
        lookback_periods=3,
        optimization_method="min_volatility",
        max_weight=0.8,
    )
    lagged = run_walk_forward_backtest(
        returns,
        rebalance_dates=rebalance_dates,
        lookback_periods=3,
        optimization_method="min_volatility",
        max_weight=0.8,
        apply_previous_weights_lag=True,
    )

    first_rebalance_date = lagged.weights.index[0]
    assert lagged.weights.loc[first_rebalance_date].sum() == pytest.approx(1.0)
    assert not lagged.weights.loc[first_rebalance_date].equals(
        current.weights.loc[first_rebalance_date]
    )


def test_walk_forward_passes_ticker_bounds_to_optimizer(monkeypatch: pytest.MonkeyPatch) -> None:
    returns = make_asset_returns()
    rebalance_dates = returns.index[3:]
    captured: dict[str, object] = {}

    def fake_optimize_portfolio(*args, **kwargs):
        captured["ticker_bounds"] = kwargs.get("ticker_bounds")
        return pd.Series(
            [1.0 / len(returns.columns)] * len(returns.columns),
            index=returns.columns,
            dtype=float,
        )

    monkeypatch.setattr(
        "etf_portfolio.backtesting.engine.optimize_portfolio", fake_optimize_portfolio
    )

    run_walk_forward_backtest(
        returns,
        rebalance_dates=rebalance_dates,
        lookback_periods=3,
        optimization_method="equal_weight",
        ticker_bounds={"AAA": (0.0, 0.4)},
    )

    assert captured["ticker_bounds"] == {"AAA": (0.0, 0.4)}


def test_walk_forward_passes_prior_optimized_weights_as_initial_guess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    returns = make_asset_returns()
    rebalance_dates = returns.index[3:]
    captured_initial_guesses: list[pd.Series | None] = []
    targets = [
        pd.Series({"AAA": 0.7, "BBB": 0.2, "CCC": 0.1}, dtype=float),
        pd.Series({"AAA": 0.6, "BBB": 0.3, "CCC": 0.1}, dtype=float),
        pd.Series({"AAA": 0.5, "BBB": 0.4, "CCC": 0.1}, dtype=float),
    ]

    def fake_optimize_portfolio(*args, **kwargs):
        initial_guess = kwargs.get("initial_guess")
        captured_initial_guesses.append(None if initial_guess is None else initial_guess.copy())
        return targets[len(captured_initial_guesses) - 1].copy()

    monkeypatch.setattr(
        "etf_portfolio.backtesting.engine.optimize_portfolio", fake_optimize_portfolio
    )

    run_walk_forward_backtest(
        returns,
        rebalance_dates=rebalance_dates,
        lookback_periods=3,
        optimization_method="max_sharpe",
        max_weight=0.8,
    )

    assert captured_initial_guesses[0] is None
    pd.testing.assert_series_equal(captured_initial_guesses[1], targets[0])
    pd.testing.assert_series_equal(captured_initial_guesses[2], targets[1])


def test_contribution_only_first_rebalance_bootstraps_initial_capital(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    returns = make_asset_returns()
    rebalance_dates = returns.index[3:]
    target = pd.Series({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, dtype=float)

    def fake_optimize_portfolio(*args, **kwargs):
        return target.copy()

    monkeypatch.setattr(
        "etf_portfolio.backtesting.engine.optimize_portfolio", fake_optimize_portfolio
    )

    result = run_walk_forward_backtest(
        returns,
        rebalance_dates=rebalance_dates,
        lookback_periods=3,
        optimization_method="equal_weight",
        rebalance_mode="contribution_only",
        initial_capital=100_000.0,
        contribution_amount=1_000.0,
    )

    first_rebalance_date = result.weights.index[0]
    np.testing.assert_allclose(
        result.weights.loc[first_rebalance_date].to_numpy(),
        target.to_numpy(),
        atol=1e-12,
    )
    assert result.trades_dollars.loc[first_rebalance_date].sum() == pytest.approx(101_000.0)


def test_contribution_only_applied_weights_sum_to_one_at_each_rebalance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    returns = make_asset_returns()
    rebalance_dates = returns.index[3:]

    def fake_optimize_portfolio(*args, **kwargs):
        return pd.Series({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, dtype=float)

    monkeypatch.setattr(
        "etf_portfolio.backtesting.engine.optimize_portfolio", fake_optimize_portfolio
    )

    result = run_walk_forward_backtest(
        returns,
        rebalance_dates=rebalance_dates,
        lookback_periods=3,
        optimization_method="equal_weight",
        rebalance_mode="contribution_only",
        initial_capital=100_000.0,
        contribution_amount=1_000.0,
    )

    assert (result.weights.sum(axis=1) - 1.0).abs().max() == pytest.approx(0.0, abs=1e-10)


def test_contribution_only_first_rebalance_trades_include_initial_capital(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    returns = make_asset_returns()
    rebalance_dates = returns.index[3:]

    def fake_optimize_portfolio(*args, **kwargs):
        return pd.Series({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, dtype=float)

    monkeypatch.setattr(
        "etf_portfolio.backtesting.engine.optimize_portfolio", fake_optimize_portfolio
    )

    initial_capital = 250_000.0
    contribution = 2_500.0
    result = run_walk_forward_backtest(
        returns,
        rebalance_dates=rebalance_dates,
        lookback_periods=3,
        optimization_method="equal_weight",
        rebalance_mode="contribution_only",
        initial_capital=initial_capital,
        contribution_amount=contribution,
    )

    first_rebalance_date = result.trades_dollars.index[0]
    assert result.trades_dollars.loc[first_rebalance_date].sum() == pytest.approx(
        initial_capital + contribution
    )


def test_contribution_only_transaction_costs_use_actual_dollar_trades(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    returns = make_asset_returns()
    rebalance_dates = returns.index[3:]

    def fake_optimize_portfolio(*args, **kwargs):
        return pd.Series({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, dtype=float)

    monkeypatch.setattr(
        "etf_portfolio.backtesting.engine.optimize_portfolio", fake_optimize_portfolio
    )

    contribution = 2_500.0
    result = run_walk_forward_backtest(
        returns,
        rebalance_dates=rebalance_dates,
        lookback_periods=3,
        optimization_method="equal_weight",
        rebalance_mode="contribution_only",
        initial_capital=250_000.0,
        contribution_amount=contribution,
        transaction_cost_rate=0.01,
    )

    second_rebalance_date = result.rebalance_summary.index[1]
    second_summary = result.rebalance_summary.loc[second_rebalance_date]
    second_trades = result.trades_dollars.loc[second_rebalance_date]

    # First return after second rebalance
    second_return_date = returns.index[returns.index > second_rebalance_date][0]
    gross_return = returns.loc[second_return_date].dot(result.weights.loc[second_rebalance_date])
    net_return = result.portfolio_returns.loc[second_return_date]

    assert (second_trades >= 0.0).all()
    assert second_trades.sum() == pytest.approx(contribution)
    assert second_summary["transaction_cost_dollars"] == pytest.approx(contribution * 0.01)
    assert second_summary["transaction_cost_rate"] == pytest.approx(0.01)
    assert second_summary["transaction_cost"] == pytest.approx(
        second_summary["transaction_cost_dollars"] / second_summary["portfolio_value"]
    )
    assert net_return == pytest.approx(gross_return - second_summary["transaction_cost"])


def test_walk_forward_stores_target_and_applied_weight_histories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    returns = make_asset_returns()
    rebalance_dates = returns.index[3:]
    target = pd.Series({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, dtype=float)

    monkeypatch.setattr(
        "etf_portfolio.backtesting.engine.optimize_portfolio",
        lambda *args, **kwargs: target.copy(),
    )

    result = run_walk_forward_backtest(
        returns,
        rebalance_dates=rebalance_dates,
        lookback_periods=3,
        optimization_method="equal_weight",
        rebalance_mode="contribution_only",
        initial_capital=100_000.0,
        contribution_amount=0.0,
    )

    first_rebalance_date = result.target_weights.index[0]
    np.testing.assert_allclose(
        result.target_weights.loc[first_rebalance_date].to_numpy(),
        target.to_numpy(),
        atol=1e-12,
    )
    pd.testing.assert_frame_equal(result.applied_weights, result.weights)
    second_rebalance_date = result.target_weights.index[1]
    assert not result.applied_weights.loc[second_rebalance_date].equals(
        result.target_weights.loc[second_rebalance_date]
    )


def test_contribution_only_records_realized_constraint_violations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    returns = make_asset_returns()
    rebalance_dates = returns.index[3:]
    target = pd.Series({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, dtype=float)

    monkeypatch.setattr(
        "etf_portfolio.backtesting.engine.optimize_portfolio",
        lambda *args, **kwargs: target.copy(),
    )

    result = run_walk_forward_backtest(
        returns,
        rebalance_dates=rebalance_dates,
        lookback_periods=3,
        optimization_method="equal_weight",
        rebalance_mode="contribution_only",
        initial_capital=100_000.0,
        contribution_amount=0.0,
        realized_constraint_policy="report_drift",
        ticker_bounds={"AAA": (0.0, 0.45)},
        asset_classes=pd.Series({"AAA": "equity", "BBB": "fixed_income", "CCC": "real_assets"}),
        asset_class_bounds={"equity": (0.0, 0.45)},
    )

    assert not result.realized_constraint_violations.empty
    assert {"ticker", "asset_class"} == set(
        result.realized_constraint_violations["constraint_type"]
    )
    assert (result.realized_constraint_violations["direction"] == "above_max").all()
    assert result.rebalance_summary["realized_constraint_violation_count"].max() >= 1


def test_contribution_only_enforce_hard_caps_forces_sell_rebalance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    returns = make_asset_returns()
    rebalance_dates = returns.index[3:]
    target = pd.Series({"AAA": 0.45, "BBB": 0.35, "CCC": 0.20}, dtype=float)

    monkeypatch.setattr(
        "etf_portfolio.backtesting.engine.optimize_portfolio",
        lambda *args, **kwargs: target.copy(),
    )

    result = run_walk_forward_backtest(
        returns,
        rebalance_dates=rebalance_dates,
        lookback_periods=3,
        optimization_method="equal_weight",
        rebalance_mode="contribution_only",
        initial_capital=100_000.0,
        contribution_amount=0.0,
        realized_constraint_policy="enforce_hard",
        ticker_bounds={"AAA": (0.0, 0.45)},
    )

    second_rebalance_date = result.applied_weights.index[1]
    np.testing.assert_allclose(
        result.applied_weights.loc[second_rebalance_date].to_numpy(),
        target.to_numpy(),
        atol=1e-12,
    )
    assert result.trades_dollars.loc[second_rebalance_date, "AAA"] < 0.0
    assert result.rebalance_summary.loc[second_rebalance_date, "forced_constraint_sell_rebalance"]
    assert (
        result.rebalance_summary.loc[second_rebalance_date, "pretrade_constraint_violation_count"]
        >= 1
    )
    assert result.realized_constraint_violations.empty
