"""Tests that intra-period returns follow the same buy-and-hold path as the weights."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf_portfolio.backtesting.engine import _buy_and_hold_returns, run_walk_forward_backtest


def _two_asset_window() -> pd.DataFrame:
    index = pd.bdate_range("2021-01-04", periods=3)
    return pd.DataFrame(
        {
            "AAA": [0.10, 0.10, 0.10],
            "BBB": [0.00, 0.00, 0.00],
        },
        index=index,
    )


def test_buy_and_hold_returns_let_weights_drift_inside_the_window() -> None:
    window = _two_asset_window()
    weights = pd.Series({"AAA": 0.5, "BBB": 0.5})

    returns = _buy_and_hold_returns(window, weights)

    # Day 1: 0.5 * 10% = 5%. Portfolio value 1.05, of which AAA is 0.55.
    # Day 2: AAA earns 10% on 0.55 -> +0.055 on 1.05 = 5.238...%, NOT 5%.
    assert returns.iloc[0] == pytest.approx(0.05)
    assert returns.iloc[1] == pytest.approx(0.055 / 1.05)
    assert returns.iloc[2] == pytest.approx(0.0605 / 1.105)


def test_buy_and_hold_returns_compound_to_the_value_of_untouched_holdings() -> None:
    rng = np.random.default_rng(11)
    index = pd.bdate_range("2021-01-04", periods=21)
    window = pd.DataFrame(
        rng.normal(loc=0.0005, scale=0.02, size=(21, 3)),
        index=index,
        columns=["AAA", "BBB", "CCC"],
    )
    weights = pd.Series({"AAA": 0.2, "BBB": 0.3, "CCC": 0.5})

    returns = _buy_and_hold_returns(window, weights)

    expected_growth = float(((1.0 + window).prod() * weights).sum())
    assert float((1.0 + returns).prod()) == pytest.approx(expected_growth, rel=1e-12)


def test_buy_and_hold_returns_treat_unallocated_weight_as_zero_return_cash() -> None:
    window = _two_asset_window()
    weights = pd.Series({"AAA": 0.5, "BBB": 0.0})

    returns = _buy_and_hold_returns(window, weights)

    assert returns.iloc[0] == pytest.approx(0.05)
    assert returns.iloc[1] == pytest.approx(0.055 / 1.05)


def test_engine_portfolio_value_matches_untouched_holdings_between_rebalances() -> None:
    """Engine-level check: one period's compounded return equals the holdings' growth."""

    rng = np.random.default_rng(5)
    index = pd.bdate_range("2020-01-01", periods=120)
    returns = pd.DataFrame(
        rng.normal(loc=0.0004, scale=0.015, size=(120, 3)),
        index=index,
        columns=["AAA", "BBB", "CCC"],
    )
    rebalance_dates = [index[40], index[80]]

    result = run_walk_forward_backtest(
        returns,
        rebalance_dates=rebalance_dates,
        lookback_periods=30,
        optimization_method="equal_weight",
    )

    first_weights = result.applied_weights.iloc[0]
    first_window = returns.loc[(returns.index > index[40]) & (returns.index <= index[80])]
    period_returns = result.portfolio_returns.loc[first_window.index]

    expected_growth = float(((1.0 + first_window).prod() * first_weights).sum())
    assert float((1.0 + period_returns).prod()) == pytest.approx(expected_growth, rel=1e-12)
