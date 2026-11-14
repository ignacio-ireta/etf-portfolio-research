"""Tests for fixed-allocation benchmarks simulated on the strategy's rebalance schedule."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf_portfolio import cli


def _random_returns() -> pd.DataFrame:
    rng = np.random.default_rng(21)
    index = pd.bdate_range("2021-01-04", periods=30)
    return pd.DataFrame(
        rng.normal(loc=0.0005, scale=0.02, size=(30, 2)),
        index=index,
        columns=["AAA", "BBB"],
    )


ALLOCATIONS = {"AAA": 0.6, "BBB": 0.4}


def test_without_rebalance_dates_the_mix_is_buy_and_hold() -> None:
    returns = _random_returns()

    series = cli._static_mix_benchmark_returns(
        returns,
        ALLOCATIONS,
        rebalance_dates=pd.DatetimeIndex([]),
        transaction_cost_rate=0.0,
    )

    weights = pd.Series(ALLOCATIONS)
    expected_growth = float(((1.0 + returns).prod() * weights).sum())
    assert float((1.0 + series).prod()) == pytest.approx(expected_growth, rel=1e-12)


def test_rebalancing_every_day_for_free_equals_constant_weights() -> None:
    returns = _random_returns()

    series = cli._static_mix_benchmark_returns(
        returns,
        ALLOCATIONS,
        rebalance_dates=returns.index,
        transaction_cost_rate=0.0,
    )

    expected = returns.mul(pd.Series(ALLOCATIONS), axis=1).sum(axis=1)
    pd.testing.assert_series_equal(series, expected, check_names=False)


def test_cost_of_a_rebalance_is_charged_on_the_next_return_date() -> None:
    index = pd.bdate_range("2021-01-04", periods=3)
    returns = pd.DataFrame(
        {"AAA": [0.10, 0.0, 0.0], "BBB": [0.0, 0.0, 0.0]},
        index=index,
    )

    series = cli._static_mix_benchmark_returns(
        returns,
        {"AAA": 0.5, "BBB": 0.5},
        rebalance_dates=pd.DatetimeIndex([index[0]]),
        transaction_cost_rate=0.01,
    )

    # Day 1: 0.5 * 10% = 5%. Drifted weights: AAA 0.55/1.05, BBB 0.50/1.05.
    # Rebalancing back to 50/50 trades |0.5 - 0.55/1.05| * 2 of the portfolio.
    traded_weight = 2.0 * (0.55 / 1.05 - 0.5)
    assert series.iloc[0] == pytest.approx(0.05)
    assert series.iloc[1] == pytest.approx(-0.01 * traded_weight)
    assert series.iloc[2] == pytest.approx(0.0)


def test_rebalance_dates_outside_the_data_are_ignored() -> None:
    returns = _random_returns()

    series = cli._static_mix_benchmark_returns(
        returns,
        ALLOCATIONS,
        rebalance_dates=pd.DatetimeIndex(["1999-01-01"]),
        transaction_cost_rate=0.01,
    )

    assert len(series) == len(returns)
    assert series.index.equals(returns.index)


def test_old_costless_daily_rebalanced_helper_is_gone() -> None:
    assert not hasattr(cli, "_composite_benchmark_returns")
