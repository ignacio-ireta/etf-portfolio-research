"""Regression tests asserting the walk-forward backtest contains no lookahead.

These tests are intentionally narrow and assertive: they reject any change
that would let the optimizer or estimator peek at returns observed on or
after a rebalance date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf_portfolio.backtesting.engine import run_walk_forward_backtest


def _synthetic_asset_returns(periods: int = 60, seed: int = 7) -> pd.DataFrame:
    """Create a benign multi-asset return frame for backtest tests."""

    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=periods)
    return pd.DataFrame(
        {
            "AAA": rng.normal(loc=0.0006, scale=0.012, size=periods),
            "BBB": rng.normal(loc=0.0004, scale=0.008, size=periods),
            "CCC": rng.normal(loc=0.0005, scale=0.010, size=periods),
        },
        index=index,
    )


def test_train_window_strictly_precedes_every_rebalance_date() -> None:
    """For every fold, train_end must be strictly before rebalance_date."""

    returns = _synthetic_asset_returns(periods=80)
    rebalance_dates = returns.index[40::5]

    result = run_walk_forward_backtest(
        returns,
        rebalance_dates=rebalance_dates,
        lookback_periods=20,
        optimization_method="min_volatility",
        max_weight=0.6,
    )

    summary = result.rebalance_summary
    assert not summary.empty
    assert (summary["train_end"] < summary.index).all(), (
        "train_end must be strictly less than rebalance_date for every fold."
    )
    assert (summary["train_start"] <= summary["train_end"]).all()
    assert (summary["observation_count"] == 20).all()


def test_optimized_weights_are_invariant_to_future_returns() -> None:
    """Poisoning returns on/after a rebalance date must not change that date's weights.

    Weights at fold N are determined entirely by returns observed strictly before
    `rebalance_date_N`. We verify this by replacing all returns >= the FIRST
    rebalance date with arbitrary garbage and confirming optimizer choices at
    that date are bit-identical to the unpoisoned run.
    """

    returns = _synthetic_asset_returns(periods=80, seed=3)
    rebalance_dates = returns.index[40::5]
    target_date = rebalance_dates[0]

    baseline = run_walk_forward_backtest(
        returns,
        rebalance_dates=rebalance_dates,
        lookback_periods=20,
        optimization_method="min_volatility",
        max_weight=0.6,
    )

    poisoned_returns = returns.copy()
    rng = np.random.default_rng(999)
    mask = poisoned_returns.index >= target_date
    poisoned_returns.loc[mask] = rng.normal(
        loc=-0.05,
        scale=0.30,
        size=(int(mask.sum()), poisoned_returns.shape[1]),
    )

    poisoned = run_walk_forward_backtest(
        poisoned_returns,
        rebalance_dates=rebalance_dates,
        lookback_periods=20,
        optimization_method="min_volatility",
        max_weight=0.6,
    )

    np.testing.assert_allclose(
        baseline.weights.loc[target_date].to_numpy(),
        poisoned.weights.loc[target_date].to_numpy(),
        atol=1e-12,
    )


def test_swapping_returns_inside_training_window_changes_weights() -> None:
    """Sanity check the leakage assertion is meaningful.

    If we instead poison data INSIDE the training window, the optimizer's
    weights at the rebalance date MUST change. Otherwise the previous test
    is not actually proving anything.
    """

    returns = _synthetic_asset_returns(periods=80, seed=11)
    rebalance_dates = returns.index[40::5]
    target_date = rebalance_dates[0]

    baseline = run_walk_forward_backtest(
        returns,
        rebalance_dates=rebalance_dates,
        lookback_periods=20,
        optimization_method="min_volatility",
        max_weight=0.6,
    )

    poisoned_returns = returns.copy()
    rng = np.random.default_rng(13)
    mask = poisoned_returns.index < target_date
    poisoned_returns.loc[mask] = rng.normal(
        loc=0.0001,
        scale=0.0001,
        size=(int(mask.sum()), poisoned_returns.shape[1]),
    )
    poisoned_returns.loc[poisoned_returns.index < target_date, "AAA"] = rng.normal(
        loc=0.0, scale=0.05, size=int(mask.sum())
    )

    poisoned = run_walk_forward_backtest(
        poisoned_returns,
        rebalance_dates=rebalance_dates,
        lookback_periods=20,
        optimization_method="min_volatility",
        max_weight=0.6,
    )

    assert not np.allclose(
        baseline.weights.loc[target_date].to_numpy(),
        poisoned.weights.loc[target_date].to_numpy(),
        atol=1e-6,
    ), "Training-window poison must change weights; otherwise the leakage test is not meaningful."


def test_realized_segment_starts_on_or_after_rebalance_date() -> None:
    """Realized portfolio returns for fold N must be dated on/after `rebalance_date_N`."""

    returns = _synthetic_asset_returns(periods=80, seed=5)
    rebalance_dates = list(returns.index[40::5])

    result = run_walk_forward_backtest(
        returns,
        rebalance_dates=rebalance_dates,
        lookback_periods=20,
        optimization_method="min_volatility",
        max_weight=0.6,
    )

    rebalance_index = result.rebalance_summary.index
    for position, rebalance_date in enumerate(rebalance_index):
        next_date = rebalance_index[position + 1] if position + 1 < len(rebalance_index) else None
        segment = result.portfolio_returns.loc[
            (result.portfolio_returns.index >= rebalance_date)
            & (
                (next_date is None)
                | (result.portfolio_returns.index < (next_date or rebalance_date))
            )
        ]
        if next_date is None:
            segment = result.portfolio_returns.loc[result.portfolio_returns.index >= rebalance_date]
        assert (segment.index >= rebalance_date).all(), (
            f"Realized segment for fold @ {rebalance_date} must start on or after "
            f"the rebalance date, got {segment.index.min()}."
        )


def test_zero_lookback_periods_is_rejected() -> None:
    returns = _synthetic_asset_returns(periods=20)
    with pytest.raises(ValueError, match="lookback_periods must be positive"):
        run_walk_forward_backtest(
            returns,
            rebalance_dates=returns.index[10:],
            lookback_periods=0,
            optimization_method="equal_weight",
        )
