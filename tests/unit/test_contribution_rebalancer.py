"""Behavioral tests for the contribution-only rebalancer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf_portfolio.backtesting.contributions import (
    CASH_CONSERVATION_TOLERANCE,
    allocate_contribution,
)
from etf_portfolio.backtesting.rebalancing import apply_rebalance_mode


def _equal_target() -> pd.Series:
    return pd.Series({"VTI": 0.6, "BND": 0.3, "IAU": 0.1}, dtype=float)


def test_contribution_never_sells_overweight_tickers() -> None:
    target = _equal_target()
    previous = pd.Series({"VTI": 0.80, "BND": 0.10, "IAU": 0.10}, dtype=float)
    portfolio_value = 100_000.0

    trades = allocate_contribution(
        previous_weights=previous,
        target_weights=target,
        cash_contribution=2_000.0,
        portfolio_value=portfolio_value,
    )

    assert (trades >= -1e-12).all(), f"Trades must be non-negative, got {trades.to_dict()}"
    assert trades["VTI"] == pytest.approx(0.0, abs=1e-9)


def test_large_contribution_can_allocate_surplus_after_underweight_gaps_are_filled() -> None:
    target = _equal_target()
    previous = pd.Series({"VTI": 0.80, "BND": 0.10, "IAU": 0.10}, dtype=float)
    portfolio_value = 100_000.0

    trades = allocate_contribution(
        previous_weights=previous,
        target_weights=target,
        cash_contribution=100_000.0,
        portfolio_value=portfolio_value,
    )

    assert trades["BND"] > 0.0
    assert trades["VTI"] > 0.0
    assert trades.sum() == pytest.approx(100_000.0, abs=1e-8)


def test_contribution_routes_capital_to_largest_underweight_first() -> None:
    target = _equal_target()
    previous = pd.Series({"VTI": 0.50, "BND": 0.40, "IAU": 0.10}, dtype=float)
    portfolio_value = 100_000.0

    trades = allocate_contribution(
        previous_weights=previous,
        target_weights=target,
        cash_contribution=2_000.0,
        portfolio_value=portfolio_value,
    )

    assert trades["BND"] == pytest.approx(0.0, abs=1e-9)
    assert trades["VTI"] > 0.0
    assert trades["IAU"] >= 0.0
    assert trades["VTI"] > trades["IAU"]


def test_contribution_conserves_cash_within_tolerance() -> None:
    target = _equal_target()
    previous = pd.Series({"VTI": 0.50, "BND": 0.40, "IAU": 0.10}, dtype=float)

    for contribution in (0.0, 1.0, 1_000.0, 12_345.6789, 1_000_000.0):
        trades = allocate_contribution(
            previous_weights=previous,
            target_weights=target,
            cash_contribution=contribution,
            portfolio_value=100_000.0,
        )
        delta = float(trades.sum() - contribution)
        assert abs(delta) <= CASH_CONSERVATION_TOLERANCE * 100, (
            f"Cash conservation broken for contribution={contribution}: "
            f"sum(trades)={float(trades.sum())}, delta={delta}"
        )


def test_zero_contribution_yields_zero_trades() -> None:
    target = _equal_target()
    previous = pd.Series({"VTI": 0.50, "BND": 0.40, "IAU": 0.10}, dtype=float)

    trades = allocate_contribution(
        previous_weights=previous,
        target_weights=target,
        cash_contribution=0.0,
        portfolio_value=100_000.0,
    )

    assert np.allclose(trades.to_numpy(), 0.0, atol=1e-12)


def test_contribution_when_all_at_target_distributes_proportionally() -> None:
    target = _equal_target()
    previous = target.copy()
    contribution = 1_000.0

    trades = allocate_contribution(
        previous_weights=previous,
        target_weights=target,
        cash_contribution=contribution,
        portfolio_value=100_000.0,
    )

    np.testing.assert_allclose(
        trades.to_numpy(),
        (contribution * target).to_numpy(),
        atol=1e-9,
    )


def test_apply_rebalance_mode_contribution_only_routes_through_allocator() -> None:
    target = _equal_target()
    previous = pd.Series({"VTI": 0.80, "BND": 0.10, "IAU": 0.10}, dtype=float)

    decision = apply_rebalance_mode(
        mode="contribution_only",
        previous_weights=previous,
        target_weights=target,
        portfolio_value=100_000.0,
        contribution_amount=2_000.0,
    )

    assert decision.mode == "contribution_only"
    assert decision.trades_dollars["VTI"] == pytest.approx(0.0, abs=1e-9)
    assert (decision.trades_dollars >= -1e-12).all()
    assert decision.portfolio_value_after == pytest.approx(102_000.0)
    new_total = decision.portfolio_value_after
    np.testing.assert_allclose(decision.applied_weights.sum(), 1.0, atol=1e-8)
    new_holdings = decision.applied_weights * new_total
    expected_vti_holdings = 0.80 * 100_000.0
    assert new_holdings["VTI"] == pytest.approx(expected_vti_holdings, abs=1e-6)


def test_contribution_only_no_fallback_sell_when_disabled() -> None:
    target = pd.Series({"VTI": 0.5, "BND": 0.5}, dtype=float)
    previous = pd.Series({"VTI": 0.95, "BND": 0.05}, dtype=float)

    decision = apply_rebalance_mode(
        mode="contribution_only",
        previous_weights=previous,
        target_weights=target,
        portfolio_value=100_000.0,
        contribution_amount=1_000.0,
        fallback_sell_allowed=False,
        fallback_drift_threshold=0.10,
    )

    assert decision.rebalanced
    assert (decision.trades_dollars >= -1e-12).all()
    assert decision.applied_weights["VTI"] > target["VTI"]


def test_contribution_only_force_sell_rebalances_when_constraints_are_hard() -> None:
    target = pd.Series({"VTI": 0.5, "BND": 0.5}, dtype=float)
    previous = pd.Series({"VTI": 0.95, "BND": 0.05}, dtype=float)

    decision = apply_rebalance_mode(
        mode="contribution_only",
        previous_weights=previous,
        target_weights=target,
        portfolio_value=100_000.0,
        contribution_amount=1_000.0,
        fallback_sell_allowed=False,
        fallback_drift_threshold=None,
        force_sell_rebalance=True,
    )

    assert decision.rebalanced
    np.testing.assert_allclose(
        decision.applied_weights.to_numpy(),
        target.to_numpy(),
        atol=1e-9,
    )
    assert decision.trades_dollars["VTI"] < 0.0


def test_contribution_only_fallback_sells_when_drift_exceeds_threshold() -> None:
    target = pd.Series({"VTI": 0.5, "BND": 0.5}, dtype=float)
    previous = pd.Series({"VTI": 0.95, "BND": 0.05}, dtype=float)

    decision = apply_rebalance_mode(
        mode="contribution_only",
        previous_weights=previous,
        target_weights=target,
        portfolio_value=100_000.0,
        contribution_amount=1_000.0,
        fallback_sell_allowed=True,
        fallback_drift_threshold=0.10,
    )

    assert decision.rebalanced
    np.testing.assert_allclose(
        decision.applied_weights.to_numpy(),
        target.to_numpy(),
        atol=1e-9,
    )
    assert decision.trades_dollars["VTI"] < 0.0


def test_contribution_only_no_fallback_sell_when_drift_under_threshold() -> None:
    target = pd.Series({"VTI": 0.5, "BND": 0.5}, dtype=float)
    previous = pd.Series({"VTI": 0.55, "BND": 0.45}, dtype=float)

    decision = apply_rebalance_mode(
        mode="contribution_only",
        previous_weights=previous,
        target_weights=target,
        portfolio_value=100_000.0,
        contribution_amount=1_000.0,
        fallback_sell_allowed=True,
        fallback_drift_threshold=0.10,
    )

    assert decision.rebalanced
    assert (decision.trades_dollars >= -1e-12).all()
