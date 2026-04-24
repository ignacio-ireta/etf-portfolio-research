"""Behavioral tests for the tolerance-band rebalancer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf_portfolio.backtesting.rebalancing import apply_rebalance_mode
from etf_portfolio.config import RebalanceToleranceBandsConfig


def _bands(per_ticker: float = 0.05, per_class: float = 0.10) -> RebalanceToleranceBandsConfig:
    return RebalanceToleranceBandsConfig(
        per_ticker_abs_drift=per_ticker,
        per_asset_class_abs_drift=per_class,
    )


def test_no_trades_when_all_drifts_inside_band() -> None:
    target = pd.Series({"VTI": 0.6, "BND": 0.3, "IAU": 0.1}, dtype=float)
    previous = pd.Series({"VTI": 0.62, "BND": 0.29, "IAU": 0.09}, dtype=float)

    decision = apply_rebalance_mode(
        mode="tolerance_band",
        previous_weights=previous,
        target_weights=target,
        portfolio_value=100_000.0,
        contribution_amount=0.0,
        tolerance_bands=_bands(per_ticker=0.05),
    )

    assert decision.rebalanced is False
    np.testing.assert_allclose(
        decision.applied_weights.to_numpy(),
        previous.to_numpy(),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        decision.trades_dollars.to_numpy(),
        np.zeros(len(target)),
        atol=1e-12,
    )


def test_rebalance_to_band_edge_when_breached_above() -> None:
    target = pd.Series({"VTI": 0.50, "BND": 0.50}, dtype=float)
    previous = pd.Series({"VTI": 0.62, "BND": 0.38}, dtype=float)
    band = 0.05

    decision = apply_rebalance_mode(
        mode="tolerance_band",
        previous_weights=previous,
        target_weights=target,
        portfolio_value=100_000.0,
        contribution_amount=0.0,
        tolerance_bands=_bands(per_ticker=band),
    )

    assert decision.rebalanced is True
    assert decision.applied_weights["VTI"] == pytest.approx(target["VTI"] + band, abs=1e-9)
    assert decision.applied_weights["VTI"] != pytest.approx(target["VTI"], abs=1e-3), (
        "tolerance-band rebalance must NOT pull all the way to target."
    )
    assert decision.applied_weights.sum() == pytest.approx(1.0, abs=1e-9)


def test_rebalance_to_band_edge_when_breached_below() -> None:
    target = pd.Series({"VTI": 0.50, "BND": 0.50}, dtype=float)
    previous = pd.Series({"VTI": 0.40, "BND": 0.60}, dtype=float)
    band = 0.05

    decision = apply_rebalance_mode(
        mode="tolerance_band",
        previous_weights=previous,
        target_weights=target,
        portfolio_value=100_000.0,
        contribution_amount=0.0,
        tolerance_bands=_bands(per_ticker=band),
    )

    assert decision.rebalanced is True
    assert decision.applied_weights["VTI"] == pytest.approx(target["VTI"] - band, abs=1e-9)
    assert decision.applied_weights["VTI"] != pytest.approx(target["VTI"], abs=1e-3)


def test_asset_class_band_can_trigger_rebalance() -> None:
    target = pd.Series(
        {"VTI": 0.40, "VEA": 0.20, "BND": 0.40},
        dtype=float,
    )
    previous = pd.Series(
        {"VTI": 0.41, "VEA": 0.30, "BND": 0.29},
        dtype=float,
    )
    asset_classes = pd.Series(
        {"VTI": "equity", "VEA": "equity", "BND": "fixed_income"},
        dtype=str,
    )

    decision = apply_rebalance_mode(
        mode="tolerance_band",
        previous_weights=previous,
        target_weights=target,
        portfolio_value=100_000.0,
        contribution_amount=0.0,
        tolerance_bands=_bands(per_ticker=0.20, per_class=0.05),
        asset_classes=asset_classes,
    )

    assert decision.rebalanced is True


def test_full_rebalance_pulls_to_target_exactly() -> None:
    target = pd.Series({"VTI": 0.5, "BND": 0.5}, dtype=float)
    previous = pd.Series({"VTI": 0.7, "BND": 0.3}, dtype=float)

    decision = apply_rebalance_mode(
        mode="full_rebalance",
        previous_weights=previous,
        target_weights=target,
        portfolio_value=100_000.0,
        contribution_amount=0.0,
    )

    np.testing.assert_allclose(
        decision.applied_weights.to_numpy(),
        target.to_numpy(),
        atol=1e-12,
    )
