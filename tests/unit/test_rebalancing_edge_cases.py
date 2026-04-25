import pandas as pd
import pytest

from etf_portfolio.backtesting.contributions import allocate_contribution
from etf_portfolio.backtesting.rebalancing import apply_rebalance_mode
from etf_portfolio.config import RebalanceToleranceBandsConfig


def test_contribution_only_at_target_fill():
    """Verify that if total gap matches contribution, it fills gaps exactly."""
    # Assets A and B. Target 50/50.
    # Current value 100. Holdings: A=70, B=30.
    # Contribution 100. New total = 200.
    # Target holdings: A=100, B=100.
    # Gaps: A=30, B=70. Total gap = 100.
    # Contribution 100.
    # Trades: A=30, B=70.

    previous_weights = pd.Series({"A": 0.7, "B": 0.3})
    target_weights = pd.Series({"A": 0.5, "B": 0.5})
    portfolio_value = 100.0
    contribution_amount = 100.0

    trades = allocate_contribution(
        previous_weights=previous_weights,
        target_weights=target_weights,
        cash_contribution=contribution_amount,
        portfolio_value=portfolio_value,
    )

    assert trades["A"] == pytest.approx(30.0)
    assert trades["B"] == pytest.approx(70.0)
    assert trades.sum() == pytest.approx(contribution_amount)


def test_contribution_only_proportional_fill():
    """Verify behavior when contribution is less than total gap."""
    previous_weights = pd.Series({"A": 0.8, "B": 0.2})
    target_weights = pd.Series({"A": 0.5, "B": 0.5})
    # Portfolio 200. A=160, B=40.
    # Contribution 100. New total 300.
    # Target: A=150, B=150.
    # Gaps: A=0, B=110. Total gap = 110.
    # Contribution 100 < 110.
    # Trades: A = 0 * (100/110) = 0.
    # B = 110 * (100/110) = 100.

    portfolio_value = 200.0
    contribution_amount = 100.0

    trades = allocate_contribution(
        previous_weights=previous_weights,
        target_weights=target_weights,
        cash_contribution=contribution_amount,
        portfolio_value=portfolio_value,
    )

    assert trades["A"] == pytest.approx(0.0)
    assert trades["B"] == pytest.approx(100.0)


def test_tolerance_band_with_contribution_no_pre_breach():
    """Verify that if no breach exists, tolerance-band uses contribution_only logic."""
    # Target 50/50. Band 0.1.
    # Current 55/45. No breach (0.05 < 0.1).
    # Contribution 10. Value 100.
    # New total 110. Target A=55, B=55.
    # Current A=55, B=45.
    # Gaps: A=0, B=10. Total gap = 10.
    # Trades: A=0, B=10.
    # Applied weights: A=55/110=0.5, B=55/110=0.5.

    previous_weights = pd.Series({"A": 0.55, "B": 0.45})
    target_weights = pd.Series({"A": 0.5, "B": 0.5})
    bands = RebalanceToleranceBandsConfig(per_ticker_abs_drift=0.1, per_asset_class_abs_drift=0.2)

    decision = apply_rebalance_mode(
        mode="tolerance_band",
        previous_weights=previous_weights,
        target_weights=target_weights,
        portfolio_value=100.0,
        contribution_amount=10.0,
        tolerance_bands=bands,
    )

    assert decision.mode == "contribution_only"
    assert decision.applied_weights["A"] == pytest.approx(0.5)
    assert decision.applied_weights["B"] == pytest.approx(0.5)
    assert decision.trades_dollars["A"] == pytest.approx(0.0)
    assert decision.trades_dollars["B"] == pytest.approx(10.0)
    assert decision.rebalanced is True


def test_tolerance_band_with_breach_and_contribution():
    """Verify that if a breach exists, it rebalances to bands and uses contribution."""
    previous_weights = pd.Series({"A": 0.6, "B": 0.4})
    target_weights = pd.Series({"A": 0.5, "B": 0.5})
    bands = RebalanceToleranceBandsConfig(per_ticker_abs_drift=0.05, per_asset_class_abs_drift=0.2)

    decision = apply_rebalance_mode(
        mode="tolerance_band",
        previous_weights=previous_weights,
        target_weights=target_weights,
        portfolio_value=100.0,
        contribution_amount=10.0,
        tolerance_bands=bands,
    )

    assert decision.applied_weights["A"] == pytest.approx(0.55, abs=1e-6)
    assert decision.applied_weights["B"] == pytest.approx(0.45, abs=1e-6)
    assert decision.trades_dollars["A"] == pytest.approx(0.5, abs=1e-6)
    assert decision.trades_dollars["B"] == pytest.approx(9.5, abs=1e-6)


def test_tolerance_band_asset_class_breach():
    """Verify asset class drift triggers rebalance."""
    previous_weights = pd.Series({"A": 0.4, "B": 0.4, "C": 0.2})
    target_weights = pd.Series({"A": 0.3, "B": 0.3, "C": 0.4})
    asset_classes = pd.Series({"A": "Equity", "B": "Equity", "C": "Fixed Income"})

    bands = RebalanceToleranceBandsConfig(per_ticker_abs_drift=0.5, per_asset_class_abs_drift=0.1)

    decision = apply_rebalance_mode(
        mode="tolerance_band",
        previous_weights=previous_weights,
        target_weights=target_weights,
        portfolio_value=100.0,
        contribution_amount=0.0,
        tolerance_bands=bands,
        asset_classes=asset_classes,
    )

    assert decision.rebalanced is True
    equity_weight = decision.applied_weights["A"] + decision.applied_weights["B"]
    assert equity_weight == pytest.approx(0.7, abs=1e-6)


def test_large_contribution_near_target():
    """Verify large contribution brings portfolio very close to target."""
    previous_weights = pd.Series({"E": 0.9, "F": 0.1})
    target_weights = pd.Series({"E": 0.6, "F": 0.4})

    trades = allocate_contribution(
        previous_weights=previous_weights,
        target_weights=target_weights,
        cash_contribution=1000.0,
        portfolio_value=100.0,
    )

    new_holdings = previous_weights * 100.0 + trades
    new_weights = new_holdings / 1100.0
    assert new_weights["E"] == pytest.approx(0.6)
    assert new_weights["F"] == pytest.approx(0.4)
