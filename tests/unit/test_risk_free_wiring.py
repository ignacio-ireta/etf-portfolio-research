"""Verify the configured risk-free rate flows through metrics and optimizer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf_portfolio.backtesting.metrics import (
    sharpe_ratio,
    sortino_ratio,
    summarize_backtest_metrics,
)
from etf_portfolio.config import RiskFreeConfig
from etf_portfolio.features.risk_free import get_risk_free_rate
from etf_portfolio.optimization.optimizer import optimize_portfolio


def _make_returns(seed: int = 7, periods: int = 504) -> pd.Series:
    rng = np.random.default_rng(seed)
    daily = rng.normal(loc=0.0006, scale=0.012, size=periods)
    index = pd.bdate_range("2020-01-02", periods=periods)
    return pd.Series(daily, index=index, name="portfolio")


def test_get_risk_free_rate_returns_configured_constant() -> None:
    config = RiskFreeConfig(source="constant", value=0.03)
    assert get_risk_free_rate(config) == pytest.approx(0.03)


def test_sharpe_with_risk_free_rate_differs_from_zero_baseline() -> None:
    returns = _make_returns()

    sharpe_zero = sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252)
    sharpe_rf = sharpe_ratio(returns, risk_free_rate=0.03, periods_per_year=252)

    assert sharpe_zero != pytest.approx(sharpe_rf, abs=1e-6), (
        "Sharpe ratio must change when the risk-free rate is non-zero."
    )
    expected_delta = -0.03 / (returns.std(ddof=1) * np.sqrt(252))
    assert (sharpe_rf - sharpe_zero) == pytest.approx(expected_delta, rel=1e-6)


def test_sortino_with_risk_free_rate_differs_from_zero_baseline() -> None:
    returns = _make_returns(seed=11)

    sortino_zero = sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252)
    sortino_rf = sortino_ratio(returns, risk_free_rate=0.03, periods_per_year=252)

    assert sortino_zero != pytest.approx(sortino_rf, abs=1e-6)


def test_summarize_backtest_metrics_uses_configured_risk_free_rate() -> None:
    returns = _make_returns(seed=23)
    weights = pd.DataFrame(
        {"asset": [1.0]},
        index=pd.DatetimeIndex([returns.index.min()], name="rebalance_date"),
    )

    summary_zero = summarize_backtest_metrics(
        returns,
        weights=weights,
        periods_per_year=252,
        risk_free_rate=0.0,
    )
    summary_rf = summarize_backtest_metrics(
        returns,
        weights=weights,
        periods_per_year=252,
        risk_free_rate=0.03,
    )

    assert summary_zero["Sharpe Ratio"] != pytest.approx(summary_rf["Sharpe Ratio"], abs=1e-6)
    assert summary_zero["Sortino Ratio"] != pytest.approx(summary_rf["Sortino Ratio"], abs=1e-6)
    assert summary_zero["CAGR"] == pytest.approx(summary_rf["CAGR"])
    assert summary_zero["Max Drawdown"] == pytest.approx(summary_rf["Max Drawdown"])


def test_max_sharpe_optimizer_responds_to_risk_free_rate() -> None:
    rng = np.random.default_rng(42)
    asset_count = 4
    returns_matrix = rng.normal(loc=0.0008, scale=0.012, size=(252, asset_count))
    returns_matrix[:, 0] += 0.0004
    returns_matrix[:, 3] -= 0.0002
    columns = ["A", "B", "C", "D"]
    returns_frame = pd.DataFrame(returns_matrix, columns=columns)

    expected_returns = returns_frame.mean() * 252
    covariance = returns_frame.cov() * 252

    weights_zero_rf = optimize_portfolio(
        expected_returns,
        covariance,
        method="max_sharpe",
        max_weight=0.6,
        risk_free_rate=0.0,
    )
    weights_high_rf = optimize_portfolio(
        expected_returns,
        covariance,
        method="max_sharpe",
        max_weight=0.6,
        risk_free_rate=0.05,
    )

    assert not np.allclose(weights_zero_rf.to_numpy(), weights_high_rf.to_numpy(), atol=1e-6), (
        "max_sharpe weights should shift when the risk-free rate changes; "
        f"weights with rf=0: {weights_zero_rf.to_dict()}, "
        f"with rf=0.05: {weights_high_rf.to_dict()}"
    )
