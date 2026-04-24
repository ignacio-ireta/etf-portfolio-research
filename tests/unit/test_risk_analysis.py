from __future__ import annotations

import pandas as pd
import pytest

from etf_portfolio.risk.drawdown import (
    rolling_correlation,
    rolling_sharpe,
    rolling_volatility,
)
from etf_portfolio.risk.exposure import (
    aggregate_group_exposure,
    latest_portfolio_weights,
    weighted_expense_ratio,
    weighted_expense_ratio_history,
)
from etf_portfolio.risk.stress import infer_recent_drawdown_period, stress_period_returns


def test_exposure_helpers_aggregate_latest_weights() -> None:
    weights = pd.DataFrame(
        [
            {"VTI": 0.6, "BND": 0.3, "IAU": 0.1},
            {"VTI": 0.5, "BND": 0.35, "IAU": 0.15},
        ],
        index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
    )
    classifications = pd.Series({"VTI": "equity", "BND": "fixed_income", "IAU": "alternatives"})
    expense_ratios = pd.Series({"VTI": 0.0003, "BND": 0.0004, "IAU": 0.0025})

    latest = latest_portfolio_weights(weights)
    exposure = aggregate_group_exposure(latest, classifications)

    assert latest.index.tolist() == ["VTI", "BND", "IAU"]
    assert exposure.to_dict() == {
        "equity": 0.5,
        "fixed_income": 0.35,
        "alternatives": 0.15,
    }
    assert weighted_expense_ratio(latest, expense_ratios) == 0.000665


def test_weighted_expense_ratio_history_matches_per_date() -> None:
    weights = pd.DataFrame(
        [
            {"VTI": 0.6, "BND": 0.3, "IAU": 0.1},
            {"VTI": 0.5, "BND": 0.35, "IAU": 0.15},
        ],
        index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
    )
    expense_ratios = pd.Series({"VTI": 0.0003, "BND": 0.0004, "IAU": 0.0025})

    history = weighted_expense_ratio_history(weights, expense_ratios)

    assert history.index.tolist() == weights.index.tolist()
    assert history.iloc[0] == pytest.approx(weighted_expense_ratio(weights.iloc[0], expense_ratios))
    assert history.iloc[1] == pytest.approx(weighted_expense_ratio(weights.iloc[1], expense_ratios))


def test_rolling_risk_helpers_return_series() -> None:
    dates = pd.bdate_range("2024-01-01", periods=10)
    returns = pd.Series(
        [0.01, -0.005, 0.004, 0.006, -0.003, 0.002, 0.005, -0.004, 0.003, 0.004],
        index=dates,
    )
    benchmark = pd.Series(
        [0.008, -0.004, 0.003, 0.005, -0.002, 0.001, 0.004, -0.003, 0.002, 0.003],
        index=dates,
    )

    vol = rolling_volatility(returns, window=5, periods_per_year=252)
    sharpe = rolling_sharpe(returns, window=5, periods_per_year=252)
    correlation = rolling_correlation(returns, benchmark, window=5)

    assert vol.notna().sum() == 6
    assert sharpe.notna().sum() == 6
    assert correlation.notna().sum() == 6


def test_stress_period_analysis_uses_available_windows() -> None:
    dates = pd.bdate_range("2020-01-01", "2022-12-30")
    portfolio_returns = pd.Series(0.001, index=dates)
    portfolio_returns.loc["2020-03-02":"2020-03-31"] = -0.02
    portfolio_returns.loc["2022-10-03":"2022-11-15"] = -0.01
    benchmark_returns = pd.Series(0.0008, index=dates, name="Benchmark")

    table = stress_period_returns(
        portfolio_returns,
        benchmark_returns=benchmark_returns,
    )
    recent_period = infer_recent_drawdown_period(portfolio_returns)

    assert "COVID Crash (2020-02 to 2020-04)" in table["Period"].tolist()
    assert "Inflation / Rate Shock (2022)" in table["Period"].tolist()
    assert "Portfolio" in table.columns
    assert "Benchmark" in table.columns
    assert recent_period is not None
