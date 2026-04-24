from __future__ import annotations

import pandas as pd
import pandas.testing as pdt
import pytest

from etf_portfolio.backtesting.metrics import (
    alpha,
    average_number_of_holdings,
    best_month,
    beta,
    cagr,
    calculate_beta,
    calculate_portfolio_return,
    calculate_portfolio_volatility,
    calculate_sharpe_ratio,
    calmar_ratio,
    herfindahl_concentration_index,
    information_ratio,
    largest_position,
    portfolio_return,
    portfolio_volatility,
    sharpe_ratio,
    sortino_ratio,
    summarize_backtest_metrics,
    tracking_error,
    turnover,
    worst_month,
    worst_quarter,
)


def make_asset_returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AAA": [0.01, 0.02, -0.01],
            "BBB": [0.005, 0.0, 0.015],
        },
        index=pd.date_range("2024-01-31", periods=3, freq="ME"),
    )


def make_weights() -> pd.Series:
    return pd.Series({"AAA": 0.6, "BBB": 0.4})


def make_portfolio_returns() -> pd.Series:
    return pd.Series(
        [0.01, 0.02, -0.01, 0.015, -0.005, 0.01],
        index=pd.date_range("2024-01-31", periods=6, freq="ME"),
        name="portfolio",
    )


def make_benchmark_returns() -> pd.Series:
    return pd.Series(
        [0.008, 0.018, -0.012, 0.012, -0.004, 0.009],
        index=pd.date_range("2024-01-31", periods=6, freq="ME"),
        name="benchmark",
    )


def make_weight_history() -> pd.DataFrame:
    index = pd.to_datetime(["2024-03-31", "2024-04-30", "2024-05-31"])
    return pd.DataFrame(
        [
            {"AAA": 0.60, "BBB": 0.40},
            {"AAA": 0.50, "BBB": 0.50},
            {"AAA": 0.55, "BBB": 0.45},
        ],
        index=index,
    )


def test_portfolio_return_matches_weighted_sum() -> None:
    portfolio_returns = portfolio_return(make_asset_returns(), make_weights())

    expected = pd.Series(
        [0.008, 0.012, 0.0],
        index=make_asset_returns().index,
    )
    pdt.assert_series_equal(portfolio_returns, expected)


def test_calculate_portfolio_return_matches_dot_product() -> None:
    result = calculate_portfolio_return(
        make_weights(),
        pd.Series({"AAA": 0.12, "BBB": 0.06}),
    )
    assert result == pytest.approx(0.096)


def test_portfolio_return_rejects_missing_weights() -> None:
    weights = pd.Series({"AAA": 1.0})

    with pytest.raises(ValueError, match="Weights are missing entries"):
        portfolio_return(make_asset_returns(), weights)


def test_portfolio_volatility_returns_expected_value() -> None:
    result = portfolio_volatility(make_portfolio_returns(), periods_per_year=12)
    assert result == pytest.approx(0.04049691346263317)


def test_calculate_portfolio_volatility_returns_expected_value() -> None:
    covariance = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=["AAA", "BBB"],
        columns=["AAA", "BBB"],
    )
    result = calculate_portfolio_volatility(make_weights(), covariance)
    assert result == pytest.approx(0.1833030277982336)


def test_sharpe_ratio_returns_expected_value() -> None:
    result = sharpe_ratio(
        make_portfolio_returns(),
        risk_free_rate=0.02,
        periods_per_year=12,
    )
    assert result == pytest.approx(1.481594394974384)


def test_calculate_sharpe_ratio_returns_expected_value() -> None:
    assert calculate_sharpe_ratio(0.10, 0.20, 0.02) == pytest.approx(0.4)


def test_sortino_ratio_returns_expected_value() -> None:
    result = sortino_ratio(
        make_portfolio_returns(),
        risk_free_rate=0.02,
        periods_per_year=12,
    )
    assert result == pytest.approx(3.157408869505304)


def test_beta_returns_expected_value() -> None:
    result = beta(make_portfolio_returns(), make_benchmark_returns())
    assert result == pytest.approx(1.0497688332880066)


def test_calculate_beta_alias_returns_expected_value() -> None:
    result = calculate_beta(make_portfolio_returns(), make_benchmark_returns())
    assert result == pytest.approx(1.0497688332880066)


def test_tracking_error_returns_expected_value() -> None:
    result = tracking_error(
        make_portfolio_returns(),
        make_benchmark_returns(),
        periods_per_year=12,
    )
    assert result == pytest.approx(0.004774934554525328)


def test_information_ratio_returns_expected_value() -> None:
    result = information_ratio(
        make_portfolio_returns(),
        make_benchmark_returns(),
        periods_per_year=12,
    )
    assert result == pytest.approx(3.76968517462526)


def test_alpha_returns_expected_value() -> None:
    result = alpha(
        make_portfolio_returns(),
        make_benchmark_returns(),
        periods_per_year=12,
        risk_free_rate=0.02,
    )
    assert result == pytest.approx(0.016974757266420795)


def test_cagr_returns_expected_value() -> None:
    result = cagr(make_portfolio_returns(), periods_per_year=12)
    assert result == pytest.approx(0.08226714329881624)


def test_calmar_ratio_returns_expected_value() -> None:
    result = calmar_ratio(make_portfolio_returns(), periods_per_year=12)
    assert result == pytest.approx(8.226714329881617)


def test_weight_diagnostics_return_expected_values() -> None:
    weights = make_weight_history()

    assert turnover(weights) == pytest.approx(0.10)
    assert average_number_of_holdings(weights) == pytest.approx(2.0)
    assert largest_position(weights) == pytest.approx(0.60)
    assert herfindahl_concentration_index(weights) == pytest.approx(0.5083333333333334)


def test_period_extreme_metrics_return_expected_values() -> None:
    returns = make_portfolio_returns()

    assert worst_month(returns) == pytest.approx(-0.01)
    assert worst_quarter(returns) == pytest.approx(0.01989799999999997)
    assert best_month(returns) == pytest.approx(0.02)


def test_summarize_backtest_metrics_includes_benchmark_relative_and_holdings_stats() -> None:
    summary = summarize_backtest_metrics(
        make_portfolio_returns(),
        weights=make_weight_history(),
        periods_per_year=12,
        benchmark_returns=make_benchmark_returns(),
        risk_free_rate=0.02,
    )

    assert summary["Turnover"] == pytest.approx(0.10)
    assert summary["Average Number of Holdings"] == pytest.approx(2.0)
    assert summary["Largest Position"] == pytest.approx(0.60)
    assert summary["Alpha"] == pytest.approx(0.016974757266420795)
    assert summary["Tracking Error"] == pytest.approx(0.004774934554525328)
