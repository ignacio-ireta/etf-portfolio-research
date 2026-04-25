from __future__ import annotations

import math

import numpy as np
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
    compare_against_benchmarks,
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


def test_calculate_sharpe_ratio_returns_zero_for_zero_volatility() -> None:
    assert calculate_sharpe_ratio(0.10, 0.0, 0.02) == 0.0


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


def test_zero_denominator_ratios_are_finite_for_flat_returns() -> None:
    returns = pd.Series(
        [0.0, 0.0, 0.0, 0.0],
        index=pd.date_range("2024-01-31", periods=4, freq="ME"),
    )

    ratios = [
        sharpe_ratio(returns, periods_per_year=12),
        sortino_ratio(returns, periods_per_year=12),
        calmar_ratio(returns, periods_per_year=12),
    ]

    assert ratios == [0.0, 0.0, 0.0]
    assert all(math.isfinite(ratio) for ratio in ratios)


def test_one_observation_metrics_use_documented_zero_for_undefined_cases() -> None:
    returns = pd.Series([0.01], index=pd.date_range("2024-01-31", periods=1, freq="ME"))
    benchmark = pd.Series([0.01], index=returns.index)

    metrics = [
        portfolio_volatility(returns, periods_per_year=12),
        sharpe_ratio(returns, periods_per_year=12),
        sortino_ratio(returns, periods_per_year=12),
        calmar_ratio(returns, periods_per_year=12),
        beta(returns, benchmark),
        tracking_error(returns, benchmark, periods_per_year=12),
        information_ratio(returns, benchmark, periods_per_year=12),
    ]

    assert metrics == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert all(math.isfinite(metric) for metric in metrics)


def test_flat_benchmark_relative_metrics_are_finite() -> None:
    returns = pd.Series(
        [0.01, 0.01, 0.01],
        index=pd.date_range("2024-01-31", periods=3, freq="ME"),
    )
    benchmark = pd.Series([0.0, 0.0, 0.0], index=returns.index)

    assert beta(returns, benchmark) == 0.0
    assert tracking_error(returns, returns, periods_per_year=12) == 0.0
    assert information_ratio(returns, returns, periods_per_year=12) == 0.0


def test_metrics_reject_nonfinite_inputs() -> None:
    returns = pd.Series(
        [0.01, np.nan, 0.02],
        index=pd.date_range("2024-01-31", periods=3, freq="ME"),
    )

    with pytest.raises(ValueError, match="finite"):
        sharpe_ratio(returns, periods_per_year=12)

    with pytest.raises(ValueError, match="risk_free_rate must be finite"):
        calculate_sharpe_ratio(0.10, 0.20, float("inf"))


def test_calmar_ratio_returns_zero_for_no_drawdown_series() -> None:
    returns = pd.Series(
        [0.01, 0.01, 0.01, 0.01],
        index=pd.date_range("2024-01-31", periods=4, freq="ME"),
    )

    result = calmar_ratio(returns, periods_per_year=12)

    assert result == 0.0
    assert math.isfinite(result)


def test_weight_diagnostics_return_expected_values() -> None:
    weights = make_weight_history()

    assert turnover(weights) == pytest.approx(0.15)
    assert average_number_of_holdings(weights) == pytest.approx(2.0)
    assert largest_position(weights) == pytest.approx(0.60)
    assert herfindahl_concentration_index(weights) == pytest.approx(0.5083333333333334)


def test_turnover_excludes_initial_allocation_and_uses_gross_weight_change() -> None:
    weights = pd.DataFrame(
        [
            {"AAA": 1.0, "BBB": 0.0},
            {"AAA": 0.0, "BBB": 1.0},
        ],
        index=pd.date_range("2024-01-31", periods=2, freq="ME"),
    )

    assert turnover(weights) == pytest.approx(2.0)
    assert turnover(weights.iloc[:1]) == 0.0


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

    assert summary["Turnover"] == pytest.approx(0.15)
    assert summary["Average Number of Holdings"] == pytest.approx(2.0)
    assert summary["Largest Position"] == pytest.approx(0.60)
    assert summary["Alpha"] == pytest.approx(0.016974757266420795)
    assert summary["Tracking Error"] == pytest.approx(0.004774934554525328)


def test_compare_against_benchmarks_adds_relative_metrics_to_optimized_row() -> None:
    table = compare_against_benchmarks(
        make_portfolio_returns(),
        weights=make_weight_history(),
        benchmark_returns={"Selected Benchmark ETF": make_benchmark_returns()},
        primary_benchmark_returns=make_benchmark_returns(),
        periods_per_year=12,
        risk_free_rate=0.02,
    )

    optimized = table.loc["Optimized Strategy"]
    assert optimized["Beta"] == pytest.approx(1.0497688332880066)
    assert optimized["Alpha"] == pytest.approx(0.016974757266420795)
    assert optimized["Tracking Error"] == pytest.approx(0.004774934554525328)
    assert optimized["Information Ratio"] == pytest.approx(3.76968517462526)


def test_compare_against_benchmarks_suppresses_weight_metrics_without_weights() -> None:
    table = compare_against_benchmarks(
        make_portfolio_returns(),
        weights=make_weight_history(),
        benchmark_returns={"Synthetic Benchmark": make_benchmark_returns()},
        primary_benchmark_returns=make_benchmark_returns(),
        periods_per_year=12,
    )

    synthetic = table.loc["Synthetic Benchmark"]
    assert pd.isna(synthetic["Turnover"])
    assert pd.isna(synthetic["Average Number of Holdings"])
    assert pd.isna(synthetic["Largest Position"])
    assert pd.isna(synthetic["Herfindahl Concentration Index"])


def test_compare_against_benchmarks_keeps_weight_metrics_with_supplied_weights() -> None:
    table = compare_against_benchmarks(
        make_portfolio_returns(),
        weights=make_weight_history(),
        benchmark_returns={"Weighted Benchmark": make_benchmark_returns()},
        benchmark_weights={"Weighted Benchmark": make_weight_history()},
        primary_benchmark_returns=make_benchmark_returns(),
        periods_per_year=12,
    )

    weighted = table.loc["Weighted Benchmark"]
    assert weighted["Turnover"] == pytest.approx(0.15)
    assert weighted["Average Number of Holdings"] == pytest.approx(2.0)
    assert weighted["Largest Position"] == pytest.approx(0.60)
    assert weighted["Herfindahl Concentration Index"] == pytest.approx(0.5083333333333334)
