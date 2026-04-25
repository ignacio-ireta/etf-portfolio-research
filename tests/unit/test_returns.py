from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf_portfolio.features.returns import (
    annualize_return,
    annualize_volatility,
    annualized_return,
    annualized_volatility,
    calculate_log_returns,
    calculate_simple_returns,
    cumulative_returns,
    drawdown_series,
    log_returns,
    max_drawdown,
    simple_returns,
)


def make_price_series() -> pd.Series:
    return pd.Series(
        [100.0, 110.0, 110.0, 99.0],
        index=pd.date_range("2024-01-31", periods=4, freq="ME"),
        name="price",
    )


def test_simple_returns() -> None:
    returns = simple_returns(make_price_series())
    expected = pd.Series(
        [0.1, 0.0, -0.1],
        index=returns.index,
        name="price",
    )

    pd.testing.assert_series_equal(returns, expected)


def test_calculate_simple_returns_alias() -> None:
    pd.testing.assert_series_equal(
        calculate_simple_returns(make_price_series()),
        simple_returns(make_price_series()),
    )


def test_simple_returns_preserve_missing_prices_by_default() -> None:
    prices = pd.Series(
        [100.0, np.nan, 110.0],
        index=pd.date_range("2024-01-31", periods=3, freq="ME"),
        name="price",
    )

    returns = simple_returns(prices)

    assert returns.empty


def test_simple_returns_forward_fill_requires_explicit_missing_policy() -> None:
    prices = pd.Series(
        [100.0, np.nan, 110.0],
        index=pd.date_range("2024-01-31", periods=3, freq="ME"),
        name="price",
    )

    returns = simple_returns(prices, missing="forward_fill")
    expected = pd.Series([0.0, 0.1], index=prices.index[1:], name="price")

    pd.testing.assert_series_equal(returns, expected)


def test_simple_returns_reject_non_positive_observed_prices() -> None:
    prices = pd.Series(
        [100.0, 0.0, 110.0],
        index=pd.date_range("2024-01-31", periods=3, freq="ME"),
        name="price",
    )

    with pytest.raises(ValueError, match="strictly positive"):
        simple_returns(prices)


def test_log_returns() -> None:
    returns = log_returns(make_price_series())
    expected = pd.Series(
        [np.log(1.1), 0.0, np.log(0.9)],
        index=returns.index,
        name="price",
    )

    pd.testing.assert_series_equal(returns, expected)


def test_calculate_log_returns_alias() -> None:
    pd.testing.assert_series_equal(
        calculate_log_returns(make_price_series()),
        log_returns(make_price_series()),
    )


def test_annualized_return() -> None:
    periodic_returns = pd.Series([0.1, 0.0, -0.1])

    result = annualized_return(periodic_returns, periods_per_year=12)
    expected = (1.1 * 1.0 * 0.9) ** (12 / 3) - 1.0

    assert result == pytest.approx(expected)


def test_annualize_return_alias() -> None:
    periodic_returns = pd.Series([0.1, 0.0, -0.1])
    assert annualize_return(periodic_returns, periods_per_year=12) == pytest.approx(
        annualized_return(periodic_returns, periods_per_year=12)
    )


def test_annualized_volatility() -> None:
    periodic_returns = pd.Series([0.1, 0.0, -0.1])

    result = annualized_volatility(periodic_returns, periods_per_year=12)
    expected = 0.1 * np.sqrt(12)

    assert result == pytest.approx(expected)


def test_annualize_volatility_alias() -> None:
    periodic_returns = pd.Series([0.1, 0.0, -0.1])
    assert annualize_volatility(periodic_returns, periods_per_year=12) == pytest.approx(
        annualized_volatility(periodic_returns, periods_per_year=12)
    )


def test_cumulative_returns() -> None:
    periodic_returns = pd.Series([0.1, 0.0, -0.1])

    result = cumulative_returns(periodic_returns)
    expected = pd.Series([0.1, 0.1, -0.01])

    pd.testing.assert_series_equal(result, expected)


def test_drawdown_series() -> None:
    periodic_returns = pd.Series([0.1, 0.0, -0.1])

    result = drawdown_series(periodic_returns)
    expected = pd.Series([0.0, 0.0, -0.1])

    pd.testing.assert_series_equal(result, expected)


def test_max_drawdown() -> None:
    periodic_returns = pd.Series([0.1, 0.0, -0.1])

    result = max_drawdown(periodic_returns)

    assert result == pytest.approx(-0.1)


def test_annualized_metrics_reject_empty_input() -> None:
    empty_returns = pd.Series(dtype=float)

    with pytest.raises(ValueError):
        annualized_return(empty_returns, periods_per_year=12)

    with pytest.raises(ValueError):
        annualized_volatility(empty_returns, periods_per_year=12)
