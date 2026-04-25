"""Return and drawdown calculations."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

MissingPricePolicy = Literal["preserve", "forward_fill"]


def calculate_simple_returns(
    prices: pd.Series | pd.DataFrame,
    *,
    missing: MissingPricePolicy = "preserve",
) -> pd.Series | pd.DataFrame:
    """Compute simple percentage returns from price levels."""

    _validate_non_empty(prices)
    _validate_strictly_positive_observed_prices(prices)
    if missing not in ("preserve", "forward_fill"):
        raise ValueError("missing must be either 'preserve' or 'forward_fill'.")
    if missing == "forward_fill":
        prices = prices.ffill()
    return prices.pct_change(fill_method=None).dropna()


def calculate_log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compute log returns from price levels."""

    _validate_non_empty(prices)
    _validate_strictly_positive_observed_prices(prices)
    return np.log(prices / prices.shift(1)).dropna()


def annualize_return(
    daily_returns: pd.Series | pd.DataFrame,
    *,
    periods_per_year: int = 252,
) -> float | pd.Series:
    """Compute geometric annualized return from periodic returns."""

    _validate_non_empty(daily_returns)

    compounded_growth = (1.0 + daily_returns).prod()
    periods = len(daily_returns)
    return compounded_growth ** (periods_per_year / periods) - 1.0


def annualize_volatility(
    daily_returns: pd.Series | pd.DataFrame,
    *,
    periods_per_year: int = 252,
) -> float | pd.Series:
    """Compute annualized volatility using sample standard deviation."""

    _validate_non_empty(daily_returns)
    return daily_returns.std(ddof=1) * np.sqrt(periods_per_year)


def simple_returns(
    prices: pd.Series | pd.DataFrame,
    *,
    missing: MissingPricePolicy = "preserve",
) -> pd.Series | pd.DataFrame:
    """Backward-compatible alias for simple return calculation."""

    return calculate_simple_returns(prices, missing=missing)


def log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Backward-compatible alias for log return calculation."""

    return calculate_log_returns(prices)


def annualized_return(
    returns: pd.Series | pd.DataFrame,
    *,
    periods_per_year: int,
) -> float | pd.Series:
    """Backward-compatible alias for annualized return calculation."""

    return annualize_return(returns, periods_per_year=periods_per_year)


def annualized_volatility(
    returns: pd.Series | pd.DataFrame,
    *,
    periods_per_year: int,
) -> float | pd.Series:
    """Backward-compatible alias for annualized volatility calculation."""

    return annualize_volatility(returns, periods_per_year=periods_per_year)


def cumulative_returns(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compute cumulative returns from a periodic return series."""

    return (1.0 + returns).cumprod() - 1.0


def drawdown_series(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compute drawdown series from periodic returns."""

    wealth_index = (1.0 + returns).cumprod()
    running_peak = wealth_index.cummax()
    return wealth_index / running_peak - 1.0


def max_drawdown(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Compute maximum drawdown from periodic returns."""

    drawdowns = drawdown_series(returns)
    return drawdowns.min()


def _validate_non_empty(data: pd.Series | pd.DataFrame) -> None:
    if data.empty:
        raise ValueError("Input data must not be empty.")


def _validate_strictly_positive_observed_prices(prices: pd.Series | pd.DataFrame) -> None:
    invalid = prices.le(0).fillna(False)
    if bool(invalid.any().any() if isinstance(invalid, pd.DataFrame) else invalid.any()):
        raise ValueError("Observed prices must be strictly positive.")


__all__ = [
    "annualize_return",
    "annualize_volatility",
    "annualized_return",
    "annualized_volatility",
    "calculate_log_returns",
    "calculate_simple_returns",
    "cumulative_returns",
    "drawdown_series",
    "log_returns",
    "max_drawdown",
    "MissingPricePolicy",
    "simple_returns",
]
