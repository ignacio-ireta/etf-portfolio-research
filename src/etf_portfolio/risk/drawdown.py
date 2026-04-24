"""Rolling risk helpers for backtest analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_volatility(
    returns: pd.Series,
    *,
    window: int = 63,
    periods_per_year: int = 252,
) -> pd.Series:
    """Compute rolling annualized volatility."""

    if returns.empty:
        raise ValueError("returns must not be empty.")
    if window <= 1:
        raise ValueError("window must be greater than 1.")

    return returns.rolling(window).std(ddof=1) * np.sqrt(periods_per_year)


def rolling_sharpe(
    returns: pd.Series,
    *,
    window: int = 63,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """Compute a rolling annualized Sharpe ratio."""

    if returns.empty:
        raise ValueError("returns must not be empty.")
    if window <= 1:
        raise ValueError("window must be greater than 1.")

    per_period_rf = risk_free_rate / periods_per_year
    excess_returns = returns - per_period_rf
    rolling_mean = excess_returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std(ddof=1)
    return rolling_mean.div(rolling_std).mul(np.sqrt(periods_per_year))


def rolling_correlation(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    window: int = 63,
) -> pd.Series:
    """Compute rolling correlation to a benchmark."""

    if returns.empty or benchmark_returns.empty:
        raise ValueError("returns and benchmark_returns must not be empty.")
    if window <= 1:
        raise ValueError("window must be greater than 1.")

    aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join="inner")
    return aligned_returns.rolling(window).corr(aligned_benchmark)


__all__ = ["rolling_correlation", "rolling_sharpe", "rolling_volatility"]
