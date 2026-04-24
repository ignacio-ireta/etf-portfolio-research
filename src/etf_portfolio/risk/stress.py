"""Stress-period helpers for robustness analysis."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from etf_portfolio.features.returns import drawdown_series

DEFAULT_STRESS_PERIODS: dict[str, tuple[str, str]] = {
    "COVID Crash (2020-02 to 2020-04)": ("2020-02-01", "2020-04-30"),
    "Inflation / Rate Shock (2022)": ("2022-01-01", "2022-12-31"),
}


def stress_period_returns(
    portfolio_returns: pd.Series,
    *,
    benchmark_returns: pd.Series | pd.DataFrame | None = None,
    periods: Mapping[str, tuple[str, str] | tuple[pd.Timestamp, pd.Timestamp]] | None = None,
) -> pd.DataFrame:
    """Compute compounded returns over named stress periods when data is available."""

    if portfolio_returns.empty:
        raise ValueError("portfolio_returns must not be empty.")

    period_map: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
        name: (pd.Timestamp(start), pd.Timestamp(end))
        for name, (start, end) in (periods or DEFAULT_STRESS_PERIODS).items()
    }
    recent_period = infer_recent_drawdown_period(portfolio_returns)
    if recent_period is not None:
        period_map[recent_period[0]] = recent_period[1]

    series_map = {"Portfolio": portfolio_returns.astype(float)}
    if benchmark_returns is not None:
        benchmark_frame = (
            benchmark_returns.to_frame()
            if isinstance(benchmark_returns, pd.Series)
            else benchmark_returns.copy()
        )
        for column in benchmark_frame.columns:
            series_map[str(column)] = benchmark_frame[column].astype(float)

    rows: list[dict[str, float | str]] = []
    for period_name, (start_date, end_date) in period_map.items():
        row: dict[str, float | str] = {"Period": period_name}
        has_data = False
        for series_name, returns in series_map.items():
            window = returns.loc[(returns.index >= start_date) & (returns.index <= end_date)]
            if window.empty:
                row[series_name] = float("nan")
                continue
            row[series_name] = float((1.0 + window).prod() - 1.0)
            has_data = True
        if has_data:
            rows.append(row)

    return pd.DataFrame(rows)


def infer_recent_drawdown_period(
    returns: pd.Series,
    *,
    lookback_periods: int = 252,
) -> tuple[str, tuple[pd.Timestamp, pd.Timestamp]] | None:
    """Infer a recent peak-to-trough drawdown window from the latest history."""

    if returns.empty:
        raise ValueError("returns must not be empty.")

    recent_returns = returns.tail(min(len(returns), lookback_periods))
    if len(recent_returns) < 20:
        return None

    recent_drawdowns = drawdown_series(recent_returns)
    trough_date = recent_drawdowns.idxmin()
    trough_drawdown = float(recent_drawdowns.loc[trough_date])
    if trough_drawdown >= 0.0:
        return None

    wealth = (1.0 + recent_returns).cumprod()
    peak_date = wealth.loc[:trough_date].idxmax()
    if peak_date >= trough_date:
        return None

    label = f"Recent Drawdown ({peak_date:%Y-%m} to {trough_date:%Y-%m})"
    return label, (peak_date, trough_date)


__all__ = [
    "DEFAULT_STRESS_PERIODS",
    "infer_recent_drawdown_period",
    "stress_period_returns",
]
