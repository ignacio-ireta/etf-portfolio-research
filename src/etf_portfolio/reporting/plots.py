"""Plotly chart builders for portfolio reports."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from etf_portfolio.backtesting.metrics import (
    cagr,
    calculate_portfolio_return,
    calculate_portfolio_volatility,
    portfolio_volatility,
    sharpe_ratio,
)
from etf_portfolio.features.returns import (
    cumulative_returns,
    drawdown_series,
    max_drawdown,
)
from etf_portfolio.optimization.frontier import build_efficient_frontier
from etf_portfolio.risk.drawdown import (
    rolling_correlation,
    rolling_sharpe,
    rolling_volatility,
)
from etf_portfolio.risk.exposure import (
    aggregate_group_exposure,
    latest_portfolio_weights,
    weighted_expense_ratio_history,
)
from etf_portfolio.risk.stress import stress_period_returns


def build_efficient_frontier_figure(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    *,
    portfolio_weights: pd.Series | None = None,
    max_weight: float = 1.0,
    risk_free_rate: float = 0.0,
    num_points: int = 25,
    asset_classes: pd.Series | None = None,
    asset_class_bounds: dict[str, tuple[float, float]] | None = None,
    ticker_bounds: dict[str, tuple[float | None, float | None]] | None = None,
    bond_assets: list[str] | None = None,
    min_bond_exposure: float | None = None,
    expense_ratios: pd.Series | None = None,
) -> go.Figure:
    """Build an efficient frontier chart and optionally mark the portfolio."""

    frontier_points = build_efficient_frontier(
        expected_returns,
        covariance_matrix,
        max_weight=max_weight,
        risk_free_rate=risk_free_rate,
        num_points=num_points,
        asset_classes=asset_classes,
        asset_class_bounds=asset_class_bounds,
        ticker_bounds=ticker_bounds,
        bond_assets=bond_assets,
        min_bond_exposure=min_bond_exposure,
        expense_ratios=expense_ratios,
    ).to_dict(orient="records")

    figure = go.Figure()
    if frontier_points:
        figure.add_trace(
            go.Scatter(
                x=[point["portfolio_volatility"] for point in frontier_points],
                y=[point["portfolio_return"] for point in frontier_points],
                mode="lines+markers",
                name="Efficient frontier",
            )
        )

    asset_volatility = np.sqrt(np.diag(covariance_matrix))
    figure.add_trace(
        go.Scatter(
            x=asset_volatility,
            y=expected_returns,
            mode="markers+text",
            name="Assets",
            text=list(expected_returns.index),
            textposition="top center",
        )
    )

    if portfolio_weights is not None:
        portfolio_return = calculate_portfolio_return(portfolio_weights, expected_returns)
        portfolio_vol = calculate_portfolio_volatility(portfolio_weights, covariance_matrix)
        figure.add_trace(
            go.Scatter(
                x=[portfolio_vol],
                y=[portfolio_return],
                mode="markers",
                marker={"size": 12, "symbol": "diamond"},
                name="Portfolio",
            )
        )

    figure.update_layout(
        title="Efficient Frontier",
        xaxis_title="Volatility",
        yaxis_title="Expected Return",
        template="plotly_white",
    )
    return figure


def build_cumulative_returns_figure(
    portfolio_returns: pd.Series,
    *,
    benchmark_returns: pd.Series | pd.DataFrame | None = None,
) -> go.Figure:
    """Build a cumulative returns chart."""

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=portfolio_returns.index,
            y=cumulative_returns(portfolio_returns),
            mode="lines",
            name="Portfolio",
        )
    )

    if benchmark_returns is not None:
        benchmark_frame = _coerce_benchmark_frame(benchmark_returns)
        aligned_portfolio, aligned_benchmarks = portfolio_returns.align(
            benchmark_frame,
            join="inner",
            axis=0,
        )
        for column in aligned_benchmarks.columns:
            figure.add_trace(
                go.Scatter(
                    x=aligned_benchmarks.index,
                    y=cumulative_returns(aligned_benchmarks[column]),
                    mode="lines",
                    name=column,
                )
            )
        figure.update_xaxes(range=[aligned_portfolio.index.min(), aligned_portfolio.index.max()])

    figure.update_layout(
        title="Cumulative Returns",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        template="plotly_white",
    )
    return figure


def build_drawdown_figure(
    portfolio_returns: pd.Series,
    *,
    benchmark_returns: pd.Series | pd.DataFrame | None = None,
) -> go.Figure:
    """Build a drawdown chart."""

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=portfolio_returns.index,
            y=drawdown_series(portfolio_returns),
            fill="tozeroy",
            mode="lines",
            name="Portfolio",
        )
    )

    if benchmark_returns is not None:
        benchmark_frame = _coerce_benchmark_frame(benchmark_returns)
        aligned_portfolio, aligned_benchmarks = portfolio_returns.align(
            benchmark_frame,
            join="inner",
            axis=0,
        )
        for column in aligned_benchmarks.columns:
            figure.add_trace(
                go.Scatter(
                    x=aligned_benchmarks.index,
                    y=drawdown_series(aligned_benchmarks[column]),
                    fill="tozeroy",
                    mode="lines",
                    name=column,
                )
            )
        figure.update_xaxes(range=[aligned_portfolio.index.min(), aligned_portfolio.index.max()])

    figure.update_layout(
        title="Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        template="plotly_white",
    )
    return figure


def build_weights_figure(weights: pd.DataFrame) -> go.Figure:
    """Build a stacked area chart of portfolio weights over rebalance dates."""

    figure = go.Figure()
    for position, column in enumerate(weights.columns):
        figure.add_trace(
            go.Scatter(
                x=weights.index,
                y=weights[column],
                mode="lines",
                stackgroup="weights",
                name=column,
                groupnorm="fraction" if position == 0 else None,
            )
        )

    figure.update_layout(
        title="Portfolio Weights",
        xaxis_title="Rebalance Date",
        yaxis_title="Weight",
        template="plotly_white",
    )
    return figure


def build_group_exposure_figure(
    weights: pd.DataFrame,
    groups: pd.Series,
    *,
    title: str,
) -> go.Figure:
    """Build a latest portfolio exposure bar chart for a metadata grouping."""

    latest = latest_portfolio_weights(weights)
    exposure = aggregate_group_exposure(latest, groups.reindex(latest.index))

    figure = go.Figure(
        data=[
            go.Bar(
                x=exposure.index.tolist(),
                y=exposure.values,
                name=title,
            )
        ]
    )
    figure.update_layout(
        title=title,
        xaxis_title="Group",
        yaxis_title="Weight",
        template="plotly_white",
    )
    return figure


def build_benchmark_comparison_figure(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | pd.DataFrame,
    *,
    periods_per_year: int,
    risk_free_rate: float = 0.0,
) -> go.Figure:
    """Build a portfolio versus benchmark comparison chart."""

    benchmark_frame = _coerce_benchmark_frame(benchmark_returns)
    aligned_portfolio, aligned_benchmarks = portfolio_returns.align(
        benchmark_frame,
        join="inner",
        axis=0,
    )
    metrics = {
        "Portfolio": [
            cagr(aligned_portfolio, periods_per_year=periods_per_year),
            portfolio_volatility(
                aligned_portfolio,
                periods_per_year=periods_per_year,
            ),
            sharpe_ratio(
                aligned_portfolio,
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year,
            ),
            float(max_drawdown(aligned_portfolio)),
        ]
    }
    for column in aligned_benchmarks.columns:
        benchmark_series = aligned_benchmarks[column]
        metrics[column] = [
            cagr(benchmark_series, periods_per_year=periods_per_year),
            portfolio_volatility(
                benchmark_series,
                periods_per_year=periods_per_year,
            ),
            sharpe_ratio(
                benchmark_series,
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year,
            ),
            float(max_drawdown(benchmark_series)),
        ]
    metrics = pd.DataFrame(metrics, index=["CAGR", "Volatility", "Sharpe", "Max Drawdown"])

    figure = go.Figure()
    for column in metrics.columns:
        figure.add_trace(
            go.Bar(
                x=metrics.index,
                y=metrics[column],
                name=column,
            )
        )

    figure.update_layout(
        title="Benchmark Comparison",
        xaxis_title="Metric",
        yaxis_title="Value",
        barmode="group",
        template="plotly_white",
    )
    return figure


def build_rolling_volatility_figure(
    portfolio_returns: pd.Series,
    *,
    window: int = 63,
    periods_per_year: int = 252,
) -> go.Figure:
    """Build a rolling annualized volatility chart."""

    rolling = rolling_volatility(
        portfolio_returns,
        window=window,
        periods_per_year=periods_per_year,
    )
    figure = go.Figure(
        data=[
            go.Scatter(
                x=rolling.index,
                y=rolling,
                mode="lines",
                name="Portfolio",
            )
        ]
    )
    figure.update_layout(
        title=f"Rolling Volatility ({window} periods)",
        xaxis_title="Date",
        yaxis_title="Annualized Volatility",
        template="plotly_white",
    )
    return figure


def build_rolling_sharpe_figure(
    portfolio_returns: pd.Series,
    *,
    window: int = 63,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> go.Figure:
    """Build a rolling Sharpe ratio chart."""

    rolling = rolling_sharpe(
        portfolio_returns,
        window=window,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )
    figure = go.Figure(
        data=[
            go.Scatter(
                x=rolling.index,
                y=rolling,
                mode="lines",
                name="Portfolio",
            )
        ]
    )
    figure.update_layout(
        title=f"Rolling Sharpe ({window} periods)",
        xaxis_title="Date",
        yaxis_title="Sharpe Ratio",
        template="plotly_white",
    )
    return figure


def build_rolling_correlation_figure(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    window: int = 63,
) -> go.Figure:
    """Build a rolling correlation-to-benchmark chart."""

    rolling = rolling_correlation(
        portfolio_returns,
        benchmark_returns,
        window=window,
    )
    figure = go.Figure(
        data=[
            go.Scatter(
                x=rolling.index,
                y=rolling,
                mode="lines",
                name="Portfolio vs Benchmark",
            )
        ]
    )
    figure.update_layout(
        title=f"Rolling Correlation To Benchmark ({window} periods)",
        xaxis_title="Date",
        yaxis_title="Correlation",
        template="plotly_white",
    )
    return figure


def build_group_exposure_pie_figure(
    weights: pd.DataFrame,
    groups: pd.Series,
    *,
    title: str,
) -> go.Figure:
    """Build a latest portfolio exposure pie chart for a metadata grouping."""

    latest = latest_portfolio_weights(weights)
    exposure = aggregate_group_exposure(latest, groups.reindex(latest.index))

    figure = go.Figure(
        data=[
            go.Pie(
                labels=exposure.index.tolist(),
                values=exposure.values,
                hole=0.45,
            )
        ]
    )
    figure.update_layout(title=title, template="plotly_white")
    return figure


def build_stress_period_figure(
    portfolio_returns: pd.Series,
    *,
    benchmark_returns: pd.Series | pd.DataFrame | None = None,
) -> go.Figure:
    """Build a stress-period bar chart from canonical stress windows."""

    table = stress_period_returns(
        portfolio_returns,
        benchmark_returns=benchmark_returns,
    )

    figure = go.Figure()
    if table.empty or "Period" not in table.columns:
        figure.update_layout(
            title="Stress Periods (no overlapping data)",
            template="plotly_white",
        )
        return figure

    series_columns = [column for column in table.columns if column != "Period"]
    for column in series_columns:
        figure.add_trace(
            go.Bar(
                x=table["Period"].astype(str).tolist(),
                y=table[column].astype(float).tolist(),
                name=column,
            )
        )
    figure.update_layout(
        title="Stress Periods",
        xaxis_title="Period",
        yaxis_title="Period Return",
        barmode="group",
        template="plotly_white",
    )
    return figure


def build_weighted_expense_ratio_over_time_figure(
    weights_history: pd.DataFrame,
    expense_ratios: pd.Series,
) -> go.Figure:
    """Plot the weighted expense ratio of the portfolio at each rebalance date."""

    series = weighted_expense_ratio_history(weights_history, expense_ratios)

    figure = go.Figure(
        data=[
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines+markers",
                name="Weighted Expense Ratio",
            )
        ]
    )
    figure.update_layout(
        title="Weighted Expense Ratio Over Time",
        xaxis_title="Rebalance Date",
        yaxis_title="Weighted Expense Ratio",
        template="plotly_white",
    )
    return figure


def _coerce_benchmark_frame(
    benchmark_returns: pd.Series | pd.DataFrame,
) -> pd.DataFrame:
    if isinstance(benchmark_returns, pd.Series):
        return benchmark_returns.to_frame(name=benchmark_returns.name or "Benchmark")
    return benchmark_returns
