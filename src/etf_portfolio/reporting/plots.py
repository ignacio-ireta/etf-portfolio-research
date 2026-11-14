"""Plotly chart builders for portfolio reports.

Every chart is produced in two steps so presentation never touches the numbers:

* ``build_*`` functions compute the plotted series from pipeline data, then
* ``render_*`` functions turn already-computed series into a styled figure.

The ``render_*`` layer owns the whole look (palette, type, axes, hover, direct
labels). The interactive HTML report and the exported PNG figures are the same
figures, so they always belong to one family.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# --------------------------------------------------------------------------- #
# Design tokens: a printed research note. Cool ink and cool benchmark series on
# off-white paper, with one warm accent reserved for the optimized strategy.
# --------------------------------------------------------------------------- #

PAPER = "#f8f7f3"
INK = "#16202b"
MUTED = "#55616e"
RULE = "#cfccc2"
GRID = "#e4e1d8"
ACCENT = "#c2491d"
ACCENT_WASH = "rgba(194, 73, 29, 0.14)"
PRIMARY_BENCHMARK = "#24466e"
NEUTRAL_BENCHMARK = "#8296ab"

FONT_SERIF = '"Iowan Old Style", "Palatino Linotype", Palatino, "Book Antiqua", Georgia, serif'
FONT_SANS = '"Avenir Next", "Segoe UI", "Helvetica Neue", Helvetica, Arial, sans-serif'
FONT_MONO = 'Menlo, "SF Mono", Consolas, "Liberation Mono", "DejaVu Sans Mono", monospace'

STRATEGY_LABEL = "Optimized Strategy"
PRIMARY_BENCHMARK_LABEL = "Selected Benchmark ETF"
_STRATEGY_ALIASES = frozenset({"Portfolio", "portfolio", STRATEGY_LABEL})

# Colour follows the entity, never its position: a benchmark keeps its hue in
# every figure. Unknown (config-defined) benchmarks draw from the spare hues.
_BENCHMARK_COLORS: dict[str, str] = {
    PRIMARY_BENCHMARK_LABEL: PRIMARY_BENCHMARK,
    "60/40 Portfolio": "#3b82c4",
    "Equal-Weight ETF Universe": "#168a7f",
    "Inverse-Volatility Portfolio": "#8068c9",
    "Minimum-Variance Portfolio": "#64748b",
    "Risk-Parity Portfolio": "#4a9960",
    "Previous Optimized Strategy": "#9a86d6",
}
_SPARE_BENCHMARK_COLORS: tuple[str, ...] = ("#1d9bc0", "#5b6fb5", "#3f8f9e", "#7c8aa0")

# Holdings are composition, not performance: cool hues plus two muted earths,
# and never the strategy accent.
_HOLDING_COLORS: tuple[str, ...] = (
    "#24466e",
    "#3b82c4",
    "#9cc3e6",
    "#168a7f",
    "#7fc4b8",
    "#64748b",
    "#b8a078",
    "#8068c9",
    "#c5b8e8",
    "#4a9960",
    "#a9b4c0",
    "#8a6f52",
)

PLOTLY_CONFIG: dict[str, Any] = {
    "responsive": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
}

FIGURE_EXPORT_WIDTH = 1000
FIGURE_EXPORT_HEIGHT = 600
FIGURE_EXPORT_SCALE = 2.8  # 1000x600 CSS px rendered at 2800x1680


def series_colors(names: Sequence[str]) -> dict[str, str]:
    """Return the fixed colour for each named return series."""

    colors: dict[str, str] = {}
    spare = iter(_SPARE_BENCHMARK_COLORS)
    for name in names:
        if name in _STRATEGY_ALIASES:
            colors[name] = ACCENT
        elif name in _BENCHMARK_COLORS:
            colors[name] = _BENCHMARK_COLORS[name]
        else:
            colors[name] = next(spare, NEUTRAL_BENCHMARK)
    return colors


def display_name(name: str) -> str:
    """Name the strategy series the same way the tables and metrics JSON do."""

    return STRATEGY_LABEL if name in _STRATEGY_ALIASES else str(name)


def without_title(figure: go.Figure) -> go.Figure:
    """Return a copy for captioned embedding: the caption carries the title."""

    embedded = go.Figure(figure)
    top = 44 if _has_subplot_titles(embedded) else 18
    embedded.update_layout(title_text="", margin_t=top)
    return embedded


def _has_subplot_titles(figure: go.Figure) -> bool:
    return any(
        getattr(annotation, "name", None) == "panel-title"
        for annotation in figure.layout.annotations
    )


def _title_html(title: str, subtitle: str | None) -> str:
    if not subtitle:
        return title
    return (
        f"{title}<br>"
        f'<span style="font-family:{_css_font(FONT_SANS)};font-size:13px;color:{MUTED}">'
        f"{subtitle}</span>"
    )


def _css_font(stack: str) -> str:
    return stack.replace('"', "'")


def _apply_theme(
    figure: go.Figure,
    *,
    title: str,
    subtitle: str | None = None,
    height: int = 460,
    legend: bool = False,
    margin_right: int = 28,
    unified_hover: bool = False,
) -> go.Figure:
    figure.update_layout(
        template="none",
        title={
            "text": _title_html(title, subtitle),
            "x": 0,
            "xref": "container",
            "xanchor": "left",
            "y": 1 - 20 / height,
            "yref": "container",
            "yanchor": "top",
            "pad": {"l": 20},
            "font": {"family": FONT_SERIF, "size": 21, "color": INK},
        },
        font={"family": FONT_MONO, "size": 11.5, "color": MUTED},
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        height=height,
        margin={"l": 64, "r": margin_right, "t": 84, "b": 92 if legend else 48},
        showlegend=legend,
        legend={
            "orientation": "h",
            "x": 0,
            "xanchor": "left",
            "y": -0.14,
            "yanchor": "top",
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "font": {"family": FONT_SANS, "size": 12, "color": INK},
            "itemsizing": "constant",
            "traceorder": "normal",
        },
        hoverlabel={
            "bgcolor": "#ffffff",
            "bordercolor": RULE,
            "font": {"family": FONT_MONO, "size": 12, "color": INK},
            "namelength": -1,
        },
        hovermode="x unified" if unified_hover else "closest",
        colorway=[ACCENT, PRIMARY_BENCHMARK, *_SPARE_BENCHMARK_COLORS],
    )
    figure.update_xaxes(
        showgrid=False,
        showline=True,
        linecolor=INK,
        linewidth=1,
        ticks="outside",
        ticklen=5,
        tickcolor=INK,
        zeroline=False,
        automargin=True,
        title_font={"family": FONT_SANS, "size": 12.5, "color": MUTED},
    )
    figure.update_yaxes(
        showgrid=True,
        gridcolor=GRID,
        gridwidth=1,
        showline=False,
        ticks="",
        zeroline=True,
        zerolinecolor=RULE,
        zerolinewidth=1,
        automargin=True,
        title_font={"family": FONT_SANS, "size": 12.5, "color": MUTED},
    )
    return figure


def _date_axis(figure: go.Figure, index: pd.Index | None = None) -> None:
    figure.update_xaxes(hoverformat="%d %b %Y", tickformat="%Y", dtick="M24", title_text="")
    if index is not None and len(index) > 0:
        figure.update_xaxes(range=[index.min(), index.max()])


def _as_series_map(
    series: Mapping[str, pd.Series] | pd.DataFrame | pd.Series,
) -> dict[str, pd.Series]:
    if isinstance(series, pd.Series):
        return {str(series.name or STRATEGY_LABEL): series}
    if isinstance(series, pd.DataFrame):
        return {str(column): series[column] for column in series.columns}
    return {str(name): values for name, values in series.items()}


def _last_valid(values: pd.Series) -> tuple[Any, float] | None:
    clean = values.dropna()
    if clean.empty:
        return None
    return clean.index[-1], float(clean.iloc[-1])


def _line_width(name: str) -> float:
    if name in _STRATEGY_ALIASES:
        return 2.4
    if name == PRIMARY_BENCHMARK_LABEL:
        return 1.8
    return 1.2


# --------------------------------------------------------------------------- #
# Renderers: styled figures from already-computed series.
# --------------------------------------------------------------------------- #


def render_efficient_frontier(
    *,
    frontier_volatility: Sequence[float],
    frontier_return: Sequence[float],
    asset_volatility: Sequence[float],
    asset_return: Sequence[float],
    asset_labels: Sequence[str],
    portfolio_point: tuple[float, float] | None = None,
) -> go.Figure:
    """Render the efficient frontier, the single assets, and the portfolio."""

    point_hover = "%{text}<br>volatility %{x:.2%}<br>expected return %{y:.2%}<extra></extra>"
    figure = go.Figure()
    if len(frontier_volatility) > 0:
        figure.add_trace(
            go.Scatter(
                x=list(frontier_volatility),
                y=list(frontier_return),
                mode="lines",
                name="Efficient frontier",
                line={"color": INK, "width": 2},
                hovertemplate=(
                    "Efficient frontier<br>volatility %{x:.2%}"
                    "<br>expected return %{y:.2%}<extra></extra>"
                ),
            )
        )
    figure.add_trace(
        go.Scatter(
            x=list(asset_volatility),
            y=list(asset_return),
            mode="markers+text",
            name="Single ETFs",
            text=[str(label) for label in asset_labels],
            textposition="top center",
            textfont={"family": FONT_MONO, "size": 11, "color": INK},
            marker={
                "size": 9,
                "color": "#3b82c4",
                "line": {"color": PAPER, "width": 1.5},
            },
            cliponaxis=False,
            hovertemplate=point_hover,
        )
    )
    if portfolio_point is not None:
        figure.add_trace(
            go.Scatter(
                x=[portfolio_point[0]],
                y=[portfolio_point[1]],
                mode="markers+text",
                name="Optimized portfolio",
                text=["Optimized portfolio"],
                textposition="middle left",
                textfont={"family": FONT_SANS, "size": 12.5, "color": INK},
                marker={
                    "size": 15,
                    "symbol": "diamond",
                    "color": ACCENT,
                    "line": {"color": PAPER, "width": 2},
                },
                cliponaxis=False,
                hovertemplate=point_hover,
            )
        )

    _apply_theme(
        figure,
        title="Efficient frontier",
        subtitle="Annualized expected return against volatility, under the configured constraints",
        height=520,
        legend=True,
    )
    figure.update_layout(margin_b=104)
    figure.update_xaxes(title_text="Annualized volatility", tickformat=".0%", hoverformat=".2%")
    figure.update_yaxes(title_text="Expected return", tickformat=".0%", hoverformat=".2%")
    return figure


def _render_return_lines(
    series: Mapping[str, pd.Series] | pd.DataFrame | pd.Series,
    *,
    title: str,
    subtitle: str,
    tickformat: str,
    hover_value: str,
    wash_strategy: bool,
    end_labels: bool,
    x_range: tuple[Any, Any] | None,
) -> go.Figure:
    series_map = _as_series_map(series)
    colors = series_colors(list(series_map))
    figure = go.Figure()
    # Draw the field first and the strategy last so the accent sits on top, but
    # keep the legend in reading order (strategy, benchmark, the rest).
    ordered = sorted(
        enumerate(series_map),
        key=lambda item: (
            2 if item[1] in _STRATEGY_ALIASES else 1 if item[1] == PRIMARY_BENCHMARK_LABEL else 0,
            item[0],
        ),
    )
    for rank, name in ordered:
        values = series_map[name]
        is_strategy = name in _STRATEGY_ALIASES
        figure.add_trace(
            go.Scatter(
                x=values.index,
                y=values.to_numpy(),
                mode="lines",
                name=display_name(name),
                legendrank=rank + 1,
                line={"color": colors[name], "width": _line_width(name)},
                fill="tozeroy" if (wash_strategy and is_strategy) else None,
                fillcolor=ACCENT_WASH if (wash_strategy and is_strategy) else None,
                hovertemplate=hover_value,
            )
        )

    _apply_theme(
        figure,
        title=title,
        subtitle=subtitle,
        height=520,
        legend=len(series_map) > 1,
        margin_right=118 if end_labels else 28,
        unified_hover=True,
    )
    figure.update_yaxes(tickformat=tickformat)
    first = next(iter(series_map.values()))
    _date_axis(figure, first.index)
    if x_range is not None:
        figure.update_xaxes(range=list(x_range))
    if end_labels:
        _add_end_labels(figure, series_map)
    return figure


def _add_end_labels(figure: go.Figure, series_map: Mapping[str, pd.Series]) -> None:
    """Direct-label the two series the chart is about; the legend carries the rest."""

    labelled: list[tuple[str, Any, float]] = []
    for name, values in series_map.items():
        if name in _STRATEGY_ALIASES:
            label = "Strategy"
        elif name == PRIMARY_BENCHMARK_LABEL:
            label = "Benchmark"
        else:
            continue
        last = _last_valid(values)
        if last is not None:
            labelled.append((label, last[0], last[1]))
    if not labelled:
        return

    all_values = pd.concat(list(series_map.values())).dropna()
    span = float(all_values.max() - all_values.min()) or 1.0
    plot_height = float(figure.layout.height or 460) - 84.0 - 92.0
    pixels_per_unit = plot_height / (span * 1.1)
    shifts = {label: 0.0 for label, _, _ in labelled}
    if len(labelled) == 2:
        (label_a, _, value_a), (label_b, _, value_b) = labelled
        gap = abs(value_a - value_b) * pixels_per_unit
        if gap < 15.0:
            nudge = (15.0 - gap) / 2.0
            upper, lower = (label_a, label_b) if value_a >= value_b else (label_b, label_a)
            shifts[upper], shifts[lower] = nudge, -nudge

    for label, x_value, y_value in labelled:
        figure.add_annotation(
            x=x_value,
            y=y_value,
            xref="x",
            yref="y",
            text=f"{label} {y_value:+.0%}",
            showarrow=False,
            xanchor="left",
            xshift=8,
            yshift=shifts[label],
            font={"family": FONT_MONO, "size": 11.5, "color": INK},
        )


def render_cumulative_returns(
    cumulative: Mapping[str, pd.Series] | pd.DataFrame | pd.Series,
    *,
    x_range: tuple[Any, Any] | None = None,
) -> go.Figure:
    """Render cumulative return paths; the first/strategy series takes the accent."""

    return _render_return_lines(
        cumulative,
        title="Cumulative returns",
        subtitle="Compounded return since the first backtest date, strategy against benchmarks",
        tickformat=".0%",
        hover_value="%{y:+.1%}",
        wash_strategy=False,
        end_labels=True,
        x_range=x_range,
    )


def render_drawdown(
    drawdowns: Mapping[str, pd.Series] | pd.DataFrame | pd.Series,
    *,
    x_range: tuple[Any, Any] | None = None,
) -> go.Figure:
    """Render drawdown paths and mark the strategy's deepest point."""

    series_map = _as_series_map(drawdowns)
    figure = _render_return_lines(
        series_map,
        title="Drawdown",
        subtitle="Decline from the running peak; the shaded area is the optimized strategy",
        tickformat=".0%",
        hover_value="%{y:.1%}",
        wash_strategy=True,
        end_labels=False,
        x_range=x_range,
    )
    strategy = next(
        (values for name, values in series_map.items() if name in _STRATEGY_ALIASES),
        None,
    )
    if strategy is not None and not strategy.dropna().empty:
        trough_date = strategy.idxmin()
        trough = float(strategy.min())
        figure.add_trace(
            go.Scatter(
                x=[trough_date],
                y=[trough],
                mode="markers",
                marker={"size": 9, "color": ACCENT, "line": {"color": PAPER, "width": 2}},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        figure.add_annotation(
            x=trough_date,
            y=trough,
            text=f"Strategy trough {trough:.1%}",
            showarrow=False,
            xanchor="left",
            xshift=10,
            yshift=-4,
            font={"family": FONT_MONO, "size": 11.5, "color": INK},
        )
    return figure


def render_weights(weights: pd.DataFrame) -> go.Figure:
    """Render holdings through time as a stacked area that sums to 100%."""

    figure = go.Figure()
    for position, column in enumerate(weights.columns):
        color = _HOLDING_COLORS[position % len(_HOLDING_COLORS)]
        figure.add_trace(
            go.Scatter(
                x=weights.index,
                y=weights[column].to_numpy(),
                mode="lines",
                stackgroup="weights",
                name=str(column),
                groupnorm="fraction" if position == 0 else None,
                line={"color": PAPER, "width": 0.8},
                fillcolor=color,
                marker={"color": color},
                hovertemplate="%{y:.1%}",
            )
        )

    _apply_theme(
        figure,
        title="Portfolio weights",
        subtitle="Share of the portfolio held in each ETF at every rebalance date",
        height=520,
        legend=True,
        unified_hover=True,
    )
    figure.update_yaxes(tickformat=".0%", range=[0, 1], zeroline=False)
    _date_axis(figure, weights.index)
    return figure


def render_group_exposure(exposure: pd.Series, *, title: str, subtitle: str) -> go.Figure:
    """Render latest group exposure as sorted, directly labelled bars."""

    ordered = exposure.astype(float).sort_values(ascending=True)
    labels = [str(label).replace("_", " ") for label in ordered.index]
    figure = go.Figure(
        data=[
            go.Bar(
                x=ordered.to_numpy(),
                y=labels,
                orientation="h",
                name=title,
                marker={"color": PRIMARY_BENCHMARK, "line": {"width": 0}},
                width=0.22,
                text=[f"{value:.1%}" for value in ordered],
                textposition="outside",
                textfont={"family": FONT_MONO, "size": 12, "color": INK},
                cliponaxis=False,
                hovertemplate="%{y}<br>%{x:.2%} of the portfolio<extra></extra>",
            )
        ]
    )
    _apply_theme(figure, title=title, subtitle=subtitle, height=120 + 62 * max(len(labels), 2))
    peak = float(ordered.max()) if len(ordered) else 1.0
    figure.update_xaxes(
        tickformat=".0%",
        range=[0, max(peak * 1.18, 0.05)],
        showline=False,
        ticks="",
    )
    figure.update_yaxes(
        showgrid=False,
        zeroline=False,
        ticksuffix="  ",
        tickfont={"family": FONT_SANS, "size": 13, "color": INK},
    )
    figure.update_layout(margin_r=64)
    return figure


_COMPARISON_PANELS: tuple[tuple[str, str, str], ...] = (
    ("CAGR", "CAGR", ".1%"),
    ("Volatility", "Annualized volatility", ".1%"),
    ("Sharpe", "Sharpe ratio", ".2f"),
    ("Max Drawdown", "Max drawdown", ".1%"),
)


def render_benchmark_comparison(metrics: pd.DataFrame) -> go.Figure:
    """Render strategy-versus-benchmark metrics as small multiples.

    ``metrics`` has one row per metric (CAGR, Volatility, Sharpe, Max Drawdown)
    and one column per series. Each metric gets its own panel and scale: a ratio
    and three percentages never share an axis.
    """

    names = [str(column) for column in metrics.columns]
    row_labels = [display_name(name) for name in names][::-1]
    bar_colors = [
        ACCENT
        if name in _STRATEGY_ALIASES
        else PRIMARY_BENCHMARK
        if name == PRIMARY_BENCHMARK_LABEL
        else NEUTRAL_BENCHMARK
        for name in names
    ][::-1]
    panels = [panel for panel in _COMPARISON_PANELS if panel[0] in metrics.index]
    columns = 2 if len(panels) > 1 else 1
    rows = max((len(panels) + 1) // 2, 1)
    figure = make_subplots(
        rows=rows,
        cols=columns,
        shared_yaxes=True,
        horizontal_spacing=0.05,
        vertical_spacing=0.14,
    )
    _apply_theme(
        figure,
        title="Benchmark comparison",
        subtitle="Four headline metrics over the common backtest window, one scale per panel",
        height=150 + rows * (70 + 24 * len(names)),
    )
    for position, (metric_key, panel_title, value_format) in enumerate(panels):
        row, column = position // 2 + 1, position % 2 + 1
        values = metrics.loc[metric_key].astype(float).to_numpy()[::-1]
        figure.add_trace(
            go.Bar(
                x=values,
                y=row_labels,
                orientation="h",
                name=panel_title,
                marker={"color": bar_colors, "line": {"width": 0}},
                width=0.62,
                text=[format(value, value_format) for value in values],
                textposition="outside",
                textfont={"family": FONT_MONO, "size": 11.5, "color": INK},
                cliponaxis=False,
                hovertemplate=f"%{{y}}<br>{panel_title} %{{x:{value_format}}}<extra></extra>",
            ),
            row=row,
            col=column,
        )
        finite = values[np.isfinite(values)]
        low = min(float(finite.min()), 0.0) if finite.size else 0.0
        high = max(float(finite.max()), 0.0) if finite.size else 1.0
        span = (high - low) or 1.0
        figure.update_xaxes(
            range=[
                low - (0.30 * span if low < 0 else 0.0),
                high + (0.30 * span if high > 0 else 0),
            ],
            showticklabels=False,
            showline=False,
            ticks="",
            showgrid=False,
            zeroline=True,
            zerolinecolor=INK,
            zerolinewidth=1,
            row=row,
            col=column,
        )
        figure.update_yaxes(
            showgrid=False,
            zeroline=False,
            ticksuffix="  ",
            tickfont={"family": FONT_SANS, "size": 12.5, "color": INK},
            row=row,
            col=column,
        )
        axis_suffix = "" if position == 0 else str(position + 1)
        figure.add_annotation(
            name="panel-title",
            text=panel_title.upper(),
            xref=f"x{axis_suffix} domain",
            yref=f"y{axis_suffix} domain",
            x=0,
            y=1,
            xanchor="left",
            yanchor="bottom",
            yshift=8,
            showarrow=False,
            font={"family": FONT_MONO, "size": 11, "color": MUTED},
        )
    figure.update_layout(showlegend=False, margin_r=36, margin_b=28, bargap=0.3)
    return figure


def _render_single_line(
    series: pd.Series,
    *,
    title: str,
    subtitle: str,
    tickformat: str,
    hover_value: str,
    name: str = STRATEGY_LABEL,
    step: bool = False,
) -> go.Figure:
    figure = go.Figure(
        data=[
            go.Scatter(
                x=series.index,
                y=series.to_numpy(),
                mode="lines",
                name=name,
                line={"color": ACCENT, "width": 1.8, "shape": "hv" if step else "linear"},
                hovertemplate=hover_value,
            )
        ]
    )
    _apply_theme(figure, title=title, subtitle=subtitle, height=400, unified_hover=True)
    figure.update_yaxes(tickformat=tickformat)
    _date_axis(figure, series.dropna().index)
    return figure


def render_rolling_volatility(series: pd.Series, *, window: int) -> go.Figure:
    """Render the strategy's rolling annualized volatility."""

    return _render_single_line(
        series,
        title=f"Rolling volatility ({window} periods)",
        subtitle=f"Annualized volatility of the optimized strategy over a {window}-period window",
        tickformat=".0%",
        hover_value="%{y:.1%}",
    )


def render_rolling_sharpe(series: pd.Series, *, window: int) -> go.Figure:
    """Render the strategy's rolling Sharpe ratio."""

    return _render_single_line(
        series,
        title=f"Rolling Sharpe ({window} periods)",
        subtitle=f"Annualized Sharpe ratio of the optimized strategy over a {window}-period window",
        tickformat=".1f",
        hover_value="%{y:.2f}",
    )


def render_rolling_correlation(series: pd.Series, *, window: int) -> go.Figure:
    """Render the strategy's rolling correlation to the primary benchmark."""

    return _render_single_line(
        series,
        title=f"Rolling correlation to benchmark ({window} periods)",
        subtitle=(
            f"Correlation of strategy and primary-benchmark returns over a {window}-period window"
        ),
        tickformat=".2f",
        hover_value="%{y:.3f}",
        name="Strategy vs benchmark",
    )


def render_stress_periods(table: pd.DataFrame) -> go.Figure:
    """Render compounded returns over the named stress windows."""

    figure = go.Figure()
    if table.empty or "Period" not in table.columns:
        return _apply_theme(
            figure,
            title="Stress periods (no overlapping data)",
            subtitle="None of the named stress windows overlap the backtest",
        )

    series_columns = [str(column) for column in table.columns if column != "Period"]
    colors = series_colors(series_columns)
    periods = [str(period).replace(" (", "<br>(") for period in table["Period"]]
    for column in series_columns:
        figure.add_trace(
            go.Bar(
                x=periods,
                y=table[column].astype(float).tolist(),
                name=display_name(column),
                marker={"color": colors[column], "line": {"color": PAPER, "width": 1}},
                hovertemplate="%{y:+.1%}",
            )
        )
    _apply_theme(
        figure,
        title="Stress periods",
        subtitle="Compounded return of each series inside the named stress windows",
        height=520,
        legend=True,
        unified_hover=True,
    )
    figure.update_layout(barmode="group", bargap=0.24, bargroupgap=0.0, margin_b=124)
    figure.update_layout(legend_y=-0.22)
    figure.update_xaxes(
        showline=False, ticks="", tickfont={"family": FONT_SANS, "size": 12, "color": INK}
    )
    figure.update_yaxes(tickformat=".0%", zerolinecolor=INK)
    return figure


def render_weighted_expense_ratio(series: pd.Series) -> go.Figure:
    """Render the weighted expense ratio held between rebalance dates."""

    return _render_single_line(
        series,
        title="Weighted expense ratio over time",
        subtitle="Weight-averaged ETF expense ratio of the optimized strategy at each rebalance",
        tickformat=".3%",
        hover_value="%{y:.4%}",
        name="Weighted expense ratio",
        step=True,
    )


# --------------------------------------------------------------------------- #
# Builders: compute the plotted series from pipeline data, then render.
# --------------------------------------------------------------------------- #


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

    portfolio_point: tuple[float, float] | None = None
    if portfolio_weights is not None:
        portfolio_point = (
            float(calculate_portfolio_volatility(portfolio_weights, covariance_matrix)),
            float(calculate_portfolio_return(portfolio_weights, expected_returns)),
        )

    return render_efficient_frontier(
        frontier_volatility=[point["portfolio_volatility"] for point in frontier_points],
        frontier_return=[point["portfolio_return"] for point in frontier_points],
        asset_volatility=np.sqrt(np.diag(covariance_matrix)).tolist(),
        asset_return=expected_returns.tolist(),
        asset_labels=[str(label) for label in expected_returns.index],
        portfolio_point=portfolio_point,
    )


def _aligned_series(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | pd.DataFrame | None,
    transform: Any,
) -> tuple[dict[str, pd.Series], tuple[Any, Any] | None]:
    """Transform the strategy and each aligned benchmark with one function."""

    series: dict[str, pd.Series] = {STRATEGY_LABEL: transform(portfolio_returns)}
    x_range: tuple[Any, Any] | None = None
    if benchmark_returns is not None:
        benchmark_frame = _coerce_benchmark_frame(benchmark_returns)
        aligned_portfolio, aligned_benchmarks = portfolio_returns.align(
            benchmark_frame,
            join="inner",
            axis=0,
        )
        for column in aligned_benchmarks.columns:
            series[str(column)] = transform(aligned_benchmarks[column])
        if len(aligned_portfolio.index) > 0:
            x_range = (aligned_portfolio.index.min(), aligned_portfolio.index.max())
    return series, x_range


def build_cumulative_returns_figure(
    portfolio_returns: pd.Series,
    *,
    benchmark_returns: pd.Series | pd.DataFrame | None = None,
) -> go.Figure:
    """Build a cumulative returns chart."""

    series, x_range = _aligned_series(portfolio_returns, benchmark_returns, cumulative_returns)
    return render_cumulative_returns(series, x_range=x_range)


def build_drawdown_figure(
    portfolio_returns: pd.Series,
    *,
    benchmark_returns: pd.Series | pd.DataFrame | None = None,
) -> go.Figure:
    """Build a drawdown chart."""

    series, x_range = _aligned_series(portfolio_returns, benchmark_returns, drawdown_series)
    return render_drawdown(series, x_range=x_range)


def build_weights_figure(weights: pd.DataFrame) -> go.Figure:
    """Build a stacked area chart of portfolio weights over rebalance dates."""

    return render_weights(weights)


def build_group_exposure_figure(
    weights: pd.DataFrame,
    groups: pd.Series,
    *,
    title: str,
    subtitle: str = "Share of the latest portfolio in each group",
) -> go.Figure:
    """Build a latest portfolio exposure bar chart for a metadata grouping."""

    latest = latest_portfolio_weights(weights)
    exposure = aggregate_group_exposure(latest, groups.reindex(latest.index))
    return render_group_exposure(exposure, title=title, subtitle=subtitle)


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
        STRATEGY_LABEL: [
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
    return render_benchmark_comparison(
        pd.DataFrame(metrics, index=["CAGR", "Volatility", "Sharpe", "Max Drawdown"])
    )


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
    return render_rolling_volatility(rolling, window=window)


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
    return render_rolling_sharpe(rolling, window=window)


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
    return render_rolling_correlation(rolling, window=window)


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
    return render_stress_periods(table)


def build_weighted_expense_ratio_over_time_figure(
    weights_history: pd.DataFrame,
    expense_ratios: pd.Series,
) -> go.Figure:
    """Plot the weighted expense ratio of the portfolio at each rebalance date."""

    return render_weighted_expense_ratio(
        weighted_expense_ratio_history(weights_history, expense_ratios)
    )


def _coerce_benchmark_frame(
    benchmark_returns: pd.Series | pd.DataFrame,
) -> pd.DataFrame:
    if isinstance(benchmark_returns, pd.Series):
        return benchmark_returns.to_frame(name=benchmark_returns.name or "Benchmark")
    return benchmark_returns
