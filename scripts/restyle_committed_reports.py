"""Re-render the committed report artefacts in the current presentation.

Why this exists
---------------
The committed reports were produced by run ``run-all-20260425T083756Z-f6f9560b``.
Re-running the pipeline today reproduces the optimized strategy exactly, but the
risk-parity benchmark solves to slightly different weights on a different
machine, so a fresh run would change 17 published benchmark metrics in the
fourth decimal. A change of presentation must not change a number.

This script therefore never touches the optimizer. It reads the series and
table cells already stored inside the previously generated HTML (the pre-restyle
"card" format), and feeds them, untouched, to the same ``render_*`` functions and
page layout the pipeline now uses. Table cell text is carried over verbatim;
figure series are the same float64 values.

It rewrites only presentation artefacts: ``html/latest_report.html``,
``html/backtest_report.html``, ``html/frontier.html`` and ``figures/*.png``.
Metrics JSON, run records and workbooks are not opened.

Usage
-----
Export the pre-restyle HTML from git, then re-render into ``reports/``::

    mkdir -p /tmp/legacy
    git show 0781b7a:reports/html/latest_report.html > /tmp/legacy/latest_report.html
    git show 0781b7a:reports/html/frontier.html > /tmp/legacy/frontier.html
    uv run python scripts/restyle_committed_reports.py --source /tmp/legacy --output reports
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import re
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from plotly.graph_objects import Figure
from plotly.io import write_image

from etf_portfolio.io_utils import atomic_path, atomic_write_text
from etf_portfolio.reporting.layout import (
    Block,
    FigureBlock,
    ListBlock,
    ReportMeta,
    ReportSection,
    TableBlock,
    TextBlock,
    headline_from_metrics,
    render_report_page,
    table_cells,
)
from etf_portfolio.reporting.plots import (
    FIGURE_EXPORT_HEIGHT,
    FIGURE_EXPORT_SCALE,
    FIGURE_EXPORT_WIDTH,
    render_benchmark_comparison,
    render_cumulative_returns,
    render_drawdown,
    render_efficient_frontier,
    render_group_exposure,
    render_rolling_correlation,
    render_rolling_sharpe,
    render_rolling_volatility,
    render_stress_periods,
    render_weighted_expense_ratio,
    render_weights,
)
from etf_portfolio.reporting.report import (
    REPORT_LEDE,
    RESEARCH_TOOL_DISCLAIMER,
    SECTION_EXPLANATIONS,
    write_frontier_page,
)

_NEW_PLOT = re.compile(r'Plotly\.newPlot\(\s*"[^"]+",\s*')
_SECTION = re.compile(r'<section class="card"><div class="card-inner"><h2>(.*?)</h2>', re.S)
_EXPLANATION = re.compile(r'<div class="section-explanation">.*?</div>', re.S)
_BLOCK = re.compile(
    r'(?P<table><table.*?</table>)|(?P<figure><div class="figure-block">.*?</script>)'
    r"|(?P<list><ul>.*?</ul>)|<p>(?P<text>.*?)</p>",
    re.S,
)
_WINDOW = re.compile(r"\((\d+) periods\)")
EXPECTED_FIGURES = frozenset(
    {
        "asset_class_exposure",
        "benchmark_comparison",
        "cumulative_returns",
        "drawdown",
        "efficient_frontier",
        "portfolio_weights",
        "region_exposure",
        "rolling_correlation",
        "rolling_sharpe",
        "rolling_volatility",
        "stress_periods",
        "weighted_expense_over_time",
    }
)


def _decode(values: Any) -> Any:
    """Decode Plotly's base64 typed-array form back to the exact float64 values."""

    if isinstance(values, dict) and "bdata" in values:
        array = np.frombuffer(base64.b64decode(values["bdata"]), dtype=np.dtype(values["dtype"]))
        if "shape" in values:
            array = array.reshape([int(size) for size in str(values["shape"]).split(",")])
        return array
    return values


def _stored_figures(document: str) -> list[tuple[list[dict[str, Any]], dict[str, Any]]]:
    decoder = json.JSONDecoder()
    figures = []
    for match in _NEW_PLOT.finditer(document):
        data, end = decoder.raw_decode(document, match.end())
        layout, _ = decoder.raw_decode(document, document.index("{", end))
        figures.append((data, layout))
    return figures


def _series(trace: dict[str, Any]) -> pd.Series:
    return pd.Series(
        np.asarray(_decode(trace["y"]), dtype=float),
        index=pd.to_datetime(_decode(trace["x"])),
        name=trace.get("name"),
    )


def _x_range(layout: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    stored = layout.get("xaxis", {}).get("range")
    return (pd.Timestamp(stored[0]), pd.Timestamp(stored[1])) if stored else None


def _frontier(data: list[dict[str, Any]]) -> Figure:
    by_name = {trace.get("name"): trace for trace in data}
    line, assets, portfolio = (
        by_name.get("Efficient frontier"),
        by_name["Assets"],
        by_name.get("Portfolio"),
    )
    point = None
    if portfolio is not None:
        point = (float(_decode(portfolio["x"])[0]), float(_decode(portfolio["y"])[0]))
    return render_efficient_frontier(
        frontier_volatility=[] if line is None else list(_decode(line["x"])),
        frontier_return=[] if line is None else list(_decode(line["y"])),
        asset_volatility=list(_decode(assets["x"])),
        asset_return=list(_decode(assets["y"])),
        asset_labels=[str(label) for label in assets["text"]],
        portfolio_point=point,
    )


def _rebuild_figure(data: list[dict[str, Any]], layout: dict[str, Any]) -> tuple[str, Figure]:
    """Map a stored legacy figure to its name and re-render it from its own series."""

    title = str(layout.get("title", {}).get("text", ""))
    window_match = _WINDOW.search(title)
    window = int(window_match.group(1)) if window_match else 0
    if title == "Efficient Frontier":
        return "efficient_frontier", _frontier(data)
    if title == "Cumulative Returns":
        series = {str(trace["name"]): _series(trace) for trace in data}
        return "cumulative_returns", render_cumulative_returns(series, x_range=_x_range(layout))
    if title == "Drawdown":
        series = {str(trace["name"]): _series(trace) for trace in data}
        return "drawdown", render_drawdown(series, x_range=_x_range(layout))
    if title == "Portfolio Weights":
        weights = pd.DataFrame({str(trace["name"]): _series(trace) for trace in data})
        return "portfolio_weights", render_weights(weights)
    if title in {"Portfolio Weights By Asset Class", "Portfolio Weights By Region"}:
        group = "asset class" if title.endswith("Asset Class") else "region"
        exposure = pd.Series(
            np.asarray(_decode(data[0]["values"]), dtype=float),
            index=[str(label) for label in data[0]["labels"]],
        )
        name = "asset_class_exposure" if group == "asset class" else "region_exposure"
        return name, render_group_exposure(
            exposure,
            title=f"Portfolio weights by {group}",
            subtitle=f"Share of the latest portfolio in each {group}",
        )
    if title == "Benchmark Comparison":
        metrics = pd.DataFrame(
            {str(trace["name"]): np.asarray(_decode(trace["y"]), dtype=float) for trace in data},
            index=[str(label) for label in data[0]["x"]],
        )
        return "benchmark_comparison", render_benchmark_comparison(metrics)
    if title.startswith("Rolling Volatility"):
        return "rolling_volatility", render_rolling_volatility(_series(data[0]), window=window)
    if title.startswith("Rolling Sharpe"):
        return "rolling_sharpe", render_rolling_sharpe(_series(data[0]), window=window)
    if title.startswith("Rolling Correlation To Benchmark") and "Unavailable" not in title:
        return "rolling_correlation", render_rolling_correlation(_series(data[0]), window=window)
    if title == "Stress Periods":
        table = pd.DataFrame({"Period": [str(period) for period in data[0]["x"]]})
        for trace in data:
            table[str(trace["name"])] = np.asarray(_decode(trace["y"]), dtype=float)
        return "stress_periods", render_stress_periods(table)
    if title == "Weighted Expense Ratio Over Time":
        return "weighted_expense_over_time", render_weighted_expense_ratio(_series(data[0]))
    raise SystemExit(f"Unrecognised stored figure: {title!r}")


def _parse_report(
    document: str,
) -> tuple[str, list[ReportSection], dict[str, Figure], dict[str, Any]]:
    title_match = re.search(r"<h1>(.*?)</h1>", document, re.S)
    if title_match is None or '<section class="card">' not in document:
        raise SystemExit("Source report is not in the pre-restyle card format.")
    title = html.unescape(title_match.group(1))

    body = document[document.index("<body>") :]
    starts = [match.start() for match in _SECTION.finditer(body)] + [len(body)]
    sections: list[ReportSection] = []
    figures: dict[str, Figure] = {}
    facts: dict[str, Any] = {}
    for start, end in pairwise(starts):
        chunk = body[start:end]
        heading = _SECTION.match(chunk)
        assert heading is not None
        section_title = html.unescape(heading.group(1))
        content = _EXPLANATION.sub("", chunk[heading.end() :], count=1)
        blocks: list[Block] = []
        for match in _BLOCK.finditer(content):
            if match.group("table"):
                blocks.append(TableBlock(match.group("table")))
            elif match.group("figure"):
                (stored,) = _stored_figures(match.group("figure"))
                name, figure = _rebuild_figure(*stored)
                figures[name] = figure
                blocks.append(FigureBlock(name, figure))
                _collect_facts(name, stored[0], facts)
            elif match.group("list"):
                items = re.findall(r"<li>(.*?)</li>", match.group("list"), re.S)
                blocks.append(ListBlock(tuple(html.unescape(item) for item in items)))
            else:
                blocks.append(TextBlock(html.unescape(match.group("text"))))
        sections.append(
            ReportSection(
                title=section_title,
                blocks=tuple(blocks),
                notes=SECTION_EXPLANATIONS.get(section_title, ()),
            )
        )
    return title, sections, figures, facts


def _collect_facts(name: str, data: list[dict[str, Any]], facts: dict[str, Any]) -> None:
    if name == "cumulative_returns":
        index = _series(data[0]).index
        facts["window"] = (str(index.min().date()), str(index.max().date()))
    elif name == "portfolio_weights":
        index = _series(data[0]).index
        facts["tickers"] = tuple(str(trace["name"]) for trace in data)
        facts["rebalance_count"] = len(index)
        facts["last_rebalance"] = str(index.max().date())


def _strategy_metrics(sections: list[ReportSection]) -> dict[str, Any]:
    for section in sections:
        for block in section.blocks:
            if not isinstance(block, TableBlock):
                continue
            header, rows = table_cells(block.html)
            if "Strategy" not in header:
                continue
            for row in rows:
                cells = dict(zip(header, row, strict=False))
                if cells.get("Strategy") == "Optimized Strategy":
                    return cells
    return {}


def _write_png(figure: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as temp_path:
        write_image(
            figure,
            str(temp_path),
            format="png",
            width=FIGURE_EXPORT_WIDTH,
            height=FIGURE_EXPORT_HEIGHT,
            scale=FIGURE_EXPORT_SCALE,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, required=True, help="dir with the legacy HTML")
    parser.add_argument("--output", type=Path, default=Path("reports"), help="reports directory")
    args = parser.parse_args()

    document = (args.source / "latest_report.html").read_text(encoding="utf-8")
    title, sections, figures, facts = _parse_report(document)
    if set(figures) != EXPECTED_FIGURES:
        raise SystemExit(f"Expected {sorted(EXPECTED_FIGURES)}, found {sorted(figures)}")

    run_id = re.search(r"Run ID: ([\w.-]+)", document)
    window = facts.get("window", (None, None))
    meta = ReportMeta(
        run_id=run_id.group(1) if run_id else None,
        window_start=window[0],
        window_end=window[1],
        tickers=facts.get("tickers", ()),
        rebalance_count=facts.get("rebalance_count"),
        last_rebalance=facts.get("last_rebalance"),
        headline=headline_from_metrics(_strategy_metrics(sections)),
    )
    report_html = render_report_page(
        title=title,
        lede=REPORT_LEDE,
        disclaimer=RESEARCH_TOOL_DISCLAIMER,
        meta=meta,
        sections=sections,
    )
    for name in ("latest_report.html", "backtest_report.html"):
        atomic_write_text(args.output / "html" / name, report_html)
    for name, figure in sorted(figures.items()):
        _write_png(figure, args.output / "figures" / f"{name}.png")

    frontier_document = (args.source / "frontier.html").read_text(encoding="utf-8")
    (frontier_data, _), *_ = _stored_figures(frontier_document)
    assets = next(trace for trace in frontier_data if trace.get("name") == "Assets")
    write_frontier_page(
        _frontier(frontier_data),
        args.output / "html" / "frontier.html",
        tickers=[str(label) for label in assets["text"]],
    )
    print(f"Re-rendered report ({len(sections)} sections, {len(figures)} figures) and frontier.")


if __name__ == "__main__":
    main()
