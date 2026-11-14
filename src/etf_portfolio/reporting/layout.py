"""Page layout for the generated reports: a printed research note.

This module is presentation only. It receives finished tables, figures, and
text and decides how they sit on the page:

* structure comes from type and hairline rules, not stacked cards;
* three voices: a serif for statements, a sans for reading, a monospace for
  numbers, labels, and units (system stacks only, so the self-contained HTML
  keeps working offline);
* every chart is a numbered figure with a caption saying what it shows, and
  every table is ruled with tabular numerals;
* the sheet is deliberately light in every colour scheme. It is a printed
  tearsheet, and the figures bake the same paper colour into their exports.
"""

from __future__ import annotations

import html
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from html.parser import HTMLParser

import pandas as pd
from plotly.graph_objects import Figure
from plotly.io import to_html
from plotly.offline import get_plotlyjs

from etf_portfolio.reporting.plots import (
    ACCENT,
    FONT_MONO,
    FONT_SANS,
    FONT_SERIF,
    GRID,
    INK,
    MUTED,
    NEUTRAL_BENCHMARK,
    PAPER,
    PLOTLY_CONFIG,
    PRIMARY_BENCHMARK,
    RULE,
    without_title,
)

ACCENT_INK = "#a13a14"  # the accent darkened for small text on paper (6.5:1)


@dataclass(frozen=True)
class TableBlock:
    """A table as pandas-generated HTML (cell text is never reformatted)."""

    html: str


@dataclass(frozen=True)
class FigureBlock:
    """A named Plotly figure; the name selects its caption."""

    name: str
    figure: Figure


@dataclass(frozen=True)
class ListBlock:
    """A plain list of statements."""

    items: tuple[str, ...]


@dataclass(frozen=True)
class TextBlock:
    """A paragraph of prose."""

    text: str


Block = TableBlock | FigureBlock | ListBlock | TextBlock


@dataclass(frozen=True)
class ReportSection:
    """One numbered section of the note."""

    title: str
    blocks: tuple[Block, ...]
    notes: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ReportMeta:
    """Run facts shown in the masthead. Every value comes from the run itself."""

    run_id: str | None = None
    window_start: str | None = None
    window_end: str | None = None
    tickers: tuple[str, ...] = ()
    rebalance_count: int | None = None
    last_rebalance: str | None = None
    headline: tuple[tuple[str, str], ...] = field(default_factory=tuple)


FIGURE_CAPTIONS: dict[str, str] = {
    "efficient_frontier": (
        "Efficient frontier. Estimated return against volatility for constrained "
        "portfolios (line), each ETF held alone (dots), and the optimized portfolio (diamond)."
    ),
    "portfolio_weights": (
        "Portfolio weights. Share of the portfolio in each ETF at every rebalance date; "
        "the bands always sum to 100%."
    ),
    "asset_class_exposure": (
        "Asset-class exposure. Weight of the latest portfolio in each asset class."
    ),
    "region_exposure": "Regional exposure. Weight of the latest portfolio in each region.",
    "benchmark_comparison": (
        "Benchmark comparison. CAGR, annualized volatility, Sharpe ratio, and maximum "
        "drawdown for the optimized strategy and every benchmark. Each panel has its own scale."
    ),
    "cumulative_returns": (
        "Cumulative returns. Compounded return of the optimized strategy and each benchmark "
        "since the first backtest date."
    ),
    "drawdown": (
        "Drawdown. Decline of each series from its own running peak; the shaded area is the "
        "optimized strategy and the marker its deepest point."
    ),
    "rolling_volatility": (
        "Rolling volatility. Annualized volatility of the optimized strategy over a trailing "
        "window."
    ),
    "rolling_sharpe": (
        "Rolling Sharpe ratio. Annualized Sharpe ratio of the optimized strategy over a "
        "trailing window."
    ),
    "rolling_correlation": (
        "Rolling correlation. Correlation between the optimized strategy and the primary "
        "benchmark over a trailing window."
    ),
    "stress_periods": (
        "Stress periods. Compounded return of the optimized strategy and each benchmark "
        "inside the named stress windows."
    ),
    "weighted_expense_over_time": (
        "Weighted expense ratio. Weight-averaged ETF expense ratio of the portfolio, "
        "held constant between rebalance dates."
    ),
}

TABLE_CAPTIONS: dict[str, tuple[str, ...]] = {
    "Reader Guide": ("What this note is, and the state of its inputs.",),
    "Trust And Safety": ("Common false conclusions and the safer reading of each.",),
    "Metric Dictionary": ("Definitions of the metrics used in this note.",),
    "ETF Universe Summary": ("The ETFs the optimizer was allowed to hold.",),
    "Data Coverage Table": ("Usable price history for each ETF.",),
    "Missing Data Table": ("Missing price observations for each ETF.",),
    "Latest Realized Portfolio Table": ("Weights held after the latest rebalance.",),
    "Optimizer Target Portfolio Table": (
        "Weights the optimizer asked for at the latest rebalance.",
    ),
    "Exposure": (
        "Latest portfolio weight by asset class.",
        "Latest portfolio weight by region.",
    ),
    "Benchmark Comparison": ("Performance and risk metrics, strategy and benchmarks.",),
    "Backtest Performance": ("Performance and risk metrics, strategy and benchmarks.",),
    "Stress Periods": ("Compounded return inside each stress window.",),
    "Weighted Expense Ratio Over Time": ("Weighted expense ratio at each rebalance date.",),
    "Return Attribution": ("Contribution of each ETF and asset class to portfolio return.",),
    "Risk Attribution": ("Contribution of each ETF and asset class to portfolio volatility.",),
    "Realized Constraint Warnings": ("Realized holdings outside their configured bounds.",),
    "Assumptions and Limitations": (
        "Run parameters.",
        "Modelling assumptions.",
        "Portfolio profile.",
    ),
}

# Figures that need room to stay legible scroll sideways on a phone instead of
# being squeezed.
_WIDE_FIGURES = frozenset({"benchmark_comparison", "stress_periods"})

_HEADLINE_METRICS: tuple[tuple[str, str, str], ...] = (
    ("CAGR", "CAGR", ".1%"),
    ("Annualized Volatility", "Volatility", ".1%"),
    ("Sharpe Ratio", "Sharpe ratio", ".2f"),
    ("Max Drawdown", "Max drawdown", ".1%"),
)

_NUMERIC_CELL = re.compile(r"^[-+]?(\d[\d,]*)(\.\d+)?([eE][-+]?\d+)?%?$")
_DATE_CELL = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?$")


def headline_from_metrics(metrics: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    """Format the optimized strategy's headline metrics for the masthead strip."""

    headline: list[tuple[str, str]] = []
    for key, label, value_format in _HEADLINE_METRICS:
        if key not in metrics:
            continue
        try:
            value = float(metrics[key])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if pd.isna(value):
            continue
        headline.append((label, format(value, value_format).replace("-", "\u2212")))
    return tuple(headline)


def humanize_title(title: str) -> str:
    """Turn a config identifier such as ``etf_portfolio_research`` into words."""

    def fix(match: re.Match[str]) -> str:
        words = match.group(0).split("_")
        return " ".join(
            word.upper() if word.lower() in {"etf", "etfs"} else word.capitalize()
            for word in words
            if word
        )

    return re.sub(r"[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+", fix, title)


def table_html(table: pd.DataFrame) -> str:
    """Render a DataFrame with pandas' own cell formatting."""

    return table.to_html(index=False, border=0, na_rep="")


class _TableReader(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.head: list[list[str]] = []
        self.body: list[list[tuple[str, str]]] = []
        self._section = ""
        self._row: list[tuple[str, str]] | None = None
        self._cell: list[str] | None = None
        self._cell_tag = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"thead", "tbody"}:
            self._section = tag
        elif tag == "tr":
            self._row = []
        elif tag in {"th", "td"}:
            self._cell, self._cell_tag = [], tag
        elif self._cell is not None:
            self._cell.append(self.get_starttag_text() or "")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"th", "td"} and self._cell is not None and self._row is not None:
            self._row.append((self._cell_tag, "".join(self._cell).strip()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._section == "thead":
                self.head.append([text for _, text in self._row])
            else:
                self.body.append(self._row)
            self._row = None
        elif self._cell is not None:
            self._cell.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_entityref(self, name: str) -> None:
        if self._cell is not None:
            self._cell.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if self._cell is not None:
            self._cell.append(f"&#{name};")


def table_cells(pandas_html: str) -> tuple[list[str], list[list[str]]]:
    """Return the header and body cell text of a pandas HTML table, verbatim."""

    reader = _TableReader()
    reader.feed(pandas_html)
    header = reader.head[-1] if reader.head else []
    return header, [[text for _, text in row] for row in reader.body]


def ruled_table(pandas_html: str) -> str:
    """Re-emit a pandas HTML table as a ruled table.

    Cell text passes through untouched. The only changes are structural: numeric
    and date columns are marked so they can be set in tabular monospace and
    right-aligned, and pandas' empty index-name column is dropped.
    """

    reader = _TableReader()
    reader.feed(pandas_html)
    header = reader.head[-1] if reader.head else []
    rows = reader.body
    width = max([len(header), *(len(row) for row in rows)])
    if width == 0:
        return '<div class="table-wrap"><table class="ruled"></table></div>'
    header = header + [""] * (width - len(header))
    rows = [row + [("td", "")] * (width - len(row)) for row in rows]

    keep = [
        index
        for index in range(width)
        if not (rows and all(row[index] == ("th", "") for row in rows))
    ]
    kinds: dict[int, str] = {}
    for index in keep:
        cells = [row[index][1] for row in rows if row[index][1] != ""]
        if cells and all(_NUMERIC_CELL.match(cell) for cell in cells):
            kinds[index] = "num"
        elif cells and all(_DATE_CELL.match(cell) for cell in cells):
            kinds[index] = "date"
        else:
            kinds[index] = ""

    def cell(tag: str, text: str, kind: str) -> str:
        attribute = f' class="{kind}"' if kind else ""
        return f"<{tag}{attribute}>{text}</{tag}>"

    head_html = "".join(cell("th", header[index], kinds[index]) for index in keep)
    body_html = "".join(
        "<tr>" + "".join(cell("td", row[index][1], kinds[index]) for index in keep) + "</tr>"
        for row in rows
    )
    classes = ["table-wrap"]
    if len(keep) > 8:
        classes.append("table-wrap--wide")
    if len(rows) > 24:
        classes.append("table-wrap--tall")
    return (
        f'<div class="{" ".join(classes)}"><table class="ruled">'
        f"<thead><tr>{head_html}</tr></thead><tbody>{body_html}</tbody></table></div>"
    )


def _figure_div(figure: Figure) -> str:
    embedded = without_title(figure)
    height = int(embedded.layout.height or 460)
    return to_html(
        embedded,
        full_html=False,
        include_plotlyjs=False,
        config=PLOTLY_CONFIG,
        default_width="100%",
        default_height=f"{height}px",
    )


class _Counters:
    def __init__(self) -> None:
        self.figure = 0
        self.table = 0


def _render_block(block: Block, section_title: str, table_position: int, count: _Counters) -> str:
    if isinstance(block, FigureBlock):
        count.figure += 1
        caption = FIGURE_CAPTIONS.get(block.name, "")
        wide = " plot--wide" if block.name in _WIDE_FIGURES else ""
        return (
            f'<figure class="plate" id="fig-{count.figure}">'
            f'<div class="plot-scroll"><div class="plot{wide}">'
            f"{_figure_div(block.figure)}</div></div>"
            f'<figcaption><span class="ref">Fig. {count.figure}</span>'
            f"<span>{html.escape(caption)}</span></figcaption></figure>"
        )
    if isinstance(block, TableBlock):
        count.table += 1
        captions = TABLE_CAPTIONS.get(section_title, ())
        caption = captions[table_position] if table_position < len(captions) else ""
        return (
            f'<figure class="tabular" id="table-{count.table}">'
            f'<figcaption><span class="ref">Table {count.table}</span>'
            f"<span>{html.escape(caption)}</span></figcaption>"
            f"{ruled_table(block.html)}</figure>"
        )
    if isinstance(block, ListBlock):
        items = "".join(f"<li>{html.escape(item)}</li>" for item in block.items if item.strip())
        return f'<ul class="statements">{items}</ul>'
    return f'<p class="prose">{html.escape(block.text)}</p>'


def _render_notes(notes: Sequence[tuple[str, str]]) -> str:
    if not notes:
        return ""
    items = "".join(
        f"<div><dt>{html.escape(label)}</dt><dd>{html.escape(description)}</dd></div>"
        for label, description in notes
    )
    return (
        '<aside class="notes"><p class="label">How to read this section</p>'
        f"<dl>{items}</dl></aside>"
    )


def _section_id(position: int) -> str:
    return f"s{position:02d}"


def _render_section(position: int, section: ReportSection, count: _Counters) -> str:
    blocks: list[str] = []
    table_position = 0
    for block in section.blocks:
        blocks.append(_render_block(block, section.title, table_position, count))
        if isinstance(block, TableBlock):
            table_position += 1
    return (
        f'<section class="section" id="{_section_id(position)}">'
        f'<div class="section__margin"><span class="section__num">{position:02d}</span>'
        f"<h2>{html.escape(section.title)}</h2></div>"
        f'<div class="section__body">{_render_notes(section.notes)}{"".join(blocks)}</div>'
        "</section>"
    )


def _spec_row(label: str, value: str) -> str:
    return f"<div><dt>{html.escape(label)}</dt><dd>{value}</dd></div>"


def _render_spec(meta: ReportMeta) -> str:
    rows: list[str] = []
    if meta.run_id:
        rows.append(_spec_row("Run", html.escape(meta.run_id)))
    if meta.window_start and meta.window_end:
        rows.append(
            _spec_row(
                "Backtest window",
                f"{html.escape(meta.window_start)} → {html.escape(meta.window_end)}",
            )
        )
    if meta.rebalance_count is not None:
        latest = f", latest {html.escape(meta.last_rebalance)}" if meta.last_rebalance else ""
        rows.append(_spec_row("Rebalances", f"{meta.rebalance_count}{latest}"))
    if meta.tickers:
        tickers = " ".join(html.escape(ticker) for ticker in meta.tickers)
        rows.append(
            f'<div class="spec__wide"><dt>Universe · {len(meta.tickers)} ETFs</dt>'
            f"<dd>{tickers}</dd></div>"
        )
    return f'<dl class="spec">{"".join(rows)}</dl>' if rows else ""


def _render_headline(meta: ReportMeta) -> str:
    if not meta.headline:
        return ""
    figures = "".join(
        f'<div><span class="value">{html.escape(value)}</span>'
        f'<span class="label">{html.escape(label)}</span></div>'
        for label, value in meta.headline
    )
    return (
        '<div class="headline"><p class="label"><i class="key key--strategy"></i>'
        "Optimized strategy · full backtest window</p>"
        f'<div class="headline__figures">{figures}</div></div>'
    )


_COLOR_KEY = (
    '<p class="colorkey"><span class="label">Colour key, every figure</span>'
    '<span><i class="key key--strategy"></i>Optimized strategy</span>'
    '<span><i class="key key--benchmark"></i>Selected benchmark ETF</span>'
    '<span><i class="key key--field"></i>Other benchmarks, in cool hues</span></p>'
)


def _page(*, title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>{html.escape(title)}</title>
  <script type="text/javascript">{get_plotlyjs()}</script>
  <style>{_CSS}</style>
</head>
<body>
  <main class="note">
{body}
  </main>
</body>
</html>
"""


def render_report_page(
    *,
    title: str,
    lede: str,
    disclaimer: str,
    meta: ReportMeta,
    sections: Sequence[ReportSection],
) -> str:
    """Render the full research note as one self-contained HTML document."""

    display_title = humanize_title(title)
    count = _Counters()
    rendered = "".join(
        _render_section(position, section, count)
        for position, section in enumerate(sections, start=1)
    )
    contents = "".join(
        f'<li><a href="#{_section_id(position)}">'
        f'<span class="num">{position:02d}</span>{html.escape(section.title)}</a></li>'
        for position, section in enumerate(sections, start=1)
    )
    body = f"""    <header class="masthead">
      <p class="kicker">Research note · Walk-forward backtest</p>
      <h1>{html.escape(display_title)}</h1>
      <p class="standfirst">{html.escape(lede)}</p>
      {_render_spec(meta)}
      {_render_headline(meta)}
      {_COLOR_KEY}
      <p class="notice"><span class="label">Notice</span>{html.escape(disclaimer)}</p>
      <nav class="contents" aria-label="Contents">
        <p class="label">Contents</p>
        <ol>{contents}</ol>
      </nav>
    </header>
    {rendered}
    <footer class="colophon">
      <p>Generated by the etf-portfolio-research pipeline. Tables show pipeline values
      unrounded beyond six decimals; figures plot the same series.</p>
    </footer>"""
    return _page(title=display_title, body=body)


def render_frontier_page(
    figure: Figure,
    *,
    title: str = "Efficient Frontier",
    lede: str,
    disclaimer: str,
    meta: ReportMeta,
    notes: Sequence[tuple[str, str]] = (),
) -> str:
    """Render the standalone efficient-frontier explorer in the same language."""

    count = _Counters()
    plate = _render_block(FigureBlock("efficient_frontier", figure), "", 0, count)
    body = f"""    <header class="masthead masthead--compact">
      <p class="kicker">Research note \u00b7 Efficient-frontier explorer</p>
      <h1>{html.escape(humanize_title(title))}</h1>
      <p class="standfirst">{html.escape(lede)}</p>
    </header>
    <div class="explorer">
      {plate}
      {_render_notes(notes)}
      {_render_spec(meta)}
      <p class="notice"><span class="label">Notice</span>{html.escape(disclaimer)}</p>
    </div>"""
    return _page(title=humanize_title(title), body=body)


_CSS = f"""
    :root {{
      --paper: {PAPER};
      --ink: {INK};
      --muted: {MUTED};
      --rule: {RULE};
      --hair: {GRID};
      --accent: {ACCENT};
      --accent-ink: {ACCENT_INK};
      --benchmark: {PRIMARY_BENCHMARK};
      --field: {NEUTRAL_BENCHMARK};
      --serif: {FONT_SERIF};
      --sans: {FONT_SANS};
      --mono: {FONT_MONO};
      color-scheme: light;
    }}
    * {{ box-sizing: border-box; }}
    html {{ background: var(--paper); -webkit-text-size-adjust: 100%; }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: var(--sans);
      font-size: 16px;
      line-height: 1.55;
      -webkit-font-smoothing: antialiased;
    }}
    .note {{ max-width: 1180px; margin: 0 auto; padding: 0 32px 96px; }}
    .label, .kicker, .section__num, .ref, thead th, .notes dt, .spec dt {{
      font-family: var(--mono);
      font-size: 0.7rem;
      font-weight: 500;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    a {{ color: inherit; }}

    .masthead {{ border-top: 6px solid var(--ink); padding: 28px 0 44px; }}
    .kicker {{ color: var(--accent-ink); margin: 0 0 1.4rem; }}
    h1 {{
      font-family: var(--serif);
      font-weight: 600;
      font-size: clamp(2.1rem, 5vw, 3.6rem);
      line-height: 1.04;
      letter-spacing: -0.012em;
      margin: 0 0 1.1rem;
      max-width: 22ch;
    }}
    .standfirst {{
      font-family: var(--serif);
      font-size: 1.22rem;
      line-height: 1.45;
      max-width: 60ch;
      margin: 0 0 2.2rem;
      color: #2a3644;
    }}
    .spec {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      column-gap: 32px;
      margin: 0;
      border-top: 1px solid var(--ink);
    }}
    .spec > div {{ padding: 0.7rem 0 0.8rem; border-bottom: 1px solid var(--rule); }}
    .spec .spec__wide {{ grid-column: 1 / -1; }}
    .spec dd {{
      margin: 0.2rem 0 0;
      font-family: var(--mono);
      font-size: 0.9rem;
      font-variant-numeric: tabular-nums;
      overflow-wrap: anywhere;
    }}
    .spec__wide dd {{ word-spacing: 0.7em; }}
    .headline {{ margin-top: 2.2rem; }}
    .headline > .label, .colorkey .label {{ margin: 0 0 0.6rem; }}
    .headline__figures {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      border-top: 2px solid var(--accent);
    }}
    .headline__figures > div {{
      padding: 0.9rem 1rem 0.2rem 0;
      display: flex;
      flex-direction: column;
      gap: 0.3rem;
    }}
    .headline__figures > div + div {{ border-left: 1px solid var(--rule); padding-left: 1.2rem; }}
    .headline .value {{
      font-family: var(--mono);
      font-size: clamp(1.6rem, 3.6vw, 2.5rem);
      font-weight: 500;
      line-height: 1;
      letter-spacing: -0.03em;
      font-variant-numeric: tabular-nums;
    }}
    .key {{
      display: inline-block;
      width: 1.4em;
      height: 0;
      border-top: 3px solid var(--accent);
      vertical-align: middle;
      margin-right: 0.6em;
    }}
    .key--benchmark {{ border-color: var(--benchmark); }}
    .key--field {{ border-color: var(--field); border-top-width: 2px; }}
    .colorkey {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.4rem 1.6rem;
      align-items: baseline;
      margin: 1.8rem 0 0;
      font-size: 0.86rem;
    }}
    .colorkey .label {{ margin: 0; flex-basis: 100%; }}
    .notice {{
      display: grid;
      grid-template-columns: 200px minmax(0, 62ch);
      column-gap: 40px;
      margin: 2.2rem 0 0;
      padding-top: 0.8rem;
      border-top: 1px solid var(--rule);
      font-size: 0.9rem;
      color: #2a3644;
    }}
    .notice .label {{ color: var(--accent-ink); padding-top: 0.2rem; }}
    .contents {{
      display: grid;
      grid-template-columns: 200px minmax(0, 1fr);
      column-gap: 40px;
      margin-top: 1.6rem;
      padding-top: 0.8rem;
      border-top: 1px solid var(--rule);
    }}
    .contents .label {{ margin: 0; padding-top: 0.2rem; }}
    .contents ol {{
      list-style: none;
      margin: 0;
      padding: 0;
      columns: 3 220px;
      column-gap: 32px;
      font-size: 0.9rem;
    }}
    .contents li {{ break-inside: avoid; padding: 0.12rem 0; }}
    .contents a {{ text-decoration: none; }}
    .contents a:hover, .contents a:focus-visible {{
      text-decoration: underline;
      text-decoration-color: var(--accent);
      text-underline-offset: 3px;
    }}
    .contents .num {{
      font-family: var(--mono);
      font-size: 0.74rem;
      color: var(--muted);
      margin-right: 0.9em;
    }}

    .section {{
      display: grid;
      grid-template-columns: 200px minmax(0, 1fr);
      column-gap: 40px;
      border-top: 1px solid var(--ink);
      padding: 22px 0 64px;
    }}
    .section__margin {{ position: sticky; top: 18px; align-self: start; }}
    .section__num {{ display: block; }}
    h2 {{
      font-family: var(--serif);
      font-weight: 600;
      font-size: 1.5rem;
      line-height: 1.14;
      letter-spacing: -0.005em;
      margin: 0.35rem 0 0;
    }}
    .section__body > * {{ margin: 0; }}
    .section__body > :not(:first-child) {{ margin-top: 34px; }}
    .prose {{ max-width: 64ch; }}
    .notes > .label {{ margin: 0.25rem 0 0.7rem; }}
    .notes dl {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      column-gap: 32px;
      margin: 0;
    }}
    .notes dl > div {{ border-top: 1px solid var(--rule); padding-top: 0.6rem; }}
    .notes dd {{ margin: 0.35rem 0 0; font-size: 0.9rem; line-height: 1.5; color: #2a3644; }}
    .statements {{
      list-style: none;
      margin: 0;
      padding: 0;
      max-width: 72ch;
      counter-reset: statement;
    }}
    .statements li {{
      counter-increment: statement;
      display: grid;
      grid-template-columns: 2.6em minmax(0, 1fr);
      padding: 0.55rem 0;
      border-top: 1px solid var(--hair);
      font-size: 0.95rem;
    }}
    .statements li::before {{
      content: counter(statement, decimal-leading-zero);
      font-family: var(--mono);
      font-size: 0.72rem;
      color: var(--muted);
      padding-top: 0.28em;
    }}

    figure {{ margin: 0; }}
    figcaption {{
      display: grid;
      grid-template-columns: 5.2em minmax(0, 70ch);
      font-size: 0.88rem;
      line-height: 1.45;
      color: #2a3644;
    }}
    figcaption .ref {{ color: var(--ink); padding-top: 0.22em; white-space: nowrap; }}
    .plate .plot-scroll {{ border-top: 1px solid var(--rule); overflow-x: auto; }}
    .plate figcaption {{ border-top: 1px solid var(--rule); padding-top: 0.7rem; }}
    .plot--wide {{ min-width: 640px; }}
    .tabular figcaption {{ padding-bottom: 0.6rem; }}

    .table-wrap {{ overflow-x: auto; border-top: 1px solid var(--ink); }}
    .table-wrap--tall {{
      max-height: 27rem;
      overflow-y: auto;
      border-bottom: 1px solid var(--rule);
    }}
    table.ruled {{
      border-collapse: collapse;
      width: 100%;
      font-size: 0.875rem;
      line-height: 1.45;
    }}
    .ruled th, .ruled td {{
      padding: 0.5rem 1.1rem 0.5rem 0;
      text-align: left;
      vertical-align: top;
      border-bottom: 1px solid var(--hair);
    }}
    .ruled th:last-child, .ruled td:last-child {{ padding-right: 0; }}
    .ruled thead th {{
      vertical-align: bottom;
      border-bottom: 1px solid var(--rule);
      background: var(--paper);
      line-height: 1.3;
    }}
    .ruled .num, .ruled .date {{
      font-family: var(--mono);
      font-size: 0.82rem;
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }}
    .ruled td.num, .ruled th.num {{ text-align: right; }}
    .ruled thead th.num {{ white-space: normal; min-width: 5.5em; }}
    .ruled tbody tr:hover td {{ background: rgba(22, 32, 43, 0.04); }}
    .table-wrap--tall thead th {{ position: sticky; top: 0; z-index: 2; }}
    .table-wrap--wide td:first-child, .table-wrap--wide th:first-child {{
      position: sticky;
      left: 0;
      z-index: 1;
      background: var(--paper);
      padding-right: 1.1rem;
      min-width: 11.5em;
      box-shadow: 1px 0 0 var(--rule);
    }}
    .table-wrap--wide thead th:first-child {{ z-index: 3; }}

    .colophon {{
      border-top: 1px solid var(--ink);
      padding-top: 0.9rem;
      font-family: var(--mono);
      font-size: 0.74rem;
      color: var(--muted);
      max-width: 80ch;
    }}
    .masthead--compact {{ padding: 20px 0 18px; }}
    .masthead--compact .kicker {{ margin-bottom: 0.7rem; }}
    .masthead--compact h1 {{ font-size: clamp(1.7rem, 3.4vw, 2.3rem); margin-bottom: 0.5rem; }}
    .masthead--compact .standfirst {{ font-size: 1rem; max-width: 84ch; margin-bottom: 0; }}
    .explorer > * + * {{ margin-top: 34px; }}
    .explorer .notice {{ margin-top: 34px; }}

    @media (max-width: 860px) {{
      .note {{ padding: 0 16px 64px; }}
      .section, .notice, .contents {{ grid-template-columns: minmax(0, 1fr); }}
      .section {{ padding: 18px 0 48px; }}
      .section__margin {{ position: static; margin-bottom: 20px; }}
      .notice .label, .contents .label {{ margin-bottom: 0.4rem; }}
      .spec, .notes dl {{ grid-template-columns: minmax(0, 1fr); }}
      .notes dl > div {{ padding-bottom: 0.7rem; }}
      .headline__figures {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .headline__figures > div {{ padding-bottom: 0.9rem; }}
      .headline__figures > div:nth-child(odd) {{ border-left: 0; padding-left: 0; }}
      .headline__figures > div:nth-child(n + 3) {{ border-top: 1px solid var(--rule); }}
      figcaption {{ grid-template-columns: minmax(0, 1fr); row-gap: 0.2rem; }}
    }}
    @media print {{
      .note {{ max-width: none; padding: 0; }}
      .section__margin {{ position: static; }}
      figure {{ break-inside: avoid; }}
      .table-wrap, .table-wrap--tall {{ overflow: visible; max-height: none; }}
    }}
"""
