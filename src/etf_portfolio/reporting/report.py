"""Generated reporting bundle for ETF portfolio research."""

from __future__ import annotations

import base64
import html
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from plotly.graph_objects import Figure
from plotly.io import to_html, write_image

from etf_portfolio.backtesting.engine import WalkForwardBacktestResult
from etf_portfolio.logging_config import get_logger, log_event
from etf_portfolio.reporting.plots import (
    build_benchmark_comparison_figure,
    build_cumulative_returns_figure,
    build_drawdown_figure,
    build_efficient_frontier_figure,
    build_group_exposure_pie_figure,
    build_rolling_correlation_figure,
    build_rolling_sharpe_figure,
    build_rolling_volatility_figure,
    build_stress_period_figure,
    build_weighted_expense_ratio_over_time_figure,
    build_weights_figure,
)
from etf_portfolio.reporting.tables import (
    build_data_coverage_table,
    build_etf_universe_summary_table,
    build_group_exposure_table,
    build_metrics_table,
    build_missing_data_table,
    build_portfolio_profile_table,
    build_stress_period_table,
    build_weighted_expense_over_time_table,
    build_weights_table,
)
from etf_portfolio.risk.attribution import (
    asset_class_return_attribution,
    asset_class_risk_attribution,
    return_attribution,
    risk_attribution,
)

LOGGER = get_logger(__name__)
PLACEHOLDER_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)
RESEARCH_TOOL_DISCLAIMER = (
    "This report is for research and education only. It is not financial advice, "
    "investment advice, tax advice, legal advice, or a recommendation to buy or sell "
    "any security."
)

SECTION_EXPLANATIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "ETF Universe Summary": (
        (
            "What this shows",
            "The list of ETFs included in this run, with descriptive metadata such as name, "
            "asset class, region, currency, cost, and benchmark index when available.",
        ),
        (
            "How to interpret it",
            "Treat this as the menu the optimizer was allowed to choose from. If an asset "
            "is not in this universe, it cannot appear in the portfolio.",
        ),
        (
            "What to watch",
            "A narrow or biased universe can make results look more confident than they are.",
        ),
    ),
    "Data Coverage Table": (
        (
            "What this shows",
            "The usable price history available for each ETF after the data pipeline has "
            "loaded and aligned provider data.",
        ),
        (
            "How to interpret it",
            "Longer and more complete histories generally make estimates more stable, but "
            "they still only describe the past.",
        ),
        (
            "What to watch",
            "Short histories, late start dates, or missing values can materially affect "
            "backtests and optimization results.",
        ),
    ),
    "Missing Data Table": (
        (
            "What this shows",
            "Counts and patterns of missing price observations for each asset.",
        ),
        (
            "How to interpret it",
            "Missing data is not just a technical issue; it can change return calculations, "
            "risk estimates, and asset eligibility.",
        ),
        (
            "What to watch",
            "Investigate assets with high missing counts before trusting a result that "
            "depends heavily on them.",
        ),
    ),
    "Efficient Frontier Chart": (
        (
            "What this shows",
            "A model-based map of possible risk and return combinations under the configured "
            "optimization assumptions and constraints.",
        ),
        (
            "How to interpret it",
            "Points further up suggest higher estimated return; points further right suggest "
            "higher estimated volatility. These are estimates, not guarantees.",
        ),
        (
            "What to watch",
            "Small input changes can move the frontier. Do not read the exact location of a "
            "point as precision about the future.",
        ),
    ),
    "Latest Realized Portfolio Table": (
        (
            "What this shows",
            "The portfolio weights actually held at the latest rebalance after realized "
            "portfolio mechanics such as contribution-only rebalancing or drift handling.",
        ),
        (
            "How to interpret it",
            "This is the practical portfolio state produced by the backtest, not necessarily "
            "the optimizer's ideal target.",
        ),
        (
            "What to watch",
            "Differences from target weights may reflect rebalancing policy, constraints, "
            "transaction costs, or drift.",
        ),
    ),
    "Optimizer Target Portfolio Table": (
        (
            "What this shows",
            "The optimizer's desired weights at the latest rebalance before realized "
            "portfolio mechanics are applied.",
        ),
        (
            "How to interpret it",
            "Use this to understand what the model wanted under its objective and constraints.",
        ),
        (
            "What to watch",
            "A target can look attractive in theory while being hard to reach in practice.",
        ),
    ),
    "Portfolio Weights": (
        (
            "What this shows",
            "How portfolio allocations changed across rebalance dates.",
        ),
        (
            "How to interpret it",
            "Stable weights suggest a more consistent allocation; large swings suggest the "
            "model is reacting strongly to new data or unstable estimates.",
        ),
        (
            "What to watch",
            "Frequent large changes can imply higher turnover, implementation complexity, "
            "and model sensitivity.",
        ),
    ),
    "Exposure": (
        (
            "What this shows",
            "Portfolio weights grouped by higher-level categories such as asset class or region.",
        ),
        (
            "How to interpret it",
            "Use this to see the broad economic bets behind the ticker-level allocations.",
        ),
        (
            "What to watch",
            "A portfolio can look diversified by ticker while still being concentrated in one "
            "asset class, region, or risk factor.",
        ),
    ),
    "Benchmark Comparison": (
        (
            "What this shows",
            "Portfolio results compared with one or more configured benchmarks.",
        ),
        (
            "How to interpret it",
            "The benchmark is the yardstick. Outperformance matters only relative to the risk, "
            "drawdowns, and assumptions required to achieve it.",
        ),
        (
            "What to watch",
            "A poor benchmark choice can make a strategy look better or worse than it really is.",
        ),
    ),
    "Backtest Performance": (
        (
            "What this shows",
            "Historical simulated performance for the portfolio over the tested period.",
        ),
        (
            "How to interpret it",
            "Read this as a historical experiment: what would have happened under these exact "
            "rules, data, costs, and constraints.",
        ),
        (
            "What to watch",
            "Past performance does not guarantee future results. Strong backtests can still fail "
            "out of sample.",
        ),
    ),
    "Drawdown Chart": (
        (
            "What this shows",
            "The depth and duration of historical losses from prior portfolio highs.",
        ),
        (
            "How to interpret it",
            "Drawdown helps translate risk into a lived experience: how much the portfolio fell "
            "before recovering.",
        ),
        (
            "What to watch",
            "A strategy with attractive returns but unacceptable drawdowns may be unsuitable for "
            "many real investors.",
        ),
    ),
    "Rolling Risk Metrics": (
        (
            "What this shows",
            "How volatility, risk-adjusted returns, and benchmark relationship changed over time.",
        ),
        (
            "How to interpret it",
            "Stable rolling metrics suggest more consistent behavior; unstable metrics suggest "
            "the portfolio behaves differently across market regimes.",
        ),
        (
            "What to watch",
            "Rolling windows are backward-looking and can hide sudden breaks at the edges.",
        ),
    ),
    "Stress Periods": (
        (
            "What this shows",
            "Portfolio behavior during difficult historical windows.",
        ),
        (
            "How to interpret it",
            "Use this to understand how the strategy handled adverse environments, not just "
            "average conditions.",
        ),
        (
            "What to watch",
            "Stress tests are limited to periods represented in the available data.",
        ),
    ),
    "Weighted Expense Ratio Over Time": (
        (
            "What this shows",
            "The portfolio's weighted average ETF expense ratio across time.",
        ),
        (
            "How to interpret it",
            "Lower costs leave more gross return for the investor, but cost is only one part "
            "of portfolio quality.",
        ),
        (
            "What to watch",
            "Expense ratios exclude taxes, spreads, slippage, advisory fees, and "
            "account-level costs.",
        ),
    ),
    "Return Attribution": (
        (
            "What this shows",
            "A breakdown of which assets or groups contributed to historical portfolio returns.",
        ),
        (
            "How to interpret it",
            "Positive contributors helped the backtest; negative contributors reduced it.",
        ),
        (
            "What to watch",
            "Attribution explains the past. It does not prove the same assets will drive "
            "future returns.",
        ),
    ),
    "Risk Attribution": (
        (
            "What this shows",
            "A breakdown of which assets or groups contributed to estimated portfolio risk.",
        ),
        (
            "How to interpret it",
            "Large risk contributors are the exposures most responsible for portfolio volatility.",
        ),
        (
            "What to watch",
            "Risk contribution depends on the covariance estimate, which can change across "
            "regimes.",
        ),
    ),
    "Realized Constraint Warnings": (
        (
            "What this shows",
            "Cases where realized holdings moved outside configured bounds after applying the "
            "chosen rebalance policy.",
        ),
        (
            "How to interpret it",
            "Warnings do not necessarily mean the run failed; they show where practical mechanics "
            "diverged from ideal constraints.",
        ),
        (
            "What to watch",
            "Repeated or large breaches may mean the rebalance policy is too loose for the "
            "intended use.",
        ),
    ),
    "Assumptions and Limitations": (
        (
            "What this shows",
            "The configuration, modeling assumptions, known limitations, and caveats attached "
            "to this report.",
        ),
        (
            "How to interpret it",
            "This section defines the boundary of the result. The report should not be read "
            "without it.",
        ),
        (
            "What to watch",
            "If an assumption is unrealistic for your use case, the conclusions may not transfer.",
        ),
    ),
}


@dataclass(frozen=True)
class ReportArtifacts:
    """Paths for generated reporting artifacts."""

    html_path: Path
    workbook_path: Path
    figure_paths: dict[str, Path]


def generate_html_report(
    backtest_result: WalkForwardBacktestResult,
    *,
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    output_path: str | Path,
    benchmark_returns: pd.Series | pd.DataFrame | None = None,
    periods_per_year: int = 12,
    risk_free_rate: float = 0.0,
    max_weight: float = 1.0,
    title: str = "ETF Portfolio Research Report",
    benchmark_suite: dict[str, pd.Series] | None = None,
    metadata: pd.DataFrame | None = None,
    primary_benchmark_returns: pd.Series | None = None,
    rolling_window: int = 63,
    prices: pd.DataFrame | None = None,
    assumptions: dict[str, Any] | None = None,
    limitations: list[str] | None = None,
    asset_returns: pd.DataFrame | None = None,
    asset_classes: pd.Series | None = None,
    asset_class_bounds: dict[str, tuple[float, float]] | None = None,
    ticker_bounds: dict[str, tuple[float | None, float | None]] | None = None,
    bond_assets: list[str] | None = None,
    min_bond_exposure: float | None = None,
    expense_ratios: pd.Series | None = None,
) -> Path:
    """Generate the consolidated HTML report."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    sections = _build_report_sections(
        backtest_result,
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        benchmark_returns=benchmark_returns,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
        max_weight=max_weight,
        benchmark_suite=benchmark_suite,
        metadata=metadata,
        primary_benchmark_returns=primary_benchmark_returns,
        rolling_window=rolling_window,
        prices=prices,
        assumptions=assumptions,
        limitations=limitations,
        asset_returns=asset_returns,
        asset_classes=asset_classes,
        asset_class_bounds=asset_class_bounds,
        ticker_bounds=ticker_bounds,
        bond_assets=bond_assets,
        min_bond_exposure=min_bond_exposure,
        expense_ratios=expense_ratios,
    )

    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f4f1ea;
      --paper: #fffdf8;
      --ink: #182029;
      --muted: #5e6b75;
      --accent: #9f4f2a;
      --border: #d8cfc2;
    }}
    body {{
      font-family: "Georgia", "Iowan Old Style", serif;
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, rgba(159, 79, 42, 0.12), transparent 28rem),
        linear-gradient(180deg, #f2ede3 0%, var(--bg) 60%, #ece6db 100%);
    }}
    main {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 2rem 1rem 4rem;
    }}
    header {{
      margin-bottom: 1.5rem;
      padding: 2rem 0 1rem;
    }}
    h1, h2, h3 {{
      color: #101820;
      margin-top: 0;
    }}
    h1 {{
      font-size: 2.4rem;
      margin-bottom: 0.25rem;
    }}
    p.lede {{
      color: var(--muted);
      font-size: 1.05rem;
      max-width: 48rem;
    }}
    .disclaimer {{
      background: #fff4df;
      border: 1px solid #e6b979;
      border-radius: 14px;
      color: #5b3515;
      font-family: "Avenir Next", "Helvetica Neue", sans-serif;
      font-size: 0.95rem;
      margin-top: 1rem;
      max-width: 56rem;
      padding: 0.85rem 1rem;
    }}
    .card {{
      background: rgba(255, 253, 248, 0.94);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: 0 16px 40px rgba(24, 32, 41, 0.08);
      margin-bottom: 1.25rem;
      overflow: hidden;
    }}
    .card-inner {{
      padding: 1.15rem 1.25rem;
    }}
    .grid {{
      display: grid;
      gap: 1.25rem;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    }}
    .section-explanation {{
      background: #f8f2e9;
      border-left: 4px solid var(--accent);
      border-radius: 12px;
      font-family: "Avenir Next", "Helvetica Neue", sans-serif;
      margin: 0.75rem 0 1rem;
      padding: 0.85rem 1rem;
    }}
    .section-explanation p {{
      margin: 0.25rem 0;
    }}
    .section-explanation .eyebrow {{
      color: var(--accent);
      font-size: 0.78rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .reader-guide {{
      font-family: "Avenir Next", "Helvetica Neue", sans-serif;
    }}
    .reader-guide p {{
      margin-top: 0;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      font-family: "Avenir Next", "Helvetica Neue", sans-serif;
      font-size: 0.92rem;
    }}
    th, td {{
      border-bottom: 1px solid #e7ddd1;
      padding: 0.65rem 0.75rem;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #f7f1e8;
    }}
    ul {{
      margin: 0;
      padding-left: 1.2rem;
    }}
    .figure-block > div {{
      min-height: 360px;
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>{html.escape(title)}</h1>
      <p class="lede">
        Code-generated research report built from pipeline outputs. Use it to understand
        model behavior, assumptions, risk, and historical simulations; do not treat it as
        a recommendation.
      </p>
      <div class="disclaimer">{html.escape(RESEARCH_TOOL_DISCLAIMER)}</div>
    </header>
    {"".join(_render_section(title, content) for title, content in sections)}
  </main>
</body>
</html>
"""

    output.write_text(report_html, encoding="utf-8")
    return output


def generate_report_bundle(
    backtest_result: WalkForwardBacktestResult,
    *,
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    html_output_path: str | Path,
    workbook_output_path: str | Path,
    figures_output_dir: str | Path,
    benchmark_returns: pd.Series | pd.DataFrame | None = None,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
    max_weight: float = 1.0,
    title: str = "ETF Portfolio Research Report",
    benchmark_suite: dict[str, pd.Series] | None = None,
    metadata: pd.DataFrame | None = None,
    primary_benchmark_returns: pd.Series | None = None,
    rolling_window: int = 63,
    prices: pd.DataFrame | None = None,
    assumptions: dict[str, Any] | None = None,
    limitations: list[str] | None = None,
    asset_returns: pd.DataFrame | None = None,
    asset_classes: pd.Series | None = None,
    asset_class_bounds: dict[str, tuple[float, float]] | None = None,
    ticker_bounds: dict[str, tuple[float | None, float | None]] | None = None,
    bond_assets: list[str] | None = None,
    min_bond_exposure: float | None = None,
    expense_ratios: pd.Series | None = None,
) -> ReportArtifacts:
    """Generate the HTML report, Excel workbook, and exported figures."""

    html_path = generate_html_report(
        backtest_result,
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        output_path=html_output_path,
        benchmark_returns=benchmark_returns,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
        max_weight=max_weight,
        title=title,
        benchmark_suite=benchmark_suite,
        metadata=metadata,
        primary_benchmark_returns=primary_benchmark_returns,
        rolling_window=rolling_window,
        prices=prices,
        assumptions=assumptions,
        limitations=limitations,
        asset_returns=asset_returns,
        asset_classes=asset_classes,
        asset_class_bounds=asset_class_bounds,
        ticker_bounds=ticker_bounds,
        bond_assets=bond_assets,
        min_bond_exposure=min_bond_exposure,
        expense_ratios=expense_ratios,
    )

    workbook_path = _write_report_workbook(
        workbook_output_path,
        backtest_result=backtest_result,
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        benchmark_returns=benchmark_returns,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
        benchmark_suite=benchmark_suite,
        metadata=metadata,
        primary_benchmark_returns=primary_benchmark_returns,
        prices=prices,
        assumptions=assumptions,
        limitations=limitations,
        asset_returns=asset_returns,
    )
    figure_paths = _write_figure_exports(
        _build_figures(
            backtest_result,
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            benchmark_returns=benchmark_returns,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
            max_weight=max_weight,
            metadata=metadata,
            primary_benchmark_returns=primary_benchmark_returns,
            rolling_window=rolling_window,
            asset_classes=asset_classes,
            asset_class_bounds=asset_class_bounds,
            ticker_bounds=ticker_bounds,
            bond_assets=bond_assets,
            min_bond_exposure=min_bond_exposure,
            expense_ratios=expense_ratios,
        ),
        output_dir=figures_output_dir,
    )

    return ReportArtifacts(
        html_path=html_path,
        workbook_path=workbook_path,
        figure_paths=figure_paths,
    )


def _build_report_sections(
    backtest_result: WalkForwardBacktestResult,
    *,
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    benchmark_returns: pd.Series | pd.DataFrame | None,
    periods_per_year: int,
    risk_free_rate: float,
    max_weight: float,
    benchmark_suite: dict[str, pd.Series] | None,
    metadata: pd.DataFrame | None,
    primary_benchmark_returns: pd.Series | None,
    rolling_window: int,
    prices: pd.DataFrame | None,
    assumptions: dict[str, Any] | None,
    limitations: list[str] | None,
    asset_returns: pd.DataFrame | None,
    asset_classes: pd.Series | None,
    asset_class_bounds: dict[str, tuple[float, float]] | None,
    ticker_bounds: dict[str, tuple[float | None, float | None]] | None,
    bond_assets: list[str] | None,
    min_bond_exposure: float | None,
    expense_ratios: pd.Series | None,
) -> list[tuple[str, str]]:
    metrics_table = build_metrics_table(
        backtest_result.portfolio_returns,
        weights=backtest_result.weights,
        periods_per_year=periods_per_year,
        benchmark_returns=(
            primary_benchmark_returns
            if primary_benchmark_returns is not None
            else benchmark_returns
            if isinstance(benchmark_returns, pd.Series)
            else None
        ),
        benchmark_suite=benchmark_suite,
        risk_free_rate=risk_free_rate,
    )
    latest_realized_weights = build_weights_table(backtest_result.applied_weights)
    optimizer_target_weights = build_weights_table(backtest_result.target_weights)
    stress_table = build_stress_period_table(
        backtest_result.portfolio_returns,
        benchmark_returns=benchmark_returns,
    )
    figures = _build_figures(
        backtest_result,
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        benchmark_returns=benchmark_returns,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
        max_weight=max_weight,
        metadata=metadata,
        primary_benchmark_returns=primary_benchmark_returns,
        rolling_window=rolling_window,
        asset_classes=asset_classes,
        asset_class_bounds=asset_class_bounds,
        ticker_bounds=ticker_bounds,
        bond_assets=bond_assets,
        min_bond_exposure=min_bond_exposure,
        expense_ratios=expense_ratios,
    )

    sections: list[tuple[str, str]] = [
        (
            "Reader Guide",
            _build_reader_guide_section(
                backtest_result,
                benchmark_returns=benchmark_returns,
                primary_benchmark_returns=primary_benchmark_returns,
                metadata=metadata,
                prices=prices,
            ),
        )
    ]
    if metadata is not None:
        sections.append(
            (
                "ETF Universe Summary",
                _table_html(
                    build_etf_universe_summary_table(
                        metadata,
                        tickers=backtest_result.weights.columns.tolist(),
                    )
                ),
            )
        )
    if prices is not None:
        coverage_table = build_data_coverage_table(prices, metadata=metadata)
        missing_table = build_missing_data_table(prices)
        sections.extend(
            [
                ("Data Coverage Table", _table_html(coverage_table)),
                ("Missing Data Table", _table_html(missing_table)),
            ]
        )
    sections.extend(
        [
            ("Efficient Frontier Chart", _figure_html(figures["efficient_frontier"])),
            ("Latest Realized Portfolio Table", _table_html(latest_realized_weights)),
            ("Optimizer Target Portfolio Table", _table_html(optimizer_target_weights)),
            (
                "Portfolio Weights",
                _figure_html(figures["portfolio_weights"]),
            ),
        ]
    )
    if metadata is not None:
        exposure_blocks = [
            _table_html(
                build_group_exposure_table(
                    backtest_result.weights,
                    metadata,
                    field="asset_class",
                    label="Asset Class",
                )
            ),
            _figure_html(figures["asset_class_exposure"]),
        ]
        if "region" in metadata.columns:
            exposure_blocks.extend(
                [
                    _table_html(
                        build_group_exposure_table(
                            backtest_result.weights,
                            metadata,
                            field="region",
                            label="Region",
                        )
                    ),
                    _figure_html(figures["region_exposure"]),
                ]
            )
        sections.append(("Exposure", _combine_blocks(exposure_blocks)))
    sections.append(
        (
            "Benchmark Comparison",
            _combine_blocks(
                [
                    _table_html(metrics_table),
                    _figure_html(figures["benchmark_comparison"]),
                ]
            ),
        )
    )
    sections.extend(
        [
            (
                "Backtest Performance",
                _combine_blocks(
                    [
                        _table_html(metrics_table),
                        _figure_html(figures["cumulative_returns"]),
                    ]
                ),
            ),
            ("Drawdown Chart", _figure_html(figures["drawdown"])),
            (
                "Rolling Risk Metrics",
                _combine_blocks(
                    [
                        _figure_html(figures["rolling_volatility"]),
                        _figure_html(figures["rolling_sharpe"]),
                        _figure_html(figures["rolling_correlation"]),
                    ]
                ),
            ),
            (
                "Stress Periods",
                _combine_blocks(
                    [
                        _table_html(stress_table),
                        _figure_html(figures["stress_periods"]),
                    ]
                ),
            ),
        ]
    )

    if metadata is not None and "expense_ratio" in metadata.columns:
        sections.append(
            (
                "Weighted Expense Ratio Over Time",
                _combine_blocks(
                    [
                        _table_html(
                            build_weighted_expense_over_time_table(
                                backtest_result.weights,
                                metadata,
                            )
                        ),
                        _figure_html(figures["weighted_expense_over_time"]),
                    ]
                ),
            )
        )

    return_attr_table = _build_return_attribution_table(
        backtest_result.weights,
        asset_returns=asset_returns,
        metadata=metadata,
    )
    if return_attr_table is not None:
        sections.append(("Return Attribution", _table_html(return_attr_table)))

    risk_attr_table = _build_risk_attribution_table(
        backtest_result.weights,
        covariance_matrix=covariance_matrix,
        metadata=metadata,
    )
    if risk_attr_table is not None:
        sections.append(("Risk Attribution", _table_html(risk_attr_table)))

    realized_constraint_warnings = _build_realized_constraint_warning_table(backtest_result)
    if realized_constraint_warnings is not None:
        sections.append(
            (
                "Realized Constraint Warnings",
                _table_html(realized_constraint_warnings),
            )
        )

    sections.append(
        (
            "Assumptions and Limitations",
            _build_assumptions_appendix(
                backtest_result,
                metadata=metadata,
                primary_benchmark_returns=primary_benchmark_returns,
                periods_per_year=periods_per_year,
                risk_free_rate=risk_free_rate,
                max_weight=max_weight,
                assumptions=assumptions,
                limitations=limitations,
            ),
        )
    )
    return sections


def _build_reader_guide_section(
    backtest_result: WalkForwardBacktestResult,
    *,
    benchmark_returns: pd.Series | pd.DataFrame | None,
    primary_benchmark_returns: pd.Series | None,
    metadata: pd.DataFrame | None,
    prices: pd.DataFrame | None,
) -> str:
    """Build a plain-language opening section for non-specialist readers."""

    latest_rebalance = (
        str(backtest_result.weights.index.max().date())
        if not backtest_result.weights.empty
        else "No rebalance dates available"
    )
    benchmark_status = (
        "Primary benchmark provided"
        if primary_benchmark_returns is not None
        else "Benchmark data provided"
        if benchmark_returns is not None
        else "No benchmark data provided"
    )
    metadata_status = (
        "ETF metadata included" if metadata is not None else "ETF metadata not provided"
    )
    price_coverage_status = (
        "Price coverage included" if prices is not None else "Price coverage not provided"
    )
    guide_rows = pd.DataFrame(
        [
            {
                "topic": "Purpose",
                "guidance": (
                    "Research how a rules-based ETF portfolio behaved historically under the "
                    "configured data, optimizer, constraints, and rebalance policy."
                ),
            },
            {
                "topic": "Research-tool warning",
                "guidance": RESEARCH_TOOL_DISCLAIMER,
            },
            {
                "topic": "Latest rebalance",
                "guidance": latest_rebalance,
            },
            {
                "topic": "ETF universe size",
                "guidance": str(len(backtest_result.weights.columns)),
            },
            {
                "topic": "Benchmark status",
                "guidance": benchmark_status,
            },
            {
                "topic": "Metadata status",
                "guidance": metadata_status,
            },
            {
                "topic": "Price coverage status",
                "guidance": price_coverage_status,
            },
        ]
    )
    reading_order = [
        "Start with Assumptions and Limitations so the boundaries of the report are clear.",
        "Check ETF Universe Summary and Data Coverage before interpreting performance.",
        "Compare Latest Realized Portfolio against Optimizer Target Portfolio to separate "
        "practical holdings from ideal model weights.",
        "Use Benchmark Comparison, Drawdown Chart, Rolling Risk Metrics, and Stress Periods "
        "together; no single chart is enough.",
        "Read section explanations as guardrails for what each table or chart can and "
        "cannot tell you.",
    ]
    return (
        '<div class="reader-guide">'
        "<p>"
        "This opening guide is meant to make the report readable before you inspect the "
        "tables and charts. The results are conditional on the configured inputs and "
        "should be interpreted as a historical research experiment."
        "</p>"
        f"{_combine_blocks([_table_html(guide_rows), _bullet_list_html(reading_order)])}"
        "</div>"
    )


def _build_figures(
    backtest_result: WalkForwardBacktestResult,
    *,
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    benchmark_returns: pd.Series | pd.DataFrame | None,
    periods_per_year: int,
    risk_free_rate: float,
    max_weight: float,
    metadata: pd.DataFrame | None,
    primary_benchmark_returns: pd.Series | None,
    rolling_window: int,
    asset_classes: pd.Series | None = None,
    asset_class_bounds: dict[str, tuple[float, float]] | None = None,
    ticker_bounds: dict[str, tuple[float | None, float | None]] | None = None,
    bond_assets: list[str] | None = None,
    min_bond_exposure: float | None = None,
    expense_ratios: pd.Series | None = None,
) -> dict[str, Figure]:
    latest_weights = backtest_result.weights.iloc[-1]
    figures: dict[str, Figure] = {
        "efficient_frontier": build_efficient_frontier_figure(
            expected_returns,
            covariance_matrix,
            portfolio_weights=latest_weights,
            max_weight=max_weight,
            risk_free_rate=risk_free_rate,
            asset_classes=asset_classes,
            asset_class_bounds=asset_class_bounds,
            ticker_bounds=ticker_bounds,
            bond_assets=bond_assets,
            min_bond_exposure=min_bond_exposure,
            expense_ratios=expense_ratios,
        ),
        "cumulative_returns": build_cumulative_returns_figure(
            backtest_result.portfolio_returns,
            benchmark_returns=benchmark_returns,
        ),
        "drawdown": build_drawdown_figure(
            backtest_result.portfolio_returns,
            benchmark_returns=benchmark_returns,
        ),
        "portfolio_weights": build_weights_figure(backtest_result.weights),
        "rolling_volatility": build_rolling_volatility_figure(
            backtest_result.portfolio_returns,
            window=rolling_window,
            periods_per_year=periods_per_year,
        ),
        "rolling_sharpe": build_rolling_sharpe_figure(
            backtest_result.portfolio_returns,
            window=rolling_window,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
        ),
    }
    if benchmark_returns is not None:
        figures["benchmark_comparison"] = build_benchmark_comparison_figure(
            backtest_result.portfolio_returns,
            benchmark_returns,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
        )
    else:
        figures["benchmark_comparison"] = build_cumulative_returns_figure(
            backtest_result.portfolio_returns,
        )
        figures["benchmark_comparison"].update_layout(title="Benchmark Comparison Unavailable")
    if primary_benchmark_returns is not None:
        figures["rolling_correlation"] = build_rolling_correlation_figure(
            backtest_result.portfolio_returns,
            primary_benchmark_returns,
            window=rolling_window,
        )
    else:
        figures["rolling_correlation"] = build_rolling_volatility_figure(
            backtest_result.portfolio_returns,
            window=rolling_window,
            periods_per_year=periods_per_year,
        )
        figures["rolling_correlation"].update_layout(
            title=f"Rolling Correlation To Benchmark ({rolling_window} periods) Unavailable"
        )
    if metadata is not None:
        metadata_by_ticker = metadata.copy()
        if "ticker" in metadata_by_ticker.columns:
            metadata_by_ticker = metadata_by_ticker.set_index("ticker")
        aligned_metadata = metadata_by_ticker.reindex(backtest_result.weights.columns)
        figures["asset_class_exposure"] = build_group_exposure_pie_figure(
            backtest_result.weights,
            aligned_metadata["asset_class"],
            title="Portfolio Weights By Asset Class",
        )
        if "region" in aligned_metadata.columns:
            figures["region_exposure"] = build_group_exposure_pie_figure(
                backtest_result.weights,
                aligned_metadata["region"],
                title="Portfolio Weights By Region",
            )
        if "expense_ratio" in aligned_metadata.columns:
            figures["weighted_expense_over_time"] = build_weighted_expense_ratio_over_time_figure(
                backtest_result.weights,
                aligned_metadata["expense_ratio"],
            )
    else:
        figures["asset_class_exposure"] = build_weights_figure(backtest_result.weights)
        figures["asset_class_exposure"].update_layout(title="Asset-Class Exposure Unavailable")
    figures["stress_periods"] = build_stress_period_figure(
        backtest_result.portfolio_returns,
        benchmark_returns=benchmark_returns,
    )
    return figures


def _write_report_workbook(
    output_path: str | Path,
    *,
    backtest_result: WalkForwardBacktestResult,
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    benchmark_returns: pd.Series | pd.DataFrame | None,
    periods_per_year: int,
    risk_free_rate: float,
    benchmark_suite: dict[str, pd.Series] | None,
    metadata: pd.DataFrame | None,
    primary_benchmark_returns: pd.Series | None,
    prices: pd.DataFrame | None,
    assumptions: dict[str, Any] | None,
    limitations: list[str] | None,
    asset_returns: pd.DataFrame | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    workbook_tables: dict[str, pd.DataFrame] = {
        "metrics": build_metrics_table(
            backtest_result.portfolio_returns,
            weights=backtest_result.weights,
            periods_per_year=periods_per_year,
            benchmark_returns=(
                primary_benchmark_returns
                if primary_benchmark_returns is not None
                else benchmark_returns
                if isinstance(benchmark_returns, pd.Series)
                else None
            ),
            benchmark_suite=benchmark_suite,
            risk_free_rate=risk_free_rate,
        ),
        "latest_realized_portfolio": build_weights_table(backtest_result.applied_weights),
        "optimizer_target_portfolio": build_weights_table(backtest_result.target_weights),
        "weights_history": backtest_result.weights.reset_index(),
        "optimizer_target_history": backtest_result.target_weights.reset_index(),
        "rebalance_summary": backtest_result.rebalance_summary.reset_index(),
        "expected_returns": expected_returns.rename("expected_return").reset_index(),
        "covariance": covariance_matrix.reset_index(),
        "stress_periods": build_stress_period_table(
            backtest_result.portfolio_returns,
            benchmark_returns=benchmark_returns,
        ),
    }

    if metadata is not None:
        workbook_tables["etf_universe"] = build_etf_universe_summary_table(
            metadata,
            tickers=backtest_result.weights.columns.tolist(),
        )
        workbook_tables["asset_class_exposure"] = build_group_exposure_table(
            backtest_result.weights,
            metadata,
            field="asset_class",
            label="Asset Class",
        )
        if "region" in metadata.columns:
            workbook_tables["region_exposure"] = build_group_exposure_table(
                backtest_result.weights,
                metadata,
                field="region",
                label="Region",
            )
        if "expense_ratio" in metadata.columns:
            workbook_tables["weighted_expense_over_time"] = build_weighted_expense_over_time_table(
                backtest_result.weights,
                metadata,
            )
        workbook_tables["portfolio_profile"] = build_portfolio_profile_table(
            backtest_result.portfolio_returns,
            weights=backtest_result.weights,
            metadata=metadata,
            benchmark_returns=primary_benchmark_returns,
        )
    if prices is not None:
        workbook_tables["data_coverage"] = build_data_coverage_table(prices, metadata=metadata)
        workbook_tables["missing_data"] = build_missing_data_table(prices)

    return_attr_table = _build_return_attribution_table(
        backtest_result.weights,
        asset_returns=asset_returns,
        metadata=metadata,
    )
    if return_attr_table is not None:
        workbook_tables["return_attribution"] = return_attr_table

    risk_attr_table = _build_risk_attribution_table(
        backtest_result.weights,
        covariance_matrix=covariance_matrix,
        metadata=metadata,
    )
    if risk_attr_table is not None:
        workbook_tables["risk_attribution"] = risk_attr_table
    realized_constraint_warnings = _build_realized_constraint_warning_table(backtest_result)
    if realized_constraint_warnings is not None:
        workbook_tables["realized_constraint_warnings"] = realized_constraint_warnings

    workbook_tables["assumptions"] = _assumptions_appendix_table(
        backtest_result,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
        assumptions=assumptions,
        limitations=limitations,
    )

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, table in workbook_tables.items():
            sanitized_name = sheet_name[:31]
            table.to_excel(writer, sheet_name=sanitized_name, index=False)

    return output


def _write_figure_exports(
    figures: dict[str, Figure],
    *,
    output_dir: str | Path,
) -> dict[str, Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for name, figure in figures.items():
        path = output_root / f"{name}.png"
        try:
            write_image(figure, str(path), format="png", width=1400, height=840, scale=2)
        except Exception as exc:
            path.write_bytes(PLACEHOLDER_PNG)
            log_event(
                LOGGER,
                logging.WARNING,
                "report_figure_export_fallback",
                figure_name=name,
                output_path=str(path),
                reason=str(exc),
            )
        paths[name] = path

    return paths


def _table_html(table: pd.DataFrame) -> str:
    return table.to_html(index=False, border=0)


def _figure_html(figure: Figure) -> str:
    return (
        '<div class="figure-block">'
        f"{to_html(figure, full_html=False, include_plotlyjs='inline')}"
        "</div>"
    )


def _combine_blocks(blocks: list[str]) -> str:
    cards = "".join(
        f'<div class="card"><div class="card-inner">{block}</div></div>'
        for block in blocks
        if block
    )
    return f'<div class="grid">{cards}</div>'


def _bullet_list_html(items: Any) -> str:
    lines = "".join(f"<li>{html.escape(str(item))}</li>" for item in items if str(item).strip())
    return f"<ul>{lines}</ul>"


def _section_explanation_html(section_title: str) -> str:
    explanations = SECTION_EXPLANATIONS.get(section_title)
    if explanations is None:
        return ""
    explanation_rows = "".join(
        f"<p><strong>{html.escape(label)}:</strong> {html.escape(description)}</p>"
        for label, description in explanations
    )
    return (
        '<div class="section-explanation">'
        '<p class="eyebrow">How to read this section</p>'
        f"{explanation_rows}"
        "</div>"
    )


def _render_section(section_title: str, content: str) -> str:
    return (
        '<section class="card">'
        f'<div class="card-inner"><h2>{html.escape(section_title)}</h2>'
        f"{_section_explanation_html(section_title)}{content}</div>"
        "</section>"
    )


_DEFAULT_ASSUMPTIONS: tuple[tuple[str, str], ...] = (
    (
        "scope",
        "Research-only system for long-horizon retirement-style accumulation. "
        "Not live trading, not personalized financial advice.",
    ),
    (
        "currency",
        "Base currency is USD. No FX hedging applied to non-USD exposures.",
    ),
    (
        "data",
        "Adjusted prices from the configured provider; survivorship bias and "
        "data revisions are not corrected for.",
    ),
    (
        "no_lookahead",
        "Walk-forward backtest only uses returns observed strictly before each rebalance date.",
    ),
    (
        "costs",
        "Transaction costs are modeled via the rebalance turnover times the "
        "configured cost basis points; bid/ask spreads, taxes, and slippage are not modeled.",
    ),
    (
        "rebalancing",
        "Default rebalance mode is contribution-only with optional band-edge selling "
        "above the configured drift threshold.",
    ),
    (
        "optimization",
        "Mean-variance style optimization with documented constraints; expected returns "
        "use historical means unless overridden.",
    ),
)


def _build_assumptions_appendix(
    backtest_result: WalkForwardBacktestResult,
    *,
    metadata: pd.DataFrame | None,
    primary_benchmark_returns: pd.Series | None,
    periods_per_year: int,
    risk_free_rate: float,
    max_weight: float,
    assumptions: dict[str, Any] | None,
    limitations: list[str] | None,
) -> str:
    """Build the mandatory assumptions, parameters, and limitations appendix."""

    blocks: list[str] = []

    config_rows = [
        {"parameter": "periods_per_year", "value": str(periods_per_year)},
        {"parameter": "risk_free_rate", "value": f"{risk_free_rate:.6f}"},
        {"parameter": "max_weight", "value": f"{max_weight:.4f}"},
        {"parameter": "rebalance_count", "value": str(len(backtest_result.weights.index))},
    ]
    if not backtest_result.weights.empty:
        config_rows.extend(
            [
                {
                    "parameter": "first_rebalance_date",
                    "value": str(backtest_result.weights.index.min().date()),
                },
                {
                    "parameter": "last_rebalance_date",
                    "value": str(backtest_result.weights.index.max().date()),
                },
            ]
        )
    blocks.append(_table_html(pd.DataFrame(config_rows)))

    merged_assumptions: dict[str, str] = {key: value for key, value in _DEFAULT_ASSUMPTIONS}
    for key, value in (assumptions or {}).items():
        merged_assumptions[str(key)] = str(value)
    assumption_table = pd.DataFrame(
        [{"key": key, "assumption": value} for key, value in merged_assumptions.items()]
    )
    blocks.append(_table_html(assumption_table))

    if metadata is not None:
        profile = build_portfolio_profile_table(
            backtest_result.portfolio_returns,
            weights=backtest_result.weights,
            metadata=metadata,
            benchmark_returns=primary_benchmark_returns,
        )
        blocks.append(_table_html(profile))

    standard_limitations = [
        RESEARCH_TOOL_DISCLAIMER,
        "Backtest results assume frictionless execution at end-of-period prices.",
        "Past performance does not guarantee future results.",
        "Tax treatment, account-specific constraints, and broker fees are not modeled.",
        "Provider data may be subject to revisions or vendor outages.",
    ]
    combined_limitations = list(standard_limitations) + list(limitations or [])
    realized_warning = _realized_constraint_warning_message(backtest_result)
    if realized_warning is not None:
        combined_limitations.append(realized_warning)
    blocks.append(_bullet_list_html(combined_limitations))

    return _combine_blocks(blocks)


def _assumptions_appendix_table(
    backtest_result: WalkForwardBacktestResult,
    *,
    periods_per_year: int,
    risk_free_rate: float,
    assumptions: dict[str, Any] | None,
    limitations: list[str] | None,
) -> pd.DataFrame:
    """Flat tabular representation of the appendix, suitable for the Excel workbook."""

    rows: list[dict[str, Any]] = [
        {"section": "config", "key": "periods_per_year", "value": str(periods_per_year)},
        {"section": "config", "key": "risk_free_rate", "value": f"{risk_free_rate:.6f}"},
        {
            "section": "config",
            "key": "rebalance_count",
            "value": str(len(backtest_result.weights.index)),
        },
    ]
    if not backtest_result.weights.empty:
        rows.extend(
            [
                {
                    "section": "config",
                    "key": "first_rebalance_date",
                    "value": str(backtest_result.weights.index.min().date()),
                },
                {
                    "section": "config",
                    "key": "last_rebalance_date",
                    "value": str(backtest_result.weights.index.max().date()),
                },
            ]
        )
    merged_assumptions: dict[str, str] = {key: value for key, value in _DEFAULT_ASSUMPTIONS}
    for key, value in (assumptions or {}).items():
        merged_assumptions[str(key)] = str(value)
    rows.extend(
        {"section": "assumption", "key": key, "value": value}
        for key, value in merged_assumptions.items()
    )
    standard_limitations = [
        RESEARCH_TOOL_DISCLAIMER,
        "Backtest results assume frictionless execution at end-of-period prices.",
        "Past performance does not guarantee future results.",
        "Tax treatment, account-specific constraints, and broker fees are not modeled.",
        "Provider data may be subject to revisions or vendor outages.",
    ]
    rows.extend(
        {"section": "limitation", "key": f"limitation_{idx + 1}", "value": value}
        for idx, value in enumerate(
            _combined_limitations(
                backtest_result,
                limitations=limitations,
                standard_limitations=standard_limitations,
            )
        )
    )
    return pd.DataFrame(rows)


def _combined_limitations(
    backtest_result: WalkForwardBacktestResult,
    *,
    limitations: list[str] | None,
    standard_limitations: list[str],
) -> list[str]:
    combined_limitations = list(standard_limitations) + list(limitations or [])
    realized_warning = _realized_constraint_warning_message(backtest_result)
    if realized_warning is not None:
        combined_limitations.append(realized_warning)
    return combined_limitations


def _realized_constraint_warning_message(
    backtest_result: WalkForwardBacktestResult,
) -> str | None:
    warnings = _build_realized_constraint_warning_table(backtest_result)
    if warnings is None:
        return None
    largest_breach = float(warnings["breach"].max())
    if backtest_result.realized_constraint_policy == "enforce_hard":
        return (
            "Contribution-only realized holdings still breached enforced caps in at least one "
            "rebalance window; see the realized constraint warnings table "
            f"(max breach {largest_breach:.4f})."
        )
    return (
        "Contribution-only realized holdings drifted outside configured caps under "
        "soft realized-constraint handling; see the realized constraint warnings "
        f"table (max breach {largest_breach:.4f})."
    )


def _build_realized_constraint_warning_table(
    backtest_result: WalkForwardBacktestResult,
) -> pd.DataFrame | None:
    violations = backtest_result.realized_constraint_violations
    if violations.empty:
        return None
    warnings = violations.copy()
    warnings["actual"] = warnings["actual"].astype(float).round(6)
    warnings["bound"] = warnings["bound"].astype(float).round(6)
    warnings["breach"] = warnings["breach"].astype(float).round(6)
    return warnings


def _build_return_attribution_table(
    weights_history: pd.DataFrame,
    *,
    asset_returns: pd.DataFrame | None,
    metadata: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Total per-asset and per-class contribution to portfolio return."""

    if asset_returns is None or asset_returns.empty or weights_history.empty:
        return None

    attribution = return_attribution(weights_history, asset_returns)
    per_asset_total = attribution.sum(axis=0)
    rows = [
        {
            "level": "Asset",
            "name": str(asset),
            "contribution": float(value),
        }
        for asset, value in per_asset_total.sort_values(ascending=False).items()
    ]

    if metadata is not None and "asset_class" in metadata.columns:
        asset_classes = metadata["asset_class"].reindex(attribution.columns)
        per_class_total = asset_class_return_attribution(attribution, asset_classes).sum(axis=0)
        rows.extend(
            {
                "level": "Asset Class",
                "name": str(asset_class),
                "contribution": float(value),
            }
            for asset_class, value in per_class_total.sort_values(ascending=False).items()
        )

    rows.append(
        {
            "level": "Total",
            "name": "Portfolio",
            "contribution": float(per_asset_total.sum()),
        }
    )
    return pd.DataFrame(rows)


def _build_risk_attribution_table(
    weights_history: pd.DataFrame,
    *,
    covariance_matrix: pd.DataFrame,
    metadata: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Per-asset and per-class risk contributions for the latest weights."""

    if weights_history.empty or covariance_matrix.empty:
        return None

    latest_weights = weights_history.iloc[-1].astype(float)
    common = latest_weights.index.intersection(covariance_matrix.index)
    if common.empty:
        return None
    contributions = risk_attribution(
        latest_weights.loc[common],
        covariance_matrix.loc[common, common],
    )

    rows = [
        {
            "level": "Asset",
            "name": str(asset),
            "risk_contribution": float(value),
        }
        for asset, value in contributions.sort_values(ascending=False).items()
    ]

    if metadata is not None and "asset_class" in metadata.columns:
        asset_classes = metadata["asset_class"].reindex(contributions.index)
        per_class = asset_class_risk_attribution(contributions, asset_classes)
        rows.extend(
            {
                "level": "Asset Class",
                "name": str(asset_class),
                "risk_contribution": float(value),
            }
            for asset_class, value in per_class.sort_values(ascending=False).items()
        )

    rows.append(
        {
            "level": "Total",
            "name": "Portfolio Volatility",
            "risk_contribution": float(contributions.sum()),
        }
    )
    return pd.DataFrame(rows)
