"""Generated reporting bundle for ETF portfolio research."""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from plotly.graph_objects import Figure
from plotly.io import write_image

from etf_portfolio.backtesting.engine import WalkForwardBacktestResult
from etf_portfolio.io_utils import atomic_path, atomic_write_bytes, atomic_write_text
from etf_portfolio.logging_config import get_logger, log_event
from etf_portfolio.metric_dictionary import metric_dictionary_table
from etf_portfolio.reporting.layout import (
    Block,
    FigureBlock,
    ListBlock,
    ReportMeta,
    ReportSection,
    TableBlock,
    TextBlock,
    headline_from_metrics,
    render_frontier_page,
    render_report_page,
    table_html,
)
from etf_portfolio.reporting.plots import (
    FIGURE_EXPORT_HEIGHT,
    FIGURE_EXPORT_SCALE,
    FIGURE_EXPORT_WIDTH,
    build_benchmark_comparison_figure,
    build_cumulative_returns_figure,
    build_drawdown_figure,
    build_efficient_frontier_figure,
    build_group_exposure_figure,
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
from etf_portfolio.trust_safety import common_false_conclusions_table

LOGGER = get_logger(__name__)
PLACEHOLDER_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)
RESEARCH_TOOL_DISCLAIMER = (
    "This report is for research and education only. It is not financial advice, "
    "investment advice, tax advice, legal advice, or a recommendation to buy or sell "
    "any security."
)
REPORT_LEDE = (
    "Code-generated research report built from pipeline outputs. Use it to understand "
    "model behavior, assumptions, risk, and historical simulations; do not treat it as "
    "a recommendation."
)
FRONTIER_LEDE = (
    "The risk and return combinations the optimizer could reach under the configured "
    "constraints, with each ETF and the optimized portfolio marked. Hover a point for "
    "its exact values; drag to zoom."
)
EXECUTION_COST_LIMITATION = (
    "Transaction costs and configured slippage are modeled as simple bps assumptions; "
    "taxes, spreads beyond configured slippage, market impact, and account-specific "
    "fees are not modeled."
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
    "Metric Dictionary": (
        (
            "What this shows",
            "Canonical plain-language definitions for metrics used by the report and pipeline.",
        ),
        (
            "How to interpret it",
            "Use this as the reference point for what each metric means before comparing values.",
        ),
        (
            "What to watch",
            "Metric values are only meaningful when the caveats and assumptions fit the question "
            "you are asking.",
        ),
    ),
    "Trust And Safety": (
        (
            "What this shows",
            "Common ways readers can overinterpret the report, plus safer readings to use instead.",
        ),
        (
            "How to interpret it",
            "Read this before treating any optimized portfolio, metric, or backtest as a decision.",
        ),
        (
            "What to watch",
            "The report is evidence under assumptions. It is not a forecast, guarantee, or "
            "personalized recommendation.",
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
    benchmark_weights: dict[str, pd.DataFrame] | None = None,
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
    run_id: str | None = None,
) -> Path:
    """Generate the consolidated HTML report."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    sections, strategy_metrics = _build_report_sections(
        backtest_result,
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        benchmark_returns=benchmark_returns,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
        max_weight=max_weight,
        benchmark_suite=benchmark_suite,
        benchmark_weights=benchmark_weights,
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

    report_html = render_report_page(
        title=title,
        lede=REPORT_LEDE,
        disclaimer=RESEARCH_TOOL_DISCLAIMER,
        meta=_build_report_meta(
            backtest_result,
            run_id=run_id,
            strategy_metrics=strategy_metrics,
        ),
        sections=sections,
    )

    atomic_write_text(output, report_html)
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
    benchmark_weights: dict[str, pd.DataFrame] | None = None,
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
    run_id: str | None = None,
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
        benchmark_weights=benchmark_weights,
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
        run_id=run_id,
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
        benchmark_weights=benchmark_weights,
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
    benchmark_weights: dict[str, pd.DataFrame] | None,
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
) -> tuple[list[ReportSection], dict[str, Any]]:
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
        benchmark_weights=benchmark_weights,
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

    sections: list[ReportSection] = [
        _section(
            "Reader Guide",
            *_build_reader_guide_section(
                backtest_result,
                benchmark_returns=benchmark_returns,
                primary_benchmark_returns=primary_benchmark_returns,
                metadata=metadata,
                prices=prices,
            ),
        ),
        _section("Trust And Safety", _table(common_false_conclusions_table())),
        _section("Metric Dictionary", _table(metric_dictionary_table())),
    ]
    if metadata is not None:
        sections.append(
            _section(
                "ETF Universe Summary",
                _table(
                    build_etf_universe_summary_table(
                        metadata,
                        tickers=backtest_result.weights.columns.tolist(),
                    )
                ),
            )
        )
    if prices is not None:
        sections.extend(
            [
                _section(
                    "Data Coverage Table",
                    _table(build_data_coverage_table(prices, metadata=metadata)),
                ),
                _section("Missing Data Table", _table(build_missing_data_table(prices))),
            ]
        )
    sections.extend(
        [
            _section("Efficient Frontier Chart", _figure("efficient_frontier", figures)),
            _section("Latest Realized Portfolio Table", _table(latest_realized_weights)),
            _section("Optimizer Target Portfolio Table", _table(optimizer_target_weights)),
            _section("Portfolio Weights", _figure("portfolio_weights", figures)),
        ]
    )
    if metadata is not None:
        exposure_blocks: list[Block] = [
            _table(
                build_group_exposure_table(
                    backtest_result.weights,
                    metadata,
                    field="asset_class",
                    label="Asset Class",
                )
            ),
            _figure("asset_class_exposure", figures),
        ]
        if "region" in metadata.columns:
            exposure_blocks.extend(
                [
                    _table(
                        build_group_exposure_table(
                            backtest_result.weights,
                            metadata,
                            field="region",
                            label="Region",
                        )
                    ),
                    _figure("region_exposure", figures),
                ]
            )
        sections.append(_section("Exposure", *exposure_blocks))
    sections.extend(
        [
            _section(
                "Benchmark Comparison",
                _table(metrics_table),
                _figure("benchmark_comparison", figures),
            ),
            _section(
                "Backtest Performance",
                _table(metrics_table),
                _figure("cumulative_returns", figures),
            ),
            _section("Drawdown Chart", _figure("drawdown", figures)),
            _section(
                "Rolling Risk Metrics",
                _figure("rolling_volatility", figures),
                _figure("rolling_sharpe", figures),
                _figure("rolling_correlation", figures),
            ),
            _section(
                "Stress Periods",
                _table(stress_table),
                _figure("stress_periods", figures),
            ),
        ]
    )

    if metadata is not None and "expense_ratio" in metadata.columns:
        sections.append(
            _section(
                "Weighted Expense Ratio Over Time",
                _table(
                    build_weighted_expense_over_time_table(
                        backtest_result.weights,
                        metadata,
                    )
                ),
                _figure("weighted_expense_over_time", figures),
            )
        )

    return_attr_table = _build_return_attribution_table(
        backtest_result.weights,
        asset_returns=asset_returns,
        metadata=metadata,
    )
    if return_attr_table is not None:
        sections.append(_section("Return Attribution", _table(return_attr_table)))

    risk_attr_table = _build_risk_attribution_table(
        backtest_result.weights,
        covariance_matrix=covariance_matrix,
        metadata=metadata,
    )
    if risk_attr_table is not None:
        sections.append(_section("Risk Attribution", _table(risk_attr_table)))

    realized_constraint_warnings = _build_realized_constraint_warning_table(backtest_result)
    if realized_constraint_warnings is not None:
        sections.append(
            _section("Realized Constraint Warnings", _table(realized_constraint_warnings))
        )

    sections.append(
        _section(
            "Assumptions and Limitations",
            *_build_assumptions_appendix(
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
    return sections, _strategy_metrics(metrics_table)


def _strategy_metrics(metrics_table: pd.DataFrame) -> dict[str, Any]:
    """Pick the optimized strategy's metrics out of either metrics-table shape."""

    if "Strategy" in metrics_table.columns:
        rows = metrics_table.loc[metrics_table["Strategy"] == "Optimized Strategy"]
        if rows.empty:
            return {}
        return {str(key): value for key, value in rows.iloc[0].items()}
    if "Value" in metrics_table.columns and len(metrics_table.columns) == 2:
        label_column = next(column for column in metrics_table.columns if column != "Value")
        return {
            str(label): value
            for label, value in zip(
                metrics_table[label_column], metrics_table["Value"], strict=True
            )
        }
    return {}


def _section(title: str, *blocks: Block) -> ReportSection:
    return ReportSection(
        title=title,
        blocks=tuple(blocks),
        notes=SECTION_EXPLANATIONS.get(title, ()),
    )


def _table(table: pd.DataFrame) -> TableBlock:
    return TableBlock(table_html(table))


def _figure(name: str, figures: dict[str, Figure]) -> FigureBlock:
    return FigureBlock(name, figures[name])


def _build_report_meta(
    backtest_result: WalkForwardBacktestResult,
    *,
    run_id: str | None,
    strategy_metrics: dict[str, Any],
) -> ReportMeta:
    """Collect the run facts shown in the masthead from the run itself."""

    returns = backtest_result.portfolio_returns
    weights = backtest_result.weights
    return ReportMeta(
        run_id=run_id,
        window_start=None if returns.empty else str(returns.index.min().date()),
        window_end=None if returns.empty else str(returns.index.max().date()),
        tickers=tuple(str(column) for column in weights.columns),
        rebalance_count=len(weights.index),
        last_rebalance=None if weights.empty else str(weights.index.max().date()),
        headline=headline_from_metrics(strategy_metrics),
    )


def write_frontier_page(
    figure: Figure,
    output_path: str | Path,
    *,
    tickers: list[str] | tuple[str, ...] = (),
    run_id: str | None = None,
) -> Path:
    """Write the standalone efficient-frontier explorer page."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        output,
        render_frontier_page(
            figure,
            lede=FRONTIER_LEDE,
            disclaimer=RESEARCH_TOOL_DISCLAIMER,
            meta=ReportMeta(run_id=run_id, tickers=tuple(str(ticker) for ticker in tickers)),
            notes=SECTION_EXPLANATIONS["Efficient Frontier Chart"],
        ),
    )
    return output


def _build_reader_guide_section(
    backtest_result: WalkForwardBacktestResult,
    *,
    benchmark_returns: pd.Series | pd.DataFrame | None,
    primary_benchmark_returns: pd.Series | None,
    metadata: pd.DataFrame | None,
    prices: pd.DataFrame | None,
) -> tuple[Block, ...]:
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
        TextBlock(
            "This opening guide is meant to make the report readable before you inspect the "
            "tables and charts. The results are conditional on the configured inputs and "
            "should be interpreted as a historical research experiment."
        ),
        _table(guide_rows),
        ListBlock(tuple(reading_order)),
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
        figures["benchmark_comparison"].update_layout(title_text="Benchmark comparison unavailable")
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
            title_text=f"Rolling correlation to benchmark ({rolling_window} periods) unavailable"
        )
    if metadata is not None:
        metadata_by_ticker = metadata.copy()
        if "ticker" in metadata_by_ticker.columns:
            metadata_by_ticker = metadata_by_ticker.set_index("ticker")
        aligned_metadata = metadata_by_ticker.reindex(backtest_result.weights.columns)
        figures["asset_class_exposure"] = build_group_exposure_figure(
            backtest_result.weights,
            aligned_metadata["asset_class"],
            title="Portfolio weights by asset class",
            subtitle="Share of the latest portfolio in each asset class",
        )
        if "region" in aligned_metadata.columns:
            figures["region_exposure"] = build_group_exposure_figure(
                backtest_result.weights,
                aligned_metadata["region"],
                title="Portfolio weights by region",
                subtitle="Share of the latest portfolio in each region",
            )
        if "expense_ratio" in aligned_metadata.columns:
            figures["weighted_expense_over_time"] = build_weighted_expense_ratio_over_time_figure(
                backtest_result.weights,
                aligned_metadata["expense_ratio"],
            )
    else:
        figures["asset_class_exposure"] = build_weights_figure(backtest_result.weights)
        figures["asset_class_exposure"].update_layout(title_text="Asset-class exposure unavailable")
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
    benchmark_weights: dict[str, pd.DataFrame] | None,
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
            benchmark_weights=benchmark_weights,
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
    workbook_tables["trust_and_safety"] = common_false_conclusions_table()
    workbook_tables["metric_dictionary"] = metric_dictionary_table()

    with (
        atomic_path(output) as temp_path,
        pd.ExcelWriter(temp_path, engine="openpyxl") as writer,
    ):
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
            with atomic_path(path) as temp_path:
                write_image(
                    figure,
                    str(temp_path),
                    format="png",
                    width=FIGURE_EXPORT_WIDTH,
                    height=FIGURE_EXPORT_HEIGHT,
                    scale=FIGURE_EXPORT_SCALE,
                )
        except Exception as exc:
            atomic_write_bytes(path, PLACEHOLDER_PNG)
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
        "Walk-forward backtest estimates target weights only with returns observed strictly "
        "before each rebalance date.",
    ),
    (
        "execution_timing",
        "The return labeled with a rebalance date belongs to the previous holding period; "
        "new weights and trade-cost impact start on the next return date.",
    ),
    (
        "costs",
        EXECUTION_COST_LIMITATION,
    ),
    (
        "rebalancing",
        "Contribution-only mode avoids selling unless fallback or hard-constraint policies "
        "trigger; tolerance-band mode uses constrained projection when configured drift "
        "bands are breached.",
    ),
    (
        "benchmark_fairness",
        "Pipeline optimized benchmark objectives use the same rebalance mode, contribution "
        "amount, initial capital, costs, constraints, and schedule as the main strategy; "
        "single-ETF and fixed-allocation benchmarks are theoretical return-series baselines.",
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
) -> tuple[Block, ...]:
    """Build the mandatory assumptions, parameters, and limitations appendix."""

    blocks: list[Block] = []

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
    blocks.append(_table(pd.DataFrame(config_rows)))

    merged_assumptions: dict[str, str] = dict(_DEFAULT_ASSUMPTIONS)
    for key, value in (assumptions or {}).items():
        merged_assumptions[str(key)] = str(value)
    assumption_table = pd.DataFrame(
        [{"key": key, "assumption": value} for key, value in merged_assumptions.items()]
    )
    blocks.append(_table(assumption_table))

    if metadata is not None:
        profile = build_portfolio_profile_table(
            backtest_result.portfolio_returns,
            weights=backtest_result.weights,
            metadata=metadata,
            benchmark_returns=primary_benchmark_returns,
        )
        blocks.append(_table(profile))

    standard_limitations = [
        RESEARCH_TOOL_DISCLAIMER,
        "Past performance does not guarantee future results.",
        EXECUTION_COST_LIMITATION,
        "Provider data may be subject to revisions or vendor outages.",
    ]
    combined_limitations = list(standard_limitations) + list(limitations or [])
    realized_warning = _realized_constraint_warning_message(backtest_result)
    if realized_warning is not None:
        combined_limitations.append(realized_warning)
    blocks.append(ListBlock(tuple(str(item) for item in combined_limitations)))

    return tuple(blocks)


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
    merged_assumptions: dict[str, str] = dict(_DEFAULT_ASSUMPTIONS)
    for key, value in (assumptions or {}).items():
        merged_assumptions[str(key)] = str(value)
    rows.extend(
        {"section": "assumption", "key": key, "value": value}
        for key, value in merged_assumptions.items()
    )
    standard_limitations = [
        RESEARCH_TOOL_DISCLAIMER,
        "Past performance does not guarantee future results.",
        EXECUTION_COST_LIMITATION,
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
