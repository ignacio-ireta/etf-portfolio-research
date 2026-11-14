from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from etf_portfolio.backtesting.engine import WalkForwardBacktestResult
from etf_portfolio.reporting import report
from etf_portfolio.reporting.report import generate_report_bundle


def test_generate_report_bundle_creates_expected_artifacts(tmp_path) -> None:
    dates = pd.date_range("2024-01-31", periods=6, freq="ME")
    portfolio_returns = pd.Series(
        [0.01, 0.02, -0.005, 0.015, 0.01, 0.012],
        index=dates,
        name="portfolio",
    )
    benchmark_returns = pd.Series(
        [0.008, 0.018, -0.006, 0.011, 0.009, 0.01],
        index=dates,
        name="benchmark",
    )
    weights = pd.DataFrame(
        [
            {"AAA": 0.6, "BBB": 0.4},
            {"AAA": 0.55, "BBB": 0.45},
        ],
        index=pd.to_datetime(["2024-03-31", "2024-05-31"]),
    )
    weights.index.name = "rebalance_date"
    rebalance_summary = pd.DataFrame(
        {
            "train_start": pd.to_datetime(["2024-01-31", "2024-03-31"]),
            "train_end": pd.to_datetime(["2024-02-29", "2024-04-30"]),
            "observation_count": [2, 2],
            "transaction_cost": [0.0, 0.001],
        },
        index=weights.index,
    )
    backtest_result = WalkForwardBacktestResult(
        portfolio_returns=portfolio_returns,
        target_weights=weights,
        applied_weights=weights,
        rebalance_summary=rebalance_summary,
        realized_constraint_violations=pd.DataFrame(
            [
                {
                    "rebalance_date": pd.Timestamp("2024-05-31"),
                    "constraint_type": "ticker",
                    "identifier": "AAA",
                    "direction": "above_max",
                    "actual": 0.55,
                    "bound": 0.50,
                    "breach": 0.05,
                }
            ]
        ),
        realized_constraint_policy="report_drift",
    )

    expected_returns = pd.Series({"AAA": 0.12, "BBB": 0.08})
    covariance_matrix = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.03]],
        index=expected_returns.index,
        columns=expected_returns.index,
    )
    metadata = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "name": ["Asset A", "Asset B"],
            "asset_class": ["equity", "fixed_income"],
            "region": ["US", "US"],
            "currency": ["USD", "USD"],
            "expense_ratio": [0.0003, 0.0005],
            "benchmark_index": ["Index A", "Index B"],
            "inception_date": pd.to_datetime(["2010-01-01", "2011-01-01"]),
        },
    )
    prices = pd.DataFrame(
        {
            "AAA": [100.0, 101.0, 102.0, 101.5, 103.0, 104.0],
            "BBB": [100.0, 100.2, 100.4, 100.6, 100.5, 100.7],
        },
        index=dates,
    )

    artifacts = generate_report_bundle(
        backtest_result,
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        html_output_path=tmp_path / "latest_report.html",
        workbook_output_path=tmp_path / "portfolio_results.xlsx",
        figures_output_dir=tmp_path / "figures",
        benchmark_returns=benchmark_returns,
        primary_benchmark_returns=benchmark_returns,
        periods_per_year=12,
        metadata=metadata,
        prices=prices,
        assumptions={"provider": "Data provider: test fixture"},
        limitations=["Synthetic dataset for smoke testing."],
    )

    assert artifacts.html_path.exists()
    assert artifacts.workbook_path.exists()
    assert artifacts.html_path.stat().st_size > 0
    assert artifacts.workbook_path.stat().st_size > 0
    assert "efficient_frontier" in artifacts.figure_paths
    assert artifacts.figure_paths["efficient_frontier"].exists()
    report_html = artifacts.html_path.read_text(encoding="utf-8")
    assert "ETF Portfolio Research Report" in report_html
    assert "Reader Guide" in report_html
    assert "Research-tool warning" in report_html
    assert "This report is for research and education only." in report_html
    assert "Trust And Safety" in report_html
    assert "A high Sharpe ratio guarantees strong future returns." in report_html
    assert "Contribution-only portfolios always match optimizer targets." in report_html
    assert "Metric Dictionary" in report_html
    assert "Formula-Level Summary" in report_html
    assert "Sharpe Ratio" in report_html
    assert "How to read this section" in report_html
    assert "What this shows" in report_html
    assert "ETF Universe Summary" in report_html
    assert "Data Coverage Table" in report_html
    assert "Stress Periods" in report_html
    assert "Exposure" in report_html
    assert "Weighted Expense Ratio Over Time" in report_html
    assert "Assumptions and Limitations" in report_html
    assert "new weights and trade-cost impact start on the next return date" in report_html
    assert "Pipeline optimized benchmark objectives use the same rebalance mode" in report_html
    assert (
        "single-ETF and fixed-allocation benchmarks are theoretical return-series baselines"
        in report_html
    )
    assert "Realized Constraint Warnings" in report_html
    assert "Synthetic dataset for smoke testing." in report_html
    assert "Past performance does not guarantee future results." in report_html
    assert "slippage are not modeled" not in report_html
    assert (
        "Transaction costs and configured slippage are modeled as simple bps assumptions"
        in report_html
    )
    assert report_html.count("plotly.js v") == 1
    assert "drifted outside configured caps under soft realized-constraint handling" in report_html
    workbook = pd.ExcelFile(artifacts.workbook_path)
    assert "latest_realized_portfolio" in workbook.sheet_names
    assert "optimizer_target_portfolio" in workbook.sheet_names
    assert "trust_and_safety" in workbook.sheet_names
    assert "metric_dictionary" in workbook.sheet_names
    assert "optimized_portfolio" not in workbook.sheet_names
    assert "realized_constraint_warnings" in workbook.sheet_names
    workbook_assumptions = pd.read_excel(artifacts.workbook_path, sheet_name="assumptions")
    assumption_text = "\n".join(workbook_assumptions["value"].astype(str))
    assert "new weights and trade-cost impact start on the next return date" in assumption_text
    assert "Pipeline optimized benchmark objectives use the same rebalance mode" in assumption_text


def test_generate_report_bundle_passes_constraint_bundle_to_frontier(monkeypatch, tmp_path) -> None:
    dates = pd.date_range("2024-01-31", periods=6, freq="ME")
    portfolio_returns = pd.Series(
        [0.01, 0.02, -0.005, 0.015, 0.01, 0.012],
        index=dates,
        name="portfolio",
    )
    weights = pd.DataFrame(
        [
            {"AAA": 0.6, "BBB": 0.4},
            {"AAA": 0.55, "BBB": 0.45},
        ],
        index=pd.to_datetime(["2024-03-31", "2024-05-31"]),
    )
    weights.index.name = "rebalance_date"
    rebalance_summary = pd.DataFrame(
        {
            "train_start": pd.to_datetime(["2024-01-31", "2024-03-31"]),
            "train_end": pd.to_datetime(["2024-02-29", "2024-04-30"]),
            "observation_count": [2, 2],
            "transaction_cost": [0.0, 0.001],
        },
        index=weights.index,
    )
    backtest_result = WalkForwardBacktestResult(
        portfolio_returns=portfolio_returns,
        target_weights=weights,
        applied_weights=weights,
        rebalance_summary=rebalance_summary,
        realized_constraint_policy="report_drift",
    )
    expected_returns = pd.Series({"AAA": 0.12, "BBB": 0.08})
    covariance_matrix = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.03]],
        index=expected_returns.index,
        columns=expected_returns.index,
    )
    metadata = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "name": ["Asset A", "Asset B"],
            "asset_class": ["equity", "fixed_income"],
            "region": ["US", "US"],
            "currency": ["USD", "USD"],
            "expense_ratio": [0.0003, 0.0005],
        },
    )
    asset_classes = pd.Series({"AAA": "equity", "BBB": "fixed_income"})
    expense_ratios = pd.Series({"AAA": 0.0003, "BBB": 0.0005})
    captured: list[dict[str, object]] = []

    def fake_build_efficient_frontier_figure(*args, **kwargs):
        captured.append(kwargs.copy())
        return go.Figure()

    monkeypatch.setattr(
        report,
        "build_efficient_frontier_figure",
        fake_build_efficient_frontier_figure,
    )

    generate_report_bundle(
        backtest_result,
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        html_output_path=tmp_path / "latest_report.html",
        workbook_output_path=tmp_path / "portfolio_results.xlsx",
        figures_output_dir=tmp_path / "figures",
        periods_per_year=12,
        metadata=metadata,
        asset_classes=asset_classes,
        asset_class_bounds={"fixed_income": (0.10, 0.45)},
        ticker_bounds={"AAA": (0.20, 0.80)},
        bond_assets=["BBB"],
        min_bond_exposure=0.10,
        expense_ratios=expense_ratios,
    )

    assert captured
    assert captured[0]["asset_classes"] is asset_classes
    assert captured[0]["asset_class_bounds"] == {"fixed_income": (0.10, 0.45)}
    assert captured[0]["ticker_bounds"] == {"AAA": (0.20, 0.80)}
    assert captured[0]["bond_assets"] == ["BBB"]
    assert captured[0]["min_bond_exposure"] == 0.10
    assert captured[0]["expense_ratios"] is expense_ratios


def test_ruled_table_keeps_cell_text_and_marks_numeric_columns() -> None:
    from etf_portfolio.reporting.layout import ruled_table, table_cells, table_html

    frame = pd.DataFrame(
        {
            "Strategy": ["Optimized Strategy", "A & B <benchmark>"],
            "CAGR": [0.104314, -0.05],
            "rebalance_date": pd.to_datetime(["2024-03-31", "2024-05-31"]),
        }
    )
    frame.columns.name = "Metric"
    source = table_html(frame)
    rendered = ruled_table(source)

    source_header, source_rows = table_cells(source)
    header, rows = table_cells(rendered)
    # pandas' empty index-name column is dropped; every other cell is verbatim.
    assert header == source_header[1:]
    assert rows == [row[1:] for row in source_rows]
    assert '<td class="num">0.104314</td>' in rendered
    assert '<td class="num">-0.050000</td>' in rendered
    assert '<td class="date">2024-03-31</td>' in rendered
    assert "<td>A &amp; B &lt;benchmark&gt;</td>" in rendered


def test_strategy_takes_the_accent_and_benchmarks_keep_fixed_colors() -> None:
    from etf_portfolio.reporting import plots

    names = ["Portfolio", "Selected Benchmark ETF", "60/40 Portfolio", "custom_baseline"]
    colors = plots.series_colors(names)
    assert colors["Portfolio"] == plots.ACCENT
    assert colors["Selected Benchmark ETF"] == plots.PRIMARY_BENCHMARK
    assert plots.ACCENT not in {colors[name] for name in names[1:]}
    # Colour follows the entity, not its position in the list.
    reordered = plots.series_colors(["60/40 Portfolio", "Portfolio"])
    assert reordered["60/40 Portfolio"] == colors["60/40 Portfolio"]

    dates = pd.date_range("2024-01-31", periods=4, freq="ME")
    returns = pd.Series([0.01, -0.02, 0.015, 0.01], index=dates, name="portfolio")
    benchmark = pd.Series([0.008, -0.01, 0.011, 0.009], index=dates, name="benchmark")
    for figure in (
        plots.build_cumulative_returns_figure(returns, benchmark_returns=benchmark),
        plots.build_drawdown_figure(returns, benchmark_returns=benchmark),
    ):
        strategy = [trace for trace in figure.data if trace.name == plots.STRATEGY_LABEL]
        assert len(strategy) == 1
        assert strategy[0].line.color == plots.ACCENT
        assert figure.layout.paper_bgcolor == plots.PAPER


def test_report_page_is_an_unboxed_numbered_note(tmp_path) -> None:
    from etf_portfolio.reporting.layout import (
        FigureBlock,
        ReportMeta,
        ReportSection,
        TableBlock,
        render_report_page,
        table_html,
    )

    figure = go.Figure(go.Scatter(x=[1, 2], y=[1, 2]))
    figure.update_layout(title_text="Hidden in the page, kept in the PNG")
    page = render_report_page(
        title="etf_portfolio_research Backtest Report",
        lede="Lede.",
        disclaimer="Disclaimer.",
        meta=ReportMeta(
            run_id="run-all-20260425T083756Z-f6f9560b",
            window_start="2015-02-02",
            window_end="2026-04-24",
            tickers=("VTI", "BND"),
            rebalance_count=135,
            last_rebalance="2026-03-31",
            headline=(("CAGR", "10.4%"),),
        ),
        sections=[
            ReportSection(
                title="Backtest Performance",
                blocks=(
                    TableBlock(table_html(pd.DataFrame({"Strategy": ["A"], "CAGR": [0.1]}))),
                    FigureBlock("cumulative_returns", figure),
                ),
            )
        ],
    )

    assert "<h1>ETF Portfolio Research Backtest Report</h1>" in page
    assert "run-all-20260425T083756Z-f6f9560b" in page
    assert "2015-02-02 → 2026-04-24" in page
    assert 'class="card' not in page
    assert '<span class="ref">Fig. 1</span>' in page
    assert '<span class="ref">Table 1</span>' in page
    assert "Cumulative returns. Compounded return" in page
    assert "Hidden in the page" not in page
    # Self-contained: no remote stylesheets, fonts, or scripts.
    assert "<link" not in page
    assert "src=" not in page.split("</script>", 1)[1]
    assert figure.layout.title.text == "Hidden in the page, kept in the PNG"
