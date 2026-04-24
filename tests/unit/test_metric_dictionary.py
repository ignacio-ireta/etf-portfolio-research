from __future__ import annotations

from pathlib import Path

from etf_portfolio.metric_dictionary import (
    get_metric_definition,
    metric_dictionary_markdown,
    metric_dictionary_table,
)


def test_metric_dictionary_docs_are_generated_from_canonical_source() -> None:
    docs_path = Path(__file__).parents[2] / "docs" / "metric_dictionary.md"

    assert docs_path.read_text(encoding="utf-8") == metric_dictionary_markdown()


def test_metric_dictionary_table_has_required_columns_and_content() -> None:
    table = metric_dictionary_table()

    assert list(table.columns) == [
        "Metric",
        "Category",
        "Plain-English Meaning",
        "Formula-Level Summary",
        "Good/Bad Interpretation",
        "Caveats",
    ]
    assert not table.empty
    assert not table.isna().any().any()
    assert table.astype(str).apply(lambda column: column.str.strip().ne("")).all().all()


def test_metric_dictionary_covers_backtest_summary_metrics() -> None:
    expected_metrics = {
        "CAGR",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Max Drawdown",
        "Calmar Ratio",
        "Turnover",
        "Average Number of Holdings",
        "Largest Position",
        "Herfindahl Concentration Index",
        "Worst Month",
        "Worst Quarter",
        "Best Month",
        "Beta",
        "Alpha",
        "Tracking Error",
        "Information Ratio",
    }
    metric_names = set(metric_dictionary_table()["Metric"])

    assert expected_metrics.issubset(metric_names)


def test_metric_dictionary_lookup_returns_exact_definition() -> None:
    definition = get_metric_definition("Sharpe Ratio")

    assert definition.name == "Sharpe Ratio"
    assert "risk-free rate" in definition.plain_english
