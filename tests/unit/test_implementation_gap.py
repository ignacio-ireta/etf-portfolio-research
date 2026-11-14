"""Tests for reporting the gap between realized holdings and the optimizer target."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from etf_portfolio import cli
from etf_portfolio.backtesting.metrics import summarize_implementation_gap
from etf_portfolio.config import load_config


def _write_processed_data(project_root: Path, prices: pd.DataFrame) -> None:
    processed = project_root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(processed / "prices_validated.parquet")
    prices.pct_change(fill_method=None).dropna().to_parquet(processed / "returns.parquet")


def test_gap_is_half_the_absolute_weight_difference() -> None:
    dates = pd.to_datetime(["2021-01-29", "2021-02-26"])
    target = pd.DataFrame({"AAA": [0.5, 0.5], "BBB": [0.5, 0.5]}, index=dates)
    applied = pd.DataFrame({"AAA": [0.5, 0.7], "BBB": [0.5, 0.3]}, index=dates)
    violations = pd.DataFrame(
        {
            "rebalance_date": [dates[1], dates[1]],
            "constraint_type": ["ticker", "asset_class"],
        }
    )

    summary = summarize_implementation_gap(target, applied, violations)

    assert summary["rebalance_count"] == 2
    assert summary["mean_one_way_gap"] == pytest.approx(0.1)
    assert summary["max_one_way_gap"] == pytest.approx(0.2)
    assert summary["latest_one_way_gap"] == pytest.approx(0.2)
    assert summary["rebalances_with_constraint_violation"] == 1


def test_gap_summary_of_empty_inputs_is_all_zero() -> None:
    summary = summarize_implementation_gap(pd.DataFrame(), pd.DataFrame(), None)

    assert summary == {
        "rebalance_count": 0,
        "mean_one_way_gap": 0.0,
        "max_one_way_gap": 0.0,
        "latest_one_way_gap": 0.0,
        "rebalances_with_constraint_violation": 0,
    }


def test_gap_summary_treats_missing_columns_as_zero_weight() -> None:
    dates = pd.to_datetime(["2021-01-29"])
    target = pd.DataFrame({"AAA": [1.0]}, index=dates)
    applied = pd.DataFrame({"AAA": [0.6], "BBB": [0.4]}, index=dates)

    summary = summarize_implementation_gap(target, applied)

    assert summary["mean_one_way_gap"] == pytest.approx(0.4)


def test_backtest_outputs_gap_and_full_rebalance_target(
    tiny_project: Path, synthetic_prices: pd.DataFrame
) -> None:
    _write_processed_data(tiny_project, synthetic_prices)

    cli.run_backtest("configs/base.yaml", project_root=tiny_project, lookback_periods=5)

    metrics = json.loads(
        (tiny_project / "reports/metrics/backtest_metrics.json").read_text(encoding="utf-8")
    )
    assert set(metrics["implementation_gap"]) == {
        "rebalance_count",
        "mean_one_way_gap",
        "max_one_way_gap",
        "latest_one_way_gap",
        "rebalances_with_constraint_violation",
    }
    assert metrics["implementation_gap"]["rebalance_count"] > 0
    assert cli.OPTIMIZER_TARGET_LABEL in metrics["benchmarks"]

    report_html = (tiny_project / "reports/html/latest_report.html").read_text(encoding="utf-8")
    assert "Implementation gap:" in report_html
    assert cli.OPTIMIZER_TARGET_LABEL in report_html


def test_full_rebalance_run_has_no_separate_target_benchmark(
    tiny_project: Path, synthetic_prices: pd.DataFrame
) -> None:
    _write_processed_data(tiny_project, synthetic_prices)
    config_path = tiny_project / "configs/base.yaml"
    original = config_path.read_text(encoding="utf-8")
    assert "  mode: contribution_only\n" in original
    config_path.write_text(
        original.replace("  mode: contribution_only\n", "  mode: full_rebalance\n"),
        encoding="utf-8",
    )

    cli.run_backtest("configs/base.yaml", project_root=tiny_project, lookback_periods=5)

    metrics = json.loads(
        (tiny_project / "reports/metrics/backtest_metrics.json").read_text(encoding="utf-8")
    )
    assert cli.OPTIMIZER_TARGET_LABEL not in metrics["benchmarks"]
    assert metrics["implementation_gap"]["mean_one_way_gap"] == pytest.approx(0.0, abs=1e-9)


def test_assumption_line_names_the_gap_and_points_to_the_target_row(tiny_project: Path) -> None:
    config = load_config(tiny_project / "configs/base.yaml")

    assumptions = cli._report_assumptions(
        config,
        "equal_weight",
        implementation_gap={
            "rebalance_count": 135,
            "mean_one_way_gap": 0.267,
            "max_one_way_gap": 0.46,
            "latest_one_way_gap": 0.30,
            "rebalances_with_constraint_violation": 134,
        },
    )

    text = assumptions["implementation_gap"]
    assert text.startswith("Implementation gap:")
    assert "26.7%" in text
    assert "134 of 135" in text
    assert cli.OPTIMIZER_TARGET_LABEL in text
