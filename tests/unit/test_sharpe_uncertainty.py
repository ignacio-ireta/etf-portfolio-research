"""Tests for the bootstrap confidence interval on Sharpe-ratio differences."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from etf_portfolio import cli
from etf_portfolio.backtesting.uncertainty import bootstrap_sharpe_difference
from etf_portfolio.config import load_config


def _noise(seed: int, periods: int = 500) -> pd.Series:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2018-01-01", periods=periods)
    return pd.Series(rng.normal(loc=0.0004, scale=0.01, size=periods), index=index)


def test_result_is_deterministic_for_a_fixed_seed() -> None:
    first = bootstrap_sharpe_difference(_noise(1), _noise(2), seed=7)
    second = bootstrap_sharpe_difference(_noise(1), _noise(2), seed=7)
    other_seed = bootstrap_sharpe_difference(_noise(1), _noise(2), seed=8)

    assert first == second
    assert first["ci_lower"] != other_seed["ci_lower"]


def test_identical_series_have_zero_difference() -> None:
    series = _noise(3)

    result = bootstrap_sharpe_difference(series, series)

    assert result["sharpe_difference"] == pytest.approx(0.0)
    assert result["ci_lower"] == pytest.approx(0.0)
    assert result["ci_upper"] == pytest.approx(0.0)
    assert result["probability_not_positive"] == pytest.approx(1.0)


def test_a_constant_daily_edge_gives_an_interval_above_zero() -> None:
    benchmark = _noise(4)
    strategy = benchmark + 0.001

    result = bootstrap_sharpe_difference(strategy, benchmark)

    assert result["sharpe_difference"] > 0.0
    assert result["ci_lower"] > 0.0
    assert result["probability_not_positive"] == pytest.approx(0.0)


def test_two_unrelated_noise_series_give_an_interval_that_contains_zero() -> None:
    result = bootstrap_sharpe_difference(_noise(5), _noise(6))

    assert result["ci_lower"] < 0.0 < result["ci_upper"]
    assert result["n_observations"] == 500
    assert result["n_resamples"] == 1000
    assert result["mean_block_length"] == 21
    assert result["confidence_level"] == pytest.approx(0.95)


def test_point_estimate_matches_the_metrics_module() -> None:
    from etf_portfolio.backtesting.metrics import sharpe_ratio

    strategy, benchmark = _noise(9), _noise(10)

    result = bootstrap_sharpe_difference(strategy, benchmark, risk_free_rate=0.03)

    expected = sharpe_ratio(strategy, risk_free_rate=0.03) - sharpe_ratio(
        benchmark, risk_free_rate=0.03
    )
    assert result["sharpe_difference"] == pytest.approx(expected)


def test_too_few_observations_return_none_results() -> None:
    index = pd.bdate_range("2021-01-04", periods=2)
    short = pd.Series([0.01, -0.01], index=index)

    result = bootstrap_sharpe_difference(short, short)

    assert result["sharpe_difference"] is None
    assert result["ci_lower"] is None
    assert result["ci_upper"] is None
    assert result["n_observations"] == 2


def test_missing_values_are_dropped_pairwise() -> None:
    strategy = _noise(11)
    benchmark = _noise(12)
    benchmark.iloc[:100] = np.nan

    result = bootstrap_sharpe_difference(strategy, benchmark)

    assert result["n_observations"] == 400


def test_invalid_arguments_are_rejected() -> None:
    series = _noise(13)
    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap_sharpe_difference(series, series, n_resamples=0)
    with pytest.raises(ValueError, match="confidence_level"):
        bootstrap_sharpe_difference(series, series, confidence_level=1.0)


def test_backtest_writes_sharpe_uncertainty_for_every_benchmark(
    tiny_project: Path, synthetic_prices: pd.DataFrame
) -> None:
    processed = tiny_project / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    synthetic_prices.to_parquet(processed / "prices_validated.parquet")
    synthetic_prices.pct_change(fill_method=None).dropna().to_parquet(processed / "returns.parquet")

    cli.run_backtest("configs/base.yaml", project_root=tiny_project, lookback_periods=5)

    metrics = json.loads(
        (tiny_project / "reports/metrics/backtest_metrics.json").read_text(encoding="utf-8")
    )
    uncertainty = metrics["sharpe_difference_vs_benchmarks"]
    assert set(uncertainty) == set(metrics["benchmarks"])
    primary = uncertainty["Selected Benchmark ETF"]
    assert primary["ci_lower"] <= primary["sharpe_difference"] <= primary["ci_upper"]

    report_html = (tiny_project / "reports/html/latest_report.html").read_text(encoding="utf-8")
    assert "Statistical uncertainty:" in report_html


def test_assumption_line_says_when_the_interval_contains_zero(tiny_project: Path) -> None:
    config = load_config(tiny_project / "configs/base.yaml")
    base = {
        "confidence_level": 0.95,
        "n_observations": 2800,
        "n_resamples": 1000,
        "mean_block_length": 21,
        "seed": 0,
        "probability_not_positive": 0.17,
    }

    inconclusive = cli._report_assumptions(
        config,
        "max_sharpe",
        sharpe_uncertainty={**base, "sharpe_difference": 0.10, "ci_lower": -0.10, "ci_upper": 0.30},
    )["statistical_uncertainty"]
    conclusive = cli._report_assumptions(
        config,
        "max_sharpe",
        sharpe_uncertainty={**base, "sharpe_difference": 0.40, "ci_lower": 0.10, "ci_upper": 0.70},
    )["statistical_uncertainty"]

    assert inconclusive.startswith("Statistical uncertainty:")
    assert "[-0.10, +0.30]" in inconclusive
    assert "contains zero" in inconclusive
    assert "excludes zero" in conclusive


def test_assumption_line_is_omitted_without_a_usable_interval(tiny_project: Path) -> None:
    config = load_config(tiny_project / "configs/base.yaml")

    assumptions = cli._report_assumptions(
        config,
        "max_sharpe",
        sharpe_uncertainty={"sharpe_difference": None, "ci_lower": None, "ci_upper": None},
    )

    assert "statistical_uncertainty" not in assumptions
