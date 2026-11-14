"""Tests for the versioned lookback and the pre-window estimation warm-up."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from etf_portfolio import cli
from etf_portfolio.config import BacktestConfig, load_config


def _write_processed_data(project_root: Path, prices: pd.DataFrame) -> None:
    processed = project_root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(processed / "prices_validated.parquet")
    prices.pct_change(fill_method=None).dropna().to_parquet(processed / "returns.parquet")


def _set_backtest_section(project_root: Path, *, start_date: str, lookback_periods: int) -> None:
    config_path = project_root / "configs/base.yaml"
    original = config_path.read_text(encoding="utf-8")
    old_block = '  start_date: "2020-01-02"\n  end_date: null\n'
    assert old_block in original
    new_block = (
        f'  start_date: "{start_date}"\n  end_date: null\n  lookback_periods: {lookback_periods}\n'
    )
    config_path.write_text(original.replace(old_block, new_block), encoding="utf-8")


def test_backtest_config_lookback_defaults_to_756() -> None:
    assert BacktestConfig().lookback_periods == 756


def test_backtest_config_rejects_non_positive_lookback() -> None:
    with pytest.raises(ValidationError):
        BacktestConfig(lookback_periods=0)


def test_resolve_lookback_prefers_cli_override(tiny_project: Path) -> None:
    _set_backtest_section(tiny_project, start_date="2020-01-02", lookback_periods=30)
    config = load_config(tiny_project / "configs/base.yaml")

    assert cli._resolve_lookback_periods(config, None) == 30
    assert cli._resolve_lookback_periods(config, 7) == 7


def test_history_before_start_date_warms_up_the_lookback(
    tiny_project: Path, synthetic_prices: pd.DataFrame
) -> None:
    """The first month-end inside the window must trade, using pre-window history."""

    _write_processed_data(tiny_project, synthetic_prices)
    _set_backtest_section(tiny_project, start_date="2020-03-02", lookback_periods=30)

    cli.run_backtest("configs/base.yaml", project_root=tiny_project)

    metrics = json.loads(
        (tiny_project / "reports/metrics/backtest_metrics.json").read_text(encoding="utf-8")
    )
    record = json.loads((tiny_project / metrics["run_record"]).read_text(encoding="utf-8"))

    # Rebalance on 2020-03-31 (first month-end on/after the configured start);
    # the first simulated return is the next trading day.
    assert record["start_date"] == "2020-04-01"
    assert record["lookback_periods"] == 30
    assert record["configured_lookback_periods"] == 30
    assert record["configured_backtest_start_date"] == "2020-03-02"


def test_report_assumptions_state_the_effective_window(tiny_project: Path) -> None:
    config = load_config(tiny_project / "configs/base.yaml")

    assumptions = cli._report_assumptions(
        config,
        "equal_weight",
        lookback_periods=30,
        effective_start_date=pd.Timestamp("2020-04-01"),
    )

    assert "2020-01-02" in assumptions["backtest_window"]
    assert "2020-04-01" in assumptions["backtest_window"]
    assert "30" in assumptions["backtest_window"]
