"""Tests for the `sensitivity` command (lookback / risk-free / rebalance-mode grid)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from etf_portfolio import cli
from etf_portfolio.errors import InsufficientHistoryError


def _write_processed_data(project_root: Path, prices: pd.DataFrame) -> None:
    processed = project_root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(processed / "prices_validated.parquet")
    prices.pct_change(fill_method=None).dropna().to_parquet(processed / "returns.parquet")


def test_sensitivity_grid_writes_one_row_per_combination(
    tiny_project: Path, synthetic_prices: pd.DataFrame
) -> None:
    _write_processed_data(tiny_project, synthetic_prices)

    output_path = cli.run_sensitivity(
        "configs/base.yaml",
        project_root=tiny_project,
        lookback_grid=(5, 10),
        risk_free_grid=(0.0,),
    )

    assert output_path == tiny_project / "reports/metrics/sensitivity.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    # Lookbacks: 5, 10 and the configured default 756 (too long for 79 returns).
    # Risk-free: 0.0 (also the configured value). Modes: contribution_only, full_rebalance.
    assert payload["schema_version"] == "1.0"
    assert payload["configured"] == {
        "lookback_periods": 756,
        "risk_free_rate": 0.0,
        "rebalance_mode": "contribution_only",
    }
    runs = payload["runs"]
    assert len(runs) == 6
    ok_runs = [run for run in runs if run["status"] == "ok"]
    skipped_runs = [run for run in runs if run["status"] == "insufficient_history"]
    assert len(ok_runs) == 4
    assert len(skipped_runs) == 2
    assert {run["lookback_periods"] for run in skipped_runs} == {756}
    assert {run["rebalance_mode"] for run in ok_runs} == {"contribution_only", "full_rebalance"}

    for run in ok_runs:
        assert run["evaluation_start_date"] == payload["common_start_date"]
        assert run["sharpe_minus_benchmark"] == pytest.approx(
            run["sharpe_ratio"] - run["benchmark_sharpe_ratio"]
        )
    for run in skipped_runs:
        assert run["sharpe_ratio"] is None

    summary = payload["summary"]
    assert summary["run_count"] == 6
    assert summary["ok_count"] == 4
    assert summary["sharpe_min"] <= summary["sharpe_max"]
    assert 0.0 <= summary["share_of_runs_beating_benchmark"] <= 1.0


def test_sensitivity_fails_clearly_when_no_combination_can_run(
    tiny_project: Path, synthetic_prices: pd.DataFrame
) -> None:
    _write_processed_data(tiny_project, synthetic_prices)

    with pytest.raises(InsufficientHistoryError, match="No sensitivity run"):
        cli.run_sensitivity(
            "configs/base.yaml",
            project_root=tiny_project,
            lookback_grid=(5000,),
            risk_free_grid=(0.0,),
        )


def test_sensitivity_command_is_wired_into_the_cli(
    tiny_project: Path, synthetic_prices: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_processed_data(tiny_project, synthetic_prices)
    monkeypatch.setattr(cli, "SENSITIVITY_LOOKBACK_PERIODS", (5,))
    monkeypatch.setattr(cli, "SENSITIVITY_RISK_FREE_RATES", (0.0,))

    assert cli.main(["sensitivity", "--config", "configs/base.yaml"]) == 0
    assert (tiny_project / "reports/metrics/sensitivity.json").exists()
