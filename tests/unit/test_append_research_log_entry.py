"""Tests for scripts/append_research_log_entry.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pytest

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "append_research_log_entry.py"
SPEC = importlib.util.spec_from_file_location("append_research_log_entry", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
research_log = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = research_log
SPEC.loader.exec_module(research_log)


def _interval(ci_lower: float, ci_upper: float) -> dict[str, Any]:
    return {
        "sharpe_difference": (ci_lower + ci_upper) / 2.0,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence_level": 0.95,
    }


def _metrics(ci_lower: float = -0.06, ci_upper: float = 0.36) -> dict[str, Any]:
    row = {"CAGR": 0.10, "Annualized Volatility": 0.11, "Sharpe Ratio": 0.66, "Max Drawdown": -0.25}
    return {
        "run_id": "backtest-20260101T000000Z-abcdef12",
        "run_record": "reports/runs/backtest_backtest-20260101T000000Z-abcdef12.json",
        "optimized_strategy": row,
        "benchmarks": {
            "Selected Benchmark ETF": {**row, "Sharpe Ratio": 0.51},
            "Optimizer Target (Full Rebalance)": {**row, "Sharpe Ratio": 0.60},
        },
        "sharpe_difference_vs_benchmarks": {
            "Selected Benchmark ETF": _interval(ci_lower, ci_upper),
            "Optimizer Target (Full Rebalance)": _interval(-0.06, 0.22),
        },
        "implementation_gap": {
            "rebalance_count": 147,
            "mean_one_way_gap": 0.261,
            "max_one_way_gap": 0.469,
            "latest_one_way_gap": 0.31,
            "rebalances_with_constraint_violation": 146,
        },
    }


RUN_RECORD = {
    "benchmark": "VT",
    "config_hash": "c" * 64,
    "git_commit_hash": "g" * 40,
    "optimization_method": "max_sharpe",
    "lookback_periods": 756,
    "start_date": "2014-02-03",
    "end_date": "2026-04-24",
    "configured_backtest_start_date": "2012-01-01",
}

SENSITIVITY = {
    "common_start_date": "2016-02-01",
    "summary": {
        "run_count": 16,
        "ok_count": 16,
        "sharpe_min": 0.67,
        "sharpe_max": 1.04,
        "share_of_runs_beating_benchmark": 1.0,
    },
}


def _write_project(project_root: Path, metrics: dict[str, Any], *, sensitivity: bool) -> None:
    (project_root / "reports/metrics").mkdir(parents=True)
    (project_root / "reports/runs").mkdir(parents=True)
    (project_root / "docs").mkdir(parents=True)
    (project_root / "reports/metrics/backtest_metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )
    (project_root / metrics["run_record"]).write_text(json.dumps(RUN_RECORD), encoding="utf-8")
    if sensitivity:
        (project_root / "reports/metrics/sensitivity.json").write_text(
            json.dumps(SENSITIVITY), encoding="utf-8"
        )
    (project_root / "docs/research_log.md").write_text("# Research Log\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("ci_lower", "ci_upper", "expected"),
    [
        (-0.06, 0.36, "revisit later"),
        (0.05, 0.36, "adopt"),
        (-0.40, -0.05, "reject"),
    ],
)
def test_auto_decision_follows_the_interval_rule(
    ci_lower: float, ci_upper: float, expected: str
) -> None:
    decision, reason = research_log.resolve_decision("auto", _metrics(ci_lower, ci_upper))

    assert decision == expected
    assert reason


def test_auto_decision_without_an_interval_is_revisit_later() -> None:
    metrics = _metrics()
    metrics["sharpe_difference_vs_benchmarks"] = {}

    assert research_log.resolve_decision("auto", metrics)[0] == "revisit later"


def test_explicit_decision_is_kept() -> None:
    assert research_log.resolve_decision("reject", _metrics())[0] == "reject"


def test_entry_contains_every_template_heading_and_the_key_numbers() -> None:
    entry = research_log.build_entry(
        _metrics(),
        RUN_RECORD,
        SENSITIVITY,
        hypothesis="Does the strategy beat VT after the audit fixes?",
        decision="auto",
        entry_date=date(2026, 11, 14),
    )

    for heading in ("### Date", "### Hypothesis", "### Config", "### Result", "### Decision"):
        assert heading in entry
    assert "`2026-11-14`" in entry
    assert "`backtest-20260101T000000Z-abcdef12`" in entry
    assert "lookback 756 observations" in entry
    assert "2014-02-03 to 2026-04-24 (configured start 2012-01-01)" in entry
    assert "[-0.06, +0.36] (contains zero)" in entry
    assert "146 of 147 rebalances" in entry
    assert "Sharpe from 0.67 to 1.04" in entry
    assert "revisit later" in entry


def test_append_entry_writes_to_the_log_and_refuses_duplicates(tmp_path: Path) -> None:
    _write_project(tmp_path, _metrics(), sensitivity=False)

    research_log.append_entry(
        tmp_path,
        hypothesis="First run.",
        decision="auto",
        entry_date=date(2026, 11, 14),
    )

    log_text = (tmp_path / "docs/research_log.md").read_text(encoding="utf-8")
    assert log_text.startswith("# Research Log\n")
    assert "First run." in log_text
    assert "- Sensitivity grid: not run" in log_text

    with pytest.raises(RuntimeError, match="already has an entry"):
        research_log.append_entry(
            tmp_path,
            hypothesis="Same run again.",
            decision="auto",
            entry_date=date(2026, 11, 14),
        )
    assert "Same run again." not in (tmp_path / "docs/research_log.md").read_text(encoding="utf-8")


def test_main_reports_missing_metrics_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = research_log.main(["--hypothesis", "Nothing to read."])

    assert exit_code == 1
    assert "ERROR" in capsys.readouterr().err
