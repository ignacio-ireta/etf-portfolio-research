"""Append a research-log entry built from the latest backtest outputs.

The entry is assembled from files the pipeline already wrote, so the log cannot
drift from the artifacts:

    reports/metrics/backtest_metrics.json   (required)
    the run record named inside that file   (required)
    reports/metrics/sensitivity.json        (optional)

Usage (from the repository root):

    uv run python scripts/append_research_log_entry.py --hypothesis "One sentence."

`--decision auto` (the default) applies a fixed rule to the bootstrap interval of the
Sharpe difference versus the primary benchmark: interval above zero -> adopt, interval
below zero -> reject, anything else -> revisit later.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

METRICS_PATH = Path("reports/metrics/backtest_metrics.json")
SENSITIVITY_PATH = Path("reports/metrics/sensitivity.json")
RESEARCH_LOG_PATH = Path("docs/research_log.md")
PRIMARY_BENCHMARK_LABEL = "Selected Benchmark ETF"
OPTIMIZER_TARGET_LABEL = "Optimizer Target (Full Rebalance)"
DECISIONS = ("adopt", "reject", "revisit later")


def resolve_decision(decision: str, metrics: dict[str, Any]) -> tuple[str, str]:
    """Return (decision, reason). `auto` applies the documented interval rule."""

    if decision != "auto":
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS} or 'auto'.")
        return decision, "Decision set by the person who ran the experiment."

    interval = metrics.get("sharpe_difference_vs_benchmarks", {}).get(PRIMARY_BENCHMARK_LABEL, {})
    ci_lower, ci_upper = interval.get("ci_lower"), interval.get("ci_upper")
    if ci_lower is None or ci_upper is None:
        return "revisit later", "No usable confidence interval versus the primary benchmark."
    if ci_lower > 0.0:
        return "adopt", (
            "The Sharpe advantage over the primary benchmark is unlikely to be sampling "
            "noise alone. This is not evidence about the future."
        )
    if ci_upper < 0.0:
        return "reject", "The strategy's Sharpe ratio is below the primary benchmark's."
    return "revisit later", (
        "The Sharpe difference versus the primary benchmark is not distinguishable from "
        "sampling noise. Do not describe this run as outperformance."
    )


def _interval_text(interval: dict[str, Any] | None) -> str:
    if not interval or interval.get("sharpe_difference") is None:
        return "not available"
    ci_lower, ci_upper = float(interval["ci_lower"]), float(interval["ci_upper"])
    verdict = "contains zero" if ci_lower <= 0.0 <= ci_upper else "excludes zero"
    return (
        f"{float(interval['sharpe_difference']):+.2f}, "
        f"{float(interval['confidence_level']):.0%} bootstrap interval "
        f"[{ci_lower:+.2f}, {ci_upper:+.2f}] ({verdict})"
    )


def build_entry(
    metrics: dict[str, Any],
    run_record: dict[str, Any],
    sensitivity: dict[str, Any] | None,
    *,
    hypothesis: str,
    decision: str,
    entry_date: date,
) -> str:
    """Render one research-log entry in the format of docs/research_log.md."""

    resolved_decision, reason = resolve_decision(decision, metrics)
    strategy = metrics["optimized_strategy"]
    benchmarks = metrics.get("benchmarks", {})
    intervals = metrics.get("sharpe_difference_vs_benchmarks", {})
    gap = metrics.get("implementation_gap")

    result_lines = [
        (
            f"- Optimized Strategy: CAGR {float(strategy['CAGR']):.2%}, volatility "
            f"{float(strategy['Annualized Volatility']):.2%}, Sharpe "
            f"{float(strategy['Sharpe Ratio']):.2f}, max drawdown "
            f"{float(strategy['Max Drawdown']):.2%}"
        )
    ]
    primary = benchmarks.get(PRIMARY_BENCHMARK_LABEL)
    if primary is not None:
        result_lines.append(
            f"- Primary benchmark ({run_record.get('benchmark', 'n/a')}): CAGR "
            f"{float(primary['CAGR']):.2%}, Sharpe {float(primary['Sharpe Ratio']):.2f}"
        )
    result_lines.append(
        "- Sharpe difference versus primary benchmark: "
        f"{_interval_text(intervals.get(PRIMARY_BENCHMARK_LABEL))}"
    )
    target = benchmarks.get(OPTIMIZER_TARGET_LABEL)
    if target is not None:
        result_lines.append(
            f"- {OPTIMIZER_TARGET_LABEL}: Sharpe {float(target['Sharpe Ratio']):.2f}; "
            "strategy minus target: "
            f"{_interval_text(intervals.get(OPTIMIZER_TARGET_LABEL))}"
        )
    if gap is not None:
        result_lines.append(
            f"- Implementation gap: mean {float(gap['mean_one_way_gap']):.1%}, max "
            f"{float(gap['max_one_way_gap']):.1%} (one-way); configured constraints broken "
            f"at {int(gap['rebalances_with_constraint_violation'])} of "
            f"{int(gap['rebalance_count'])} rebalances"
        )
    if sensitivity is not None:
        summary = sensitivity["summary"]
        result_lines.append(
            f"- Sensitivity grid ({int(summary['ok_count'])} of {int(summary['run_count'])} "
            f"runs completed, common window from {sensitivity['common_start_date']}): Sharpe "
            f"from {float(summary['sharpe_min']):.2f} to {float(summary['sharpe_max']):.2f}; "
            f"{float(summary['share_of_runs_beating_benchmark']):.0%} of runs above the "
            "primary benchmark"
        )
    else:
        result_lines.append("- Sensitivity grid: not run")
    result_lines.append(
        "- Artifacts: `reports/metrics/backtest_metrics.json`, "
        "`reports/html/latest_report.html`"
        + (", `reports/metrics/sensitivity.json`" if sensitivity is not None else "")
    )

    lookback = run_record.get("lookback_periods", "n/a")
    configured_start = run_record.get("configured_backtest_start_date") or "not set"
    lines = [
        "",
        "### Date",
        "",
        f"`{entry_date.isoformat()}`",
        "",
        "### Hypothesis",
        "",
        hypothesis.strip(),
        "",
        "### Config",
        "",
        f"- run ID `{metrics['run_id']}` (run record: `{metrics['run_record']}`)",
        (
            f"- config hash `{run_record.get('config_hash', 'n/a')}`, git commit "
            f"`{run_record.get('git_commit_hash', 'n/a')}`"
        ),
        (
            f"- objective `{run_record.get('optimization_method', 'n/a')}`, lookback "
            f"{lookback} observations, simulated window {run_record.get('start_date')} to "
            f"{run_record.get('end_date')} (configured start {configured_start})"
        ),
        "",
        "### Result",
        "",
        *result_lines,
        "",
        "### Decision",
        "",
        resolved_decision,
        "",
        f"Reason: {reason}",
        "",
    ]
    return "\n".join(lines)


def append_entry(project_root: Path, *, hypothesis: str, decision: str, entry_date: date) -> str:
    """Build the entry from files under `project_root` and append it to the research log."""

    metrics = json.loads((project_root / METRICS_PATH).read_text(encoding="utf-8"))
    run_record = json.loads((project_root / metrics["run_record"]).read_text(encoding="utf-8"))
    sensitivity_path = project_root / SENSITIVITY_PATH
    sensitivity = (
        json.loads(sensitivity_path.read_text(encoding="utf-8"))
        if sensitivity_path.exists()
        else None
    )

    log_path = project_root / RESEARCH_LOG_PATH
    existing = log_path.read_text(encoding="utf-8")
    if f"`{metrics['run_id']}`" in existing:
        raise RuntimeError(
            f"docs/research_log.md already has an entry for run {metrics['run_id']}. "
            "Nothing was appended."
        )

    entry = build_entry(
        metrics,
        run_record,
        sensitivity,
        hypothesis=hypothesis,
        decision=decision,
        entry_date=entry_date,
    )
    separator = "" if existing.endswith("\n") else "\n"
    log_path.write_text(existing + separator + entry, encoding="utf-8")
    return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hypothesis", required=True, help="One sentence: what was tested.")
    parser.add_argument("--decision", default="auto", choices=("auto", *DECISIONS))
    args = parser.parse_args(argv)

    try:
        entry = append_entry(
            Path.cwd(),
            hypothesis=args.hypothesis,
            decision=args.decision,
            entry_date=datetime.now(UTC).date(),
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(entry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
