# Plan 09 — Add a script that writes research-log entries from the pipeline outputs

- **Depends on:** Plan 08 must be committed
- **Branch:** `fix/audit-remediation`
- **Estimated size:** 1 script created, 1 test file created, 3 doc files edited

## 0. Rules for whoever executes this plan

You are executing one plan out of a numbered series. Follow these rules exactly.

1. Do the steps in order, top to bottom. Do not skip a step. Do not reorder steps.
2. Change ONLY what a step tells you to change. Do not "improve", rename, reformat, or
   clean up anything else, even if it looks wrong.
3. Every command is run from the repository root (the folder that contains `pyproject.toml`).
4. Use `uv run ...` for every Python command. Never use `pip`, `python -m venv`, or `conda`.
5. A FIND block must match the file **character for character**, including spaces at the
   start of each line. Each FIND block occurs exactly once in its file. If you cannot find
   it, STOP and read section "If something goes wrong". Do not guess.
6. Never edit files under `legacy/`, `data/`, `notebooks/`, or `uv.lock` unless a step names
   that exact file.
7. Never run `git push`. Never run `git reset --hard` or `git clean` unless a step tells you to.
8. If a verification command does not give the expected result, STOP. Do not move to the
   next plan. Report the command and its full output.

## 1. Goal

After this plan:

1. `scripts/append_research_log_entry.py` exists. It reads the latest backtest metrics, the
   run record and (if present) the sensitivity grid, and appends a complete entry to
   `docs/research_log.md`.
2. `docs/research_log.md` explains how to use it.
3. `docs/guided_examples.md` no longer quotes a stale start date.
4. `CHANGELOG.md` lists the audit remediation changes.

This plan does NOT run the pipeline and does NOT add a research-log entry. Plan 10 does.

## 2. Background (why this change exists)

Audit finding: `docs/research_log.md` says every material experiment must be recorded "to
prevent undocumented parameter drift and post-hoc storytelling". It contains one entry, and
that entry is about writing documentation. Seven backtest run records were committed with
no hypothesis and no decision.

A log that must be typed by hand from memory does not get written. This plan makes an
entry one command: the numbers come straight from the JSON files the pipeline wrote, so
they cannot be mistyped or embellished.

The decision rule for `--decision auto` is fixed and simple, on purpose. It looks only at
the 95% bootstrap interval of `Sharpe(strategy) - Sharpe(primary benchmark)` from Plan 06:

- whole interval above zero -> `adopt`
- whole interval below zero -> `reject`
- anything else (including "contains zero") -> `revisit later`

The script refuses to append a second entry for the same run ID.

## 3. Files this plan touches

- `scripts/append_research_log_entry.py` (create)
- `tests/unit/test_append_research_log_entry.py` (create)
- `docs/research_log.md` (edit)
- `docs/guided_examples.md` (edit)
- `CHANGELOG.md` (edit)

No other file may change. If `git status` shows any other modified file at the end, you made a mistake.

## 4. Before you start

Run each command. Compare with the expected result. If one differs, STOP.

````bash
git status --short
````

Expected: Prints nothing (the working tree is clean).

````bash
git branch --show-current
````

Expected: `fix/audit-remediation`

````bash
git log --oneline | head -1
````

Expected: The line contains `drop unused pyportfolioopt` (Plan 08's commit).

## 5. Fast path — apply the prepared patch

The patch file contains every edit in this plan, already tested. Try it first.

````bash
git apply --check plans/patches/09-research-log-tooling.patch
````

- If this prints **nothing**, the patch fits. Run:

````bash
git apply plans/patches/09-research-log-tooling.patch
````

  Then jump to section 7 (Verify).

- If it prints an **error**, do not force it. Use section 6 (manual steps) instead.

## 6. Manual steps (only if the fast path failed)

### Step 1 — create new file `scripts/append_research_log_entry.py`

The new script.

Create the file `scripts/append_research_log_entry.py` with exactly this content (the file must end with one newline):

````python
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
````

### Step 2 — create new file `tests/unit/test_append_research_log_entry.py`

Tests for the script. They use small fake JSON files, not real market data.

Create the file `tests/unit/test_append_research_log_entry.py` with exactly this content (the file must end with one newline):

````python
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
````

### Step 3 — edit `docs/research_log.md`

Tell readers how entries are produced.

Open `docs/research_log.md`. FIND this exact block (it occurs exactly once):

````markdown
Every material experiment should be recorded here. The purpose is to prevent undocumented parameter drift and post-hoc storytelling.
````

REPLACE the whole block with:

````markdown
Every material experiment should be recorded here. The purpose is to prevent undocumented parameter drift and post-hoc storytelling.

## How To Add An Entry

Do not type results by hand. After a backtest (and, ideally, a sensitivity run), let the
script build the entry from the files the pipeline wrote:

```bash
uv run etf-portfolio backtest --config configs/base.yaml
uv run etf-portfolio sensitivity --config configs/base.yaml
uv run python scripts/append_research_log_entry.py --hypothesis "One sentence: what are you testing?"
```

Write the hypothesis **before** looking at the result. The script appends the entry to the
end of this file and refuses to add a second entry for the same run ID.

`--decision auto` (the default) uses a fixed rule on the 95% bootstrap interval of the
Sharpe difference versus the primary benchmark: interval entirely above zero gives
`adopt`, entirely below zero gives `reject`, anything else gives `revisit later`. Pass
`--decision adopt`, `--decision reject` or `--decision "revisit later"` to override it,
and say why in a follow-up line.
````

### Step 4 — edit `docs/guided_examples.md`

Replace a sentence that quotes a start date from an old run.

Open `docs/guided_examples.md`. FIND this exact block (it occurs exactly once):

````markdown
Because the walk-forward backtest needs trailing lookback data, the current run record reports an effective backtest start date of `2015-01-30`.
````

REPLACE the whole block with:

````markdown
The walk-forward backtest needs `backtest.lookback_periods` (756) trading days of history before its first rebalance. Returns before the configured start date are used for that, but the committed data starts in January 2011, so the first simulated return is in February 2014, not January 2012. The exact date is the `start_date` field of the run record and the `Backtest window` line in the report's assumptions list.
````

### Step 5 — edit `CHANGELOG.md`

Record the remediation in the changelog, at the top of the Unreleased section.

Open `CHANGELOG.md`. FIND this exact block (it occurs exactly once):

````markdown
## [Unreleased]
````

REPLACE the whole block with:

````markdown
## [Unreleased]

### Audit remediation

Changes from the repository audit (step-by-step plans in `plans/`). Backtest numbers
change because of the first two items; regenerate reports before quoting any figure.

- **Changed:** the estimation lookback is now `backtest.lookback_periods` in the config
  (default 756) and is stored in every backtest run record; `--lookback-periods` is an
  optional override. Returns before `backtest.start_date` are used to warm up the
  lookback instead of being discarded.
- **Fixed:** daily returns between rebalance dates are computed as buy-and-hold,
  consistent with the weight drift, instead of as a costless daily reset.
- **Fixed:** the optimizer retries SLSQP on a tolerance ladder, so one non-converged
  month no longer aborts a whole backtest (`optimizer_tolerance_relaxed` warning).
- **Changed:** fixed-allocation benchmarks (60/40 and other `benchmark.secondary`
  entries) are simulated on the strategy's rebalance schedule and pay the same cost
  rate, instead of being rebalanced daily for free.
- **Added:** `implementation_gap` in the metrics JSON and report assumptions, plus the
  benchmark row `Optimizer Target (Full Rebalance)`.
- **Added:** `sharpe_difference_vs_benchmarks` (stationary-bootstrap confidence
  intervals) in the metrics JSON and a `Statistical uncertainty:` report assumption.
- **Added:** `etf-portfolio sensitivity` command writing `reports/metrics/sensitivity.json`.
- **Added:** `scripts/append_research_log_entry.py` to build research-log entries from
  the pipeline's own outputs.
- **Removed:** unused `pyportfolioopt` dependency.
- **Changed:** `investor_profile` is optional and documented as descriptive only.
````

## 7. Verify

Run every command below, in order. All must match the expected result.

````bash
uv run ruff check .
````

Expected: `All checks passed!`

````bash
uv run ruff format --check .
````

Expected: A line ending in `files already formatted`.

````bash
uv run pytest -q tests/unit/test_append_research_log_entry.py
````

Expected: `8 passed`.

````bash
uv run python scripts/append_research_log_entry.py --help
````

Expected: Usage text that mentions `--hypothesis` and `--decision`.

````bash
git diff --stat -- docs/research_log.md | tail -1
````

Expected: Shows insertions only (`0 deletions` or no deletions mentioned). No research-log ENTRY was added in this plan, only the `How To Add An Entry` section.

````bash
uv run pytest -q -x
````

Expected: Last line starts with `386 passed`. No `failed`, no `error`.

## 8. Commit

````bash
git add -A
git status --short
````

Expected: only the files listed in section 3 appear.

````bash
git commit -m "docs: add research-log entry script, changelog for the audit remediation"
````

## 9. Done checklist

- [ ] `scripts/append_research_log_entry.py` exists and `--help` works.
- [ ] `docs/research_log.md` has the section `How To Add An Entry`.
- [ ] `CHANGELOG.md` has the section `### Audit remediation` under `## [Unreleased]`.
- [ ] `docs/guided_examples.md` no longer contains `2015-01-30`.
- [ ] The full test suite passes.
- [ ] `git status --short` prints nothing.
- [ ] You did not run `git push`.

## 10. If something goes wrong

- **A FIND block is not in the file.** Check you are on the right branch and that every earlier plan is committed (`git log --oneline | head`). Search for the first line of the FIND block only (`grep -n "<first line>" <file>`). If the surrounding code is different from the plan, STOP and report; do not improvise.
- **`ruff format --check` reports a file would be reformatted.** Run `uv run ruff format <that file>` once, then re-run the verify commands.
- **You want to start this plan over.** Run `git checkout -- . && git clean -fd -- src tests docs configs` (this discards ONLY uncommitted work of this plan), then start again from section 4.
- **The `CHANGELOG.md` FIND block (`## [Unreleased]` followed by an empty line) is not found.** Open `CHANGELOG.md`, find the line `## [Unreleased]`, and paste the `### Audit remediation` block two lines below it, keeping one empty line before and after the block.
