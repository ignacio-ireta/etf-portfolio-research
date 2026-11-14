# Plan 05 — Report the gap between the realized portfolio and the optimizer target

- **Depends on:** Plan 04 must be committed
- **Branch:** `fix/audit-remediation`
- **Estimated size:** 2 source files, 3 doc files and 1 test file edited, 1 test file created

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

After this plan every backtest run:

1. Computes an **implementation gap** summary: how far the realized holdings were from the
   optimizer target (average, maximum, latest), and at how many rebalances the realized
   holdings broke a configured constraint.
2. Writes that summary to `reports/metrics/backtest_metrics.json` under the key
   `implementation_gap`.
3. Shows it in the report's assumptions list (line starting with `Implementation gap:`).
4. Logs a WARNING event `implementation_gap_large` when the average gap is above 10%.
5. Adds a benchmark row **Optimizer Target (Full Rebalance)**: the same optimizer, same
   costs, but allowed to sell so it actually reaches its target. This row is added only when
   `rebalance.mode` is not already `full_rebalance`.

The label `Optimized Strategy` and the JSON key `optimized_strategy` are NOT renamed.

This plan deliberately does NOT edit `src/etf_portfolio/reporting/plots.py`. The charts pick a
colour for the new benchmark row automatically.

## 2. Background (why this change exists)

Audit finding: the headline row "Optimized Strategy" is not the optimizer's portfolio. With
`rebalance.mode: contribution_only`, $1,500 a month cannot steer a $100k+ portfolio:

- realized weights were on average 27% (one-way) away from the optimizer target;
- realized holdings broke their own configured constraints at 134 of 135 rebalances;
- following the optimizer exactly (`full_rebalance`) gave a **lower** Sharpe ratio than the
  drifting portfolio.

So the published advantage came from drift, not from optimization, and nothing in the
report said so. This plan does not change the strategy. It makes the gap impossible to miss.

Why the label is not renamed: `Optimized Strategy` / `optimized_strategy` are part of the
output contract and are referenced in about 40 places (golden file, metric dictionary,
plots, tests). Renaming is a large mechanical change with no added information. Adding the
gap numbers and the full-rebalance row gives the reader the facts directly.

Definition used: the **one-way gap** at one rebalance is
`sum(|realized weight - target weight|) / 2`, measured right after the rebalance trade. It
is the share of the portfolio that would have to be sold (and the same share bought) to
match the target. 0% means "identical", 100% means "nothing in common".

## 3. Files this plan touches

- `src/etf_portfolio/backtesting/metrics.py` (edit)
- `src/etf_portfolio/cli.py` (edit)
- `tests/unit/test_backtest_interpretation.py` (edit)
- `tests/unit/test_implementation_gap.py` (create)
- `docs/output_contract.md` (edit)
- `README.md` (edit)
- `docs/how_to_read_the_report.md` (append to)

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

Expected: The line contains `simulate fixed-allocation benchmarks` (Plan 04's commit).

## 5. Fast path — apply the prepared patch

The patch file contains every edit in this plan, already tested. Try it first.

````bash
git apply --check plans/patches/05-report-implementation-gap.patch
````

- If this prints **nothing**, the patch fits. Run:

````bash
git apply plans/patches/05-report-implementation-gap.patch
````

  Then jump to section 7 (Verify).

- If it prints an **error**, do not force it. Use section 6 (manual steps) instead.

## 6. Manual steps (only if the fast path failed)

### Step 1 — edit `src/etf_portfolio/backtesting/metrics.py`

Add the summary function, directly above `summarize_backtest_metrics`.

Open `src/etf_portfolio/backtesting/metrics.py`. FIND this exact block (it occurs exactly once):

````python
def summarize_backtest_metrics(
    portfolio_returns: pd.Series,
````

REPLACE the whole block with:

````python
def summarize_implementation_gap(
    target_weights: pd.DataFrame,
    applied_weights: pd.DataFrame,
    realized_constraint_violations: pd.DataFrame | None = None,
) -> dict[str, float | int]:
    """Summarize how far realized holdings sat from the optimizer target.

    The one-way gap at a rebalance is half the sum of absolute weight differences:
    the share of the portfolio that would have to be traded to match the target.
    Assets missing from one frame count as zero weight there.
    """

    if target_weights.empty or applied_weights.empty:
        gaps = pd.Series(dtype=float)
    else:
        columns = target_weights.columns.union(applied_weights.columns)
        index = target_weights.index.intersection(applied_weights.index)
        target = target_weights.reindex(index=index, columns=columns).fillna(0.0)
        applied = applied_weights.reindex(index=index, columns=columns).fillna(0.0)
        gaps = (applied - target).abs().sum(axis=1) / 2.0

    violation_rebalances = 0
    if (
        realized_constraint_violations is not None
        and not realized_constraint_violations.empty
        and "rebalance_date" in realized_constraint_violations.columns
    ):
        violation_rebalances = int(realized_constraint_violations["rebalance_date"].nunique())

    rebalance_count = len(gaps)
    return {
        "rebalance_count": rebalance_count,
        "mean_one_way_gap": float(gaps.mean()) if rebalance_count else 0.0,
        "max_one_way_gap": float(gaps.max()) if rebalance_count else 0.0,
        "latest_one_way_gap": float(gaps.iloc[-1]) if rebalance_count else 0.0,
        "rebalances_with_constraint_violation": violation_rebalances,
    }


def summarize_backtest_metrics(
    portfolio_returns: pd.Series,
````

### Step 2 — edit `src/etf_portfolio/backtesting/metrics.py`

Export the new function (the `__all__` list is alphabetical).

Open `src/etf_portfolio/backtesting/metrics.py`. FIND this exact block (it occurs exactly once):

````python
    "summarize_backtest_metrics",
    "tracking_error",
````

REPLACE the whole block with:

````python
    "summarize_backtest_metrics",
    "summarize_implementation_gap",
    "tracking_error",
````

### Step 3 — edit `src/etf_portfolio/cli.py`

Import the new function.

Open `src/etf_portfolio/cli.py`. FIND this exact block (it occurs exactly once):

````python
from etf_portfolio.backtesting.metrics import calculate_sharpe_ratio
````

REPLACE the whole block with:

````python
from etf_portfolio.backtesting.metrics import (
    calculate_sharpe_ratio,
    summarize_implementation_gap,
)
````

### Step 4 — edit `src/etf_portfolio/cli.py`

Add two constants: the new benchmark label and the warning threshold.

Open `src/etf_portfolio/cli.py`. FIND this exact block (it occurs exactly once):

````python
BACKTEST_START_DELAY_WARNING_DAYS = 45
````

REPLACE the whole block with:

````python
BACKTEST_START_DELAY_WARNING_DAYS = 45
OPTIMIZER_TARGET_LABEL = "Optimizer Target (Full Rebalance)"
IMPLEMENTATION_GAP_WARNING_THRESHOLD = 0.10
````

### Step 5 — edit `src/etf_portfolio/cli.py`

`run_backtest`: compute the gap summary and warn when it is large.

Open `src/etf_portfolio/cli.py`. FIND this exact block (it occurs exactly once):

````python
    in_sample_returns = asset_returns.loc[asset_returns.index < backtest_result.weights.index[-1]]
````

REPLACE the whole block with:

````python
    gap_summary = summarize_implementation_gap(
        backtest_result.target_weights,
        backtest_result.applied_weights,
        backtest_result.realized_constraint_violations,
    )
    if float(gap_summary["mean_one_way_gap"]) > IMPLEMENTATION_GAP_WARNING_THRESHOLD:
        log_event(
            LOGGER,
            logging.WARNING,
            "implementation_gap_large",
            run_id=stage_run_id,
            stage="backtest",
            config_path=str(config_path),
            rebalance_mode=config.rebalance.mode,
            mean_one_way_gap=float(gap_summary["mean_one_way_gap"]),
            max_one_way_gap=float(gap_summary["max_one_way_gap"]),
            rebalances_with_constraint_violation=int(
                gap_summary["rebalances_with_constraint_violation"]
            ),
            rebalance_count=int(gap_summary["rebalance_count"]),
            reason=(
                "Realized holdings are far from the optimizer target; the simulated "
                "portfolio is not the optimizer's portfolio."
            ),
        )
    in_sample_returns = asset_returns.loc[asset_returns.index < backtest_result.weights.index[-1]]
````

### Step 6 — edit `src/etf_portfolio/cli.py`

`run_backtest`: pass the gap summary into the report assumptions.

Open `src/etf_portfolio/cli.py`. FIND this exact block (it occurs exactly once):

````python
            lookback_periods=effective_lookback,
            effective_start_date=effective_start_date,
        ),
````

REPLACE the whole block with:

````python
            lookback_periods=effective_lookback,
            effective_start_date=effective_start_date,
            implementation_gap=gap_summary,
        ),
````

### Step 7 — edit `src/etf_portfolio/cli.py`

`run_backtest`: write the gap summary into the metrics JSON.

Open `src/etf_portfolio/cli.py`. FIND this exact block (it occurs exactly once):

````python
        "optimized_strategy": strategy_metrics_payload,
        "benchmarks": metrics_by_strategy,
    }
````

REPLACE the whole block with:

````python
        "optimized_strategy": strategy_metrics_payload,
        "benchmarks": metrics_by_strategy,
        "implementation_gap": gap_summary,
    }
````

### Step 8 — edit `src/etf_portfolio/cli.py`

`_report_assumptions`: accept the gap summary.

Open `src/etf_portfolio/cli.py`. FIND this exact block (it occurs exactly once):

````python
    lookback_periods: int | None = None,
    effective_start_date: pd.Timestamp | None = None,
) -> dict[str, str]:
````

REPLACE the whole block with:

````python
    lookback_periods: int | None = None,
    effective_start_date: pd.Timestamp | None = None,
    implementation_gap: dict[str, float | int] | None = None,
) -> dict[str, str]:
````

### Step 9 — edit `src/etf_portfolio/cli.py`

`_report_assumptions`: add the `implementation_gap` line.

Open `src/etf_portfolio/cli.py`. FIND this exact block (it occurs exactly once):

````python
            "configured start, the data history was too short to fill the lookback."
        )
    if run_id is not None:
````

REPLACE the whole block with:

````python
            "configured start, the data history was too short to fill the lookback."
        )
    if implementation_gap is not None:
        gap_text = (
            "Implementation gap: right after each rebalance, realized holdings differed "
            "from the optimizer target by "
            f"{float(implementation_gap['mean_one_way_gap']):.1%} of the portfolio on "
            f"average (one-way; maximum {float(implementation_gap['max_one_way_gap']):.1%}). "
            "Realized holdings broke at least one configured constraint at "
            f"{int(implementation_gap['rebalances_with_constraint_violation'])} of "
            f"{int(implementation_gap['rebalance_count'])} rebalances."
        )
        if config.rebalance.mode != "full_rebalance":
            gap_text += (
                f" 'Optimized Strategy' is the realized {config.rebalance.mode} portfolio, "
                f"not the optimizer target; the row '{OPTIMIZER_TARGET_LABEL}' shows the "
                "same optimizer followed exactly."
            )
        assumptions["implementation_gap"] = gap_text
    if run_id is not None:
````

### Step 10 — edit `src/etf_portfolio/cli.py`

`_build_benchmark_suite`: add the full-rebalance run of the active objective. It uses the same costs, constraints, estimators and schedule as the strategy; only `rebalance_mode` differs.

Open `src/etf_portfolio/cli.py`. FIND this exact block (it occurs exactly once):

````python
    suite["Previous Optimized Strategy"] = previous_result.portfolio_returns
    benchmark_weights["Previous Optimized Strategy"] = previous_result.weights
    return BenchmarkSuiteArtifacts(returns=suite, weights=benchmark_weights)
````

REPLACE the whole block with:

````python
    suite["Previous Optimized Strategy"] = previous_result.portfolio_returns
    benchmark_weights["Previous Optimized Strategy"] = previous_result.weights
    if config.rebalance.mode != "full_rebalance":
        target_result = run_walk_forward_backtest(
            asset_returns,
            rebalance_dates=rebalance_dates,
            lookback_periods=lookback_periods,
            optimization_method=_select_optimization_method(config),
            max_weight=config.optimization.default_max_weight_per_etf,
            transaction_cost_rate=transaction_cost_rate,
            asset_classes=asset_classes,
            asset_class_bounds=asset_class_bounds,
            ticker_bounds=ticker_bounds,
            bond_assets=bond_assets,
            min_bond_exposure=min_bond_exposure,
            expense_ratios=expense_ratios,
            risk_free_rate=risk_free_rate,
            rebalance_mode="full_rebalance",
            contribution_amount=config.rebalance.contribution_amount,
            initial_capital=config.backtest.initial_capital,
            realized_constraint_policy=config.rebalance.realized_constraint_policy,
            covariance_method=config.optimization.risk_model,
            expected_return_method=config.optimization.expected_return_estimator,
        )
        suite[OPTIMIZER_TARGET_LABEL] = target_result.portfolio_returns
        benchmark_weights[OPTIMIZER_TARGET_LABEL] = target_result.weights
    return BenchmarkSuiteArtifacts(returns=suite, weights=benchmark_weights)
````

### Step 11 — edit `tests/unit/test_backtest_interpretation.py`

This existing test counts the engine calls made by `_build_benchmark_suite`. There is now one more call (the full-rebalance target). Update the expected count, methods and modes.

Open `tests/unit/test_backtest_interpretation.py`. FIND this exact block (it occurs exactly once):

````python
    # Should have called for 'min_variance' and 'Previous Optimized Strategy'
    assert len(captured_kwargs) == 2
    assert [kwargs["optimization_method"] for kwargs in captured_kwargs] == [
        "min_volatility",
        "equal_weight",
    ]
    assert all(kwargs["rebalance_mode"] == "contribution_only" for kwargs in captured_kwargs)
````

REPLACE the whole block with:

````python
    # Should have called for 'min_variance', 'Previous Optimized Strategy', and the
    # full-rebalance optimizer target.
    assert len(captured_kwargs) == 3
    assert [kwargs["optimization_method"] for kwargs in captured_kwargs] == [
        "min_volatility",
        "equal_weight",
        "equal_weight",
    ]
    assert [kwargs["rebalance_mode"] for kwargs in captured_kwargs] == [
        "contribution_only",
        "contribution_only",
        "full_rebalance",
    ]
````

### Step 12 — create new file `tests/unit/test_implementation_gap.py`

New tests.

Create the file `tests/unit/test_implementation_gap.py` with exactly this content (the file must end with one newline):

````python
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
````

### Step 13 — edit `docs/output_contract.md`

Document the new metrics JSON key.

Open `docs/output_contract.md`. FIND this exact block (it occurs exactly once):

````markdown
`benchmarks` (name → metrics), `run_record` (relative path) |
````

REPLACE the whole block with:

````markdown
`benchmarks` (name → metrics), `implementation_gap` (`rebalance_count`, `mean_one_way_gap`, `max_one_way_gap`, `latest_one_way_gap`, `rebalances_with_constraint_violation`), `run_record` (relative path) |
````

### Step 14 — edit `README.md`

README: explain what the headline row is.

Open `README.md`. FIND this exact block (it occurs exactly once):

````markdown
- Backtest reporting separates optimizer targets from realized holdings.
````

REPLACE the whole block with:

````markdown
- `Optimized Strategy` is the realized portfolio under `rebalance.mode`, not the optimizer target. Each run reports the implementation gap (average and maximum one-way distance between realized and target weights, and how many rebalances broke a configured constraint) in the report assumptions and in `reports/metrics/backtest_metrics.json` under `implementation_gap`. When the mode is not `full_rebalance`, the benchmark row `Optimizer Target (Full Rebalance)` shows the same optimizer followed exactly.
- Backtest reporting separates optimizer targets from realized holdings.
````

### Step 15 — append to the end of `docs/how_to_read_the_report.md`

Add a reader-facing section.

Go to the very end of `docs/how_to_read_the_report.md`. Do not delete anything. Add this text after the last line:

````markdown

## Implementation gap: realized holdings versus the optimizer target

The row called **Optimized Strategy** is the *realized* portfolio under the configured
rebalance mode. In `contribution_only` mode the portfolio never sells, so new cash is the
only tool for moving toward the optimizer target. When contributions are small compared
with the portfolio, realized holdings can sit far from the target for years.

The report states this in three places:

- The assumption line that starts with `Implementation gap:` gives the average and maximum
  one-way gap (the share of the portfolio that would have to be traded to match the
  target) and how many rebalances ended with at least one configured constraint broken.
- `reports/metrics/backtest_metrics.json` carries the same numbers under
  `implementation_gap`.
- The benchmark row **Optimizer Target (Full Rebalance)** shows the same optimizer followed
  exactly (sells allowed, same costs). It is present whenever `rebalance.mode` is not
  `full_rebalance`.

How to read them together: if **Optimized Strategy** beats **Optimizer Target (Full
Rebalance)**, the result came from *not* following the optimizer (drift), not from the
optimizer's choices. Do not credit the optimizer for it.
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
uv run pytest -q tests/unit/test_implementation_gap.py
````

Expected: `6 passed`.

````bash
uv run pytest -q tests/unit/test_backtest_interpretation.py
````

Expected: All pass, 0 failed.

````bash
uv run pytest -q -x
````

Expected: Last line starts with `359 passed`. No `failed`, no `error`.

## 8. Commit

````bash
git add -A
git status --short
````

Expected: only the files listed in section 3 appear.

````bash
git commit -m "feat(report): report the realized-vs-target implementation gap and the full-rebalance target"
````

## 9. Done checklist

- [ ] `summarize_implementation_gap` exists in `src/etf_portfolio/backtesting/metrics.py`.
- [ ] `OPTIMIZER_TARGET_LABEL` exists in `src/etf_portfolio/cli.py`.
- [ ] `docs/how_to_read_the_report.md` ends with the new section `Implementation gap: realized holdings versus the optimizer target`.
- [ ] The full test suite passes.
- [ ] `git status --short` prints nothing.
- [ ] You did not run `git push`.

## 10. If something goes wrong

- **A FIND block is not in the file.** Check you are on the right branch and that every earlier plan is committed (`git log --oneline | head`). Search for the first line of the FIND block only (`grep -n "<first line>" <file>`). If the surrounding code is different from the plan, STOP and report; do not improvise.
- **`ruff format --check` reports a file would be reformatted.** Run `uv run ruff format <that file>` once, then re-run the verify commands.
- **You want to start this plan over.** Run `git checkout -- . && git clean -fd -- src tests docs configs` (this discards ONLY uncommitted work of this plan), then start again from section 4.
- **A test fails with `KeyError: 'implementation_gap'`.** The step that adds `"implementation_gap": gap_summary,` to `metrics_payload` in `run_backtest` is missing.
- **`NameError: name 'gap_summary' is not defined`.** The step that inserts the `gap_summary = summarize_implementation_gap(` block before the `in_sample_returns = ...` line is missing.
