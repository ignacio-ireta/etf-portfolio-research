# Plan 07 — Add a `sensitivity` command for lookback, risk-free rate and rebalance mode

- **Depends on:** Plan 06 must be committed
- **Branch:** `fix/audit-remediation`
- **Estimated size:** 1 source file edited (about 200 new lines), 3 doc/tooling files edited, 1 test file created

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

After this plan there is a new CLI command:

    uv run etf-portfolio sensitivity --config configs/base.yaml

It re-runs the walk-forward backtest over a grid of lookbacks, risk-free rates and
rebalance modes, scores every run on one common evaluation window, and writes
`reports/metrics/sensitivity.json`. It is **not** part of `run-all`.

## 2. Background (why this change exists)

Audit finding: the published result is one run with one lookback (756), one risk-free
rate (3%) and one rebalance mode (contribution_only). Changing the lookback to 504 or 1260
or switching to full rebalancing moves the Sharpe ratio by as much as the advantage the
report shows. The project had no tool to see that.

Design decisions (do not change them):

- The grid always includes the configured values, so the published run is one of the rows.
- All successful runs are compared on the same dates. A longer lookback starts later, so
  without a common window the rows would cover different market periods.
- A run that cannot start (lookback longer than the data) is recorded with
  `status: "insufficient_history"`. A run where the optimizer raises is recorded with
  `status: "failed"`. Neither stops the other runs.
- If **no** run succeeds, the command raises `InsufficientHistoryError`.
- The command writes only `reports/metrics/sensitivity.json`.

The new function `run_sensitivity` repeats the first lines of `run_backtest` (loading and
aligning data). That duplication is deliberate: refactoring `run_backtest` is out of scope.

## 3. Files this plan touches

- `src/etf_portfolio/cli.py` (edit)
- `tests/unit/test_sensitivity_command.py` (create)
- `Makefile` (edit)
- `README.md` (edit)
- `docs/output_contract.md` (edit)
- `docs/runbook.md` (append to)

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

Expected: The line contains `add bootstrap confidence intervals` (Plan 06's commit).

## 5. Fast path — apply the prepared patch

The patch file contains every edit in this plan, already tested. Try it first.

````bash
git apply --check plans/patches/07-sensitivity-command.patch
````

- If this prints **nothing**, the patch fits. Run:

````bash
git apply plans/patches/07-sensitivity-command.patch
````

  Then jump to section 7 (Verify).

- If it prints an **error**, do not force it. Use section 6 (manual steps) instead.

## 6. Manual steps (only if the fast path failed)

### Step 1 — edit `src/etf_portfolio/cli.py`

Import three metric functions used to score each run.

Open `src/etf_portfolio/cli.py`. FIND this exact block (it occurs exactly once):

````python
from etf_portfolio.backtesting.metrics import (
    calculate_sharpe_ratio,
    summarize_implementation_gap,
)
````

REPLACE the whole block with:

````python
from etf_portfolio.backtesting.metrics import (
    cagr,
    calculate_sharpe_ratio,
    portfolio_volatility,
    sharpe_ratio,
    summarize_implementation_gap,
)
````

### Step 2 — edit `src/etf_portfolio/cli.py`

Add the default grids as module constants.

Open `src/etf_portfolio/cli.py`. FIND this exact block (it occurs exactly once):

````python
IMPLEMENTATION_GAP_WARNING_THRESHOLD = 0.10
````

REPLACE the whole block with:

````python
IMPLEMENTATION_GAP_WARNING_THRESHOLD = 0.10
# Default grids for the `sensitivity` command. The configured lookback and risk-free
# rate are always added to these.
SENSITIVITY_LOOKBACK_PERIODS: tuple[int, ...] = (252, 504, 756, 1260)
SENSITIVITY_RISK_FREE_RATES: tuple[float, ...] = (0.0,)
````

### Step 3 — edit `src/etf_portfolio/cli.py`

`main`: dispatch the new command.

Open `src/etf_portfolio/cli.py`. FIND this exact block (it occurs exactly once):

````python
        elif args.command == "ml":
            run_ml(config_path, project_root=project_root, run_id=run_id)
````

REPLACE the whole block with:

````python
        elif args.command == "ml":
            run_ml(config_path, project_root=project_root, run_id=run_id)
        elif args.command == "sensitivity":
            run_sensitivity(config_path, project_root=project_root, run_id=run_id)
````

### Step 4 — edit `src/etf_portfolio/cli.py`

CLI parser: register the new command (it takes only the common flags).

Open `src/etf_portfolio/cli.py`. FIND this exact block (it occurs exactly once):

````python
    for command_name in ("ingest", "validate", "features", "optimize", "ml"):
````

REPLACE the whole block with:

````python
    for command_name in ("ingest", "validate", "features", "optimize", "ml", "sensitivity"):
````

### Step 5 — edit `src/etf_portfolio/cli.py`

Add the new function `run_sensitivity`. It goes directly above `_default_log_file`. Note the two blank lines after the function.

Open `src/etf_portfolio/cli.py`. FIND this exact block (it occurs exactly once):

````python
def _default_log_file(args: argparse.Namespace, project_root: Path, run_id: str) -> Path | None:
````

REPLACE the whole block with:

````python
def run_sensitivity(
    config_path: str | Path,
    *,
    project_root: Path = Path("."),
    run_id: str | None = None,
    lookback_grid: tuple[int, ...] | None = None,
    risk_free_grid: tuple[float, ...] | None = None,
) -> Path:
    """Re-run the backtest over a grid of lookbacks, risk-free rates and rebalance modes.

    Every successful run is scored on the same evaluation window (starting at the
    latest first-return date across runs), so the rows are comparable. The result is
    written to reports/metrics/sensitivity.json. This command never touches the main
    report or the main metrics file.
    """

    config = _load_project_config(config_path, project_root)
    stage_run_id = run_id or generate_run_id("sensitivity")
    returns = pd.read_parquet(project_root / "data/processed/returns.parquet")
    metadata = load_etf_universe_metadata(_metadata_path(project_root)).set_index("ticker")
    asset_returns = _asset_returns(returns, config)
    benchmark_returns = _benchmark_returns(returns, config)
    common_dates = asset_returns.index.intersection(benchmark_returns.index)
    asset_returns = asset_returns.loc[common_dates]
    benchmark_returns = benchmark_returns.loc[common_dates]
    windowed_asset_returns, benchmark_returns = _apply_backtest_window(
        asset_returns,
        benchmark_returns,
        config,
    )
    asset_returns = _history_through_backtest_end(asset_returns, config)
    rebalance_dates = _rebalance_dates(
        windowed_asset_returns.index,
        config.rebalance.frequency,
    )
    constraints = _build_optimization_constraints(config, metadata, asset_returns.columns)
    method = _select_optimization_method(config)
    fallback_drift_threshold = (
        config.rebalance.fallback.sell_allowed_if_absolute_drift_exceeds
        if config.rebalance.fallback is not None
        else None
    )

    configured_risk_free = get_risk_free_rate(config)
    lookbacks = sorted(
        {
            *(SENSITIVITY_LOOKBACK_PERIODS if lookback_grid is None else lookback_grid),
            config.backtest.lookback_periods,
        }
    )
    risk_free_rates = sorted(
        {
            *(SENSITIVITY_RISK_FREE_RATES if risk_free_grid is None else risk_free_grid),
            configured_risk_free,
        }
    )
    rebalance_modes = list(dict.fromkeys([config.rebalance.mode, "full_rebalance"]))

    runs: list[dict[str, Any]] = []
    run_returns: dict[int, pd.Series] = {}
    for lookback in lookbacks:
        for risk_free_rate in risk_free_rates:
            for rebalance_mode in rebalance_modes:
                row: dict[str, Any] = {
                    "lookback_periods": int(lookback),
                    "risk_free_rate": float(risk_free_rate),
                    "rebalance_mode": rebalance_mode,
                    "status": "ok",
                    "reason": None,
                    "first_return_date": None,
                    "evaluation_start_date": None,
                    "cagr": None,
                    "annualized_volatility": None,
                    "sharpe_ratio": None,
                    "benchmark_sharpe_ratio": None,
                    "sharpe_minus_benchmark": None,
                }
                runs.append(row)
                if lookback > len(asset_returns.index) - 1:
                    row["status"] = "insufficient_history"
                    row["reason"] = (
                        f"lookback {lookback} exceeds the {len(asset_returns.index)} "
                        "available return observations."
                    )
                    continue
                try:
                    result = run_walk_forward_backtest(
                        asset_returns,
                        rebalance_dates=rebalance_dates,
                        lookback_periods=int(lookback),
                        optimization_method=method,
                        max_weight=config.optimization.default_max_weight_per_etf,
                        transaction_cost_rate=_transaction_cost_rate(config),
                        asset_classes=constraints["asset_classes"],
                        asset_class_bounds=constraints["asset_class_bounds"],
                        ticker_bounds=constraints["ticker_bounds"],
                        bond_assets=constraints["bond_assets"],
                        min_bond_exposure=constraints["min_bond_exposure"],
                        expense_ratios=constraints["expense_ratios"],
                        risk_free_rate=float(risk_free_rate),
                        rebalance_mode=rebalance_mode,
                        contribution_amount=config.rebalance.contribution_amount,
                        tolerance_bands=config.rebalance.tolerance_bands,
                        initial_capital=config.backtest.initial_capital,
                        fallback_sell_allowed=config.rebalance.fallback_sell_allowed,
                        fallback_drift_threshold=fallback_drift_threshold,
                        realized_constraint_policy=config.rebalance.realized_constraint_policy,
                        covariance_method=config.optimization.risk_model,
                        expected_return_method=config.optimization.expected_return_estimator,
                    )
                except ValueError as exc:
                    row["status"] = "failed"
                    row["reason"] = str(exc)
                    continue
                run_returns[len(runs) - 1] = result.portfolio_returns
                row["first_return_date"] = result.portfolio_returns.index.min().date().isoformat()

    if not run_returns:
        raise InsufficientHistoryError(
            "No sensitivity run could be completed. Every lookback in the grid is longer "
            "than the available return history, or every run failed."
        )

    common_start = max(series.index.min() for series in run_returns.values())
    for position, series in run_returns.items():
        row = runs[position]
        evaluated = series.loc[series.index >= common_start]
        benchmark_window = benchmark_returns.reindex(evaluated.index).dropna()
        evaluated = evaluated.reindex(benchmark_window.index)
        strategy_sharpe = sharpe_ratio(
            evaluated,
            risk_free_rate=row["risk_free_rate"],
            periods_per_year=252,
        )
        benchmark_sharpe = sharpe_ratio(
            benchmark_window,
            risk_free_rate=row["risk_free_rate"],
            periods_per_year=252,
        )
        row["evaluation_start_date"] = common_start.date().isoformat()
        row["cagr"] = cagr(evaluated, periods_per_year=252)
        row["annualized_volatility"] = portfolio_volatility(evaluated, periods_per_year=252)
        row["sharpe_ratio"] = strategy_sharpe
        row["benchmark_sharpe_ratio"] = benchmark_sharpe
        row["sharpe_minus_benchmark"] = strategy_sharpe - benchmark_sharpe

    ok_runs = [row for row in runs if row["status"] == "ok"]
    sharpe_values = [float(row["sharpe_ratio"]) for row in ok_runs]
    payload: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": stage_run_id,
        "optimization_method": method,
        "benchmark": config.benchmark.primary,
        "common_start_date": common_start.date().isoformat(),
        "end_date": asset_returns.index.max().date().isoformat(),
        "configured": {
            "lookback_periods": config.backtest.lookback_periods,
            "risk_free_rate": configured_risk_free,
            "rebalance_mode": config.rebalance.mode,
        },
        "summary": {
            "run_count": len(runs),
            "ok_count": len(ok_runs),
            "sharpe_min": min(sharpe_values),
            "sharpe_max": max(sharpe_values),
            "share_of_runs_beating_benchmark": sum(
                1 for row in ok_runs if float(row["sharpe_minus_benchmark"]) > 0.0
            )
            / len(ok_runs),
        },
        "runs": runs,
    }
    output_path = project_root / "reports/metrics/sensitivity.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_metrics_json(payload, output_path)
    log_event(
        LOGGER,
        logging.INFO,
        "pipeline_stage_completed",
        run_id=stage_run_id,
        stage="sensitivity",
        config_path=str(config_path),
        run_count=len(runs),
        ok_count=len(ok_runs),
        sharpe_min=min(sharpe_values),
        sharpe_max=max(sharpe_values),
        output_path=str(output_path),
    )
    return output_path


def _default_log_file(args: argparse.Namespace, project_root: Path, run_id: str) -> Path | None:
````

### Step 6 — create new file `tests/unit/test_sensitivity_command.py`

New tests.

Create the file `tests/unit/test_sensitivity_command.py` with exactly this content (the file must end with one newline):

````python
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
````

### Step 7 — edit `Makefile`

Makefile: declare the new target as phony.

Open `Makefile`. FIND this exact block (it occurs exactly once):

````text
.PHONY: help sync test test-fast lint format typecheck coverage check precommit run handoff-bundle
````

REPLACE the whole block with:

````text
.PHONY: help sync test test-fast lint format typecheck coverage check precommit run sensitivity handoff-bundle
````

### Step 8 — edit `Makefile`

Makefile: help text. The line must start with ONE TAB character, not spaces.

Open `Makefile`. FIND this exact block (it occurs exactly once):

````text
	@echo "  run            Run the full pipeline using configs/base.yaml"
````

REPLACE the whole block with:

````text
	@echo "  run            Run the full pipeline using configs/base.yaml"
	@echo "  sensitivity    Re-run the backtest over lookback/risk-free/rebalance-mode grids"
````

### Step 9 — edit `Makefile`

Makefile: the target itself. The command line must start with ONE TAB character.

Open `Makefile`. FIND this exact block (it occurs exactly once):

````text
run:
	uv run etf-portfolio run-all --config configs/base.yaml
````

REPLACE the whole block with:

````text
run:
	uv run etf-portfolio run-all --config configs/base.yaml

sensitivity:
	uv run etf-portfolio sensitivity --config configs/base.yaml
````

### Step 10 — edit `README.md`

README: list the command in the CLI reference.

Open `README.md`. FIND this exact block (it occurs exactly once):

````markdown
uv run etf-portfolio run-all --config configs/base.yaml
```

Global flags
````

REPLACE the whole block with:

````markdown
uv run etf-portfolio run-all --config configs/base.yaml
uv run etf-portfolio sensitivity --config configs/base.yaml
```

`sensitivity` is not part of `run-all`. It re-runs the backtest over a grid of lookbacks,
risk-free rates and rebalance modes and writes `reports/metrics/sensitivity.json`
(see [docs/runbook.md](docs/runbook.md)).

Global flags
````

### Step 11 — edit `docs/output_contract.md`

Document the new artifact.

Open `docs/output_contract.md`. FIND this exact block (it occurs exactly once):

````markdown
| Validation summary | `reports/metrics/validation_summary.json` |
````

REPLACE the whole block with:

````markdown
| Sensitivity grid | `reports/metrics/sensitivity.json` | Written only by the `sensitivity` command: `schema_version`, `run_id`, `optimization_method`, `benchmark`, `common_start_date`, `end_date`, `configured`, `summary` (`run_count`, `ok_count`, `sharpe_min`, `sharpe_max`, `share_of_runs_beating_benchmark`), `runs` (one row per lookback × risk-free rate × rebalance mode) |
| Validation summary | `reports/metrics/validation_summary.json` |
````

### Step 12 — append to the end of `docs/runbook.md`

Runbook: how to use the command and read its output.

Go to the very end of `docs/runbook.md`. Do not delete anything. Add this text after the last line:

````markdown

## How To Check Sensitivity To Arbitrary Choices

One backtest is one draw. Before trusting a headline number, check how much it moves when
choices that have no "right" value are changed:

```bash
uv run etf-portfolio sensitivity --config configs/base.yaml
```

This re-runs the walk-forward backtest for every combination of:

- lookback: 252, 504, 756 and 1260 trading days, plus `backtest.lookback_periods`;
- risk-free rate: 0%, plus `risk_free.value`;
- rebalance mode: `rebalance.mode`, plus `full_rebalance`.

It writes `reports/metrics/sensitivity.json` and nothing else (the main report and
`backtest_metrics.json` are not touched). With `configs/base.yaml` it takes well under a minute.

Reading the file:

- Every row in `runs` is scored on the **same** window, starting at `common_start_date`
  (the latest first-return date across all runs), so rows are comparable.
- `summary.sharpe_min` and `summary.sharpe_max` show the spread. If the spread is as large
  as the advantage you care about, that advantage depends on an arbitrary choice.
- `summary.share_of_runs_beating_benchmark` is the fraction of runs whose Sharpe ratio is
  above the primary benchmark's on the same window and at the same risk-free rate.
- Rows with `status` `insufficient_history` or `failed` have `null` metrics and a `reason`.

Record the outcome in `docs/research_log.md` whenever a headline result is published.
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
uv run pytest -q tests/unit/test_sensitivity_command.py
````

Expected: `3 passed`.

````bash
uv run etf-portfolio sensitivity --help
````

Expected: Usage text that starts with `usage: etf-portfolio sensitivity`.

````bash
make -n sensitivity
````

Expected: `uv run etf-portfolio sensitivity --config configs/base.yaml`

````bash
uv run pytest -q -x
````

Expected: Last line starts with `373 passed`. No `failed`, no `error`.

## 8. Commit

````bash
git add -A
git status --short
````

Expected: only the files listed in section 3 appear.

````bash
git commit -m "feat(cli): add sensitivity command for lookback, risk-free rate and rebalance mode"
````

## 9. Done checklist

- [ ] `run_sensitivity` exists in `src/etf_portfolio/cli.py`.
- [ ] `uv run etf-portfolio sensitivity --help` works.
- [ ] `docs/runbook.md` ends with the section `How To Check Sensitivity To Arbitrary Choices`.
- [ ] The full test suite passes.
- [ ] You did NOT run the command on `configs/base.yaml` in this plan (Plan 09 does that).
- [ ] `git status --short` prints nothing.
- [ ] You did not run `git push`.

## 10. If something goes wrong

- **A FIND block is not in the file.** Check you are on the right branch and that every earlier plan is committed (`git log --oneline | head`). Search for the first line of the FIND block only (`grep -n "<first line>" <file>`). If the surrounding code is different from the plan, STOP and report; do not improvise.
- **`ruff format --check` reports a file would be reformatted.** Run `uv run ruff format <that file>` once, then re-run the verify commands.
- **You want to start this plan over.** Run `git checkout -- . && git clean -fd -- src tests docs configs` (this discards ONLY uncommitted work of this plan), then start again from section 4.
- **`make -n sensitivity` prints `missing separator`.** A Makefile command line starts with spaces. Replace the leading spaces with one TAB character.
- **`ruff format --check` wants to reformat `cli.py`.** Run `uv run ruff format src/etf_portfolio/cli.py`, then `git diff --stat` must still list only the files of this plan.
