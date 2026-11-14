# Plan 03 — Stop one solver hiccup from aborting a whole backtest

- **Depends on:** Plan 02 must be committed
- **Branch:** `fix/audit-remediation`
- **Estimated size:** 1 file edited, 1 test file created, about 50 changed lines of source

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

After this plan, when the SLSQP solver cannot converge at the strict tolerance
(`ftol=1e-12`), the optimizer automatically retries with slightly looser tolerances
(`1e-10`, then `1e-8`, then `1e-6`) before giving up. When a looser tolerance was needed, a
WARNING log event `optimizer_tolerance_relaxed` records it. Only if **every** tolerance fails
does the optimizer raise `ValueError("Optimization failed: ...")`, exactly as before.

## 2. Background (why this change exists)

Audit finding: with the committed data, a walk-forward backtest with a lookback of 252 or
1008 days, or with a 4.5% risk-free rate, crashes with

    ValueError: Optimization failed: Positive directional derivative for linesearch

That message means SLSQP is already sitting at the optimum but cannot prove it at
`ftol=1e-12`. In all 17 failing optimizations found during the audit, retrying the same
start point with `ftol=1e-10` succeeded and returned the same objective value to 9+
decimal places. One failed month out of ~170 used to kill the entire run.

Design decisions (do not change them):

- The strict tolerance stays first. Any optimization that succeeded before this plan
  returns **exactly the same weights** after it. Existing results do not move.
- A looser tolerance is tried only after **all** start points failed at the tighter one.
- The helper `_solve_with_retries` now returns a pair `(result, ftol_used)` instead of
  just `result`.
- Genuinely infeasible problems still raise `ValueError("Optimization failed: ...")`.
  An existing test (`test_optimizer_logs_failed_optimization`) checks this and must keep
  passing.

## 3. Files this plan touches

- `src/etf_portfolio/optimization/optimizer.py` (edit)
- `tests/unit/test_optimizer_solver_fallback.py` (create)

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

Expected: The line contains `compute intra-period returns as buy-and-hold` (Plan 02's commit).

## 5. Fast path — apply the prepared patch

The patch file contains every edit in this plan, already tested. Try it first.

````bash
git apply --check plans/patches/03-solver-tolerance-ladder.patch
````

- If this prints **nothing**, the patch fits. Run:

````bash
git apply plans/patches/03-solver-tolerance-ladder.patch
````

  Then jump to section 7 (Verify).

- If it prints an **error**, do not force it. Use section 6 (manual steps) instead.

## 6. Manual steps (only if the fast path failed)

### Step 1 — edit `src/etf_portfolio/optimization/optimizer.py`

Add the tolerance ladder constant next to the other solver constants.

Open `src/etf_portfolio/optimization/optimizer.py`. FIND this exact block (it occurs exactly once):

````python
COVARIANCE_PSD_TOLERANCE = 1e-10
````

REPLACE the whole block with:

````python
COVARIANCE_PSD_TOLERANCE = 1e-10
# SLSQP convergence tolerances, tried in order. The first entry is the strict default.
# A later entry is used only when every start point failed at the previous one
# (typically "Positive directional derivative for linesearch": the solver is at the
# optimum but cannot certify it at the stricter tolerance).
SOLVER_FTOL_LADDER = (1e-12, 1e-10, 1e-8, 1e-6)
````

### Step 2 — edit `src/etf_portfolio/optimization/optimizer.py`

`optimize_portfolio`: the helper now returns two values.

Open `src/etf_portfolio/optimization/optimizer.py`. FIND this exact block (it occurs exactly once):

````python
    result = _solve_with_retries(
        objective=objective,
````

REPLACE the whole block with:

````python
    result, ftol_used = _solve_with_retries(
        objective=objective,
````

### Step 3 — edit `src/etf_portfolio/optimization/optimizer.py`

`optimize_portfolio`: log a warning when a looser tolerance was needed.

Open `src/etf_portfolio/optimization/optimizer.py`. FIND this exact block (it occurs exactly once):

````python
        raise ValueError(f"Optimization failed: {result.message}")

    weights = pd.Series(result.x, index=assets, dtype=float, name="weight")
````

REPLACE the whole block with:

````python
        raise ValueError(f"Optimization failed: {result.message}")

    if ftol_used != SOLVER_FTOL_LADDER[0]:
        log_event(
            LOGGER,
            logging.WARNING,
            "optimizer_tolerance_relaxed",
            method=validated_method,
            solver_used=DEFAULT_SOLVER,
            optimizer_status="success",
            strict_ftol=SOLVER_FTOL_LADDER[0],
            ftol_used=ftol_used,
            reason="SLSQP did not converge at the strict tolerance from any start point.",
        )

    weights = pd.Series(result.x, index=assets, dtype=float, name="weight")
````

### Step 4 — edit `src/etf_portfolio/optimization/optimizer.py`

`_solve_with_retries`: loop over the tolerance ladder.

Open `src/etf_portfolio/optimization/optimizer.py`. FIND this exact block (it occurs exactly once):

````python
) -> Any:
    last_result: Any = None
    for candidate in candidate_initial_weights:
        result = minimize(
            objective,
            x0=candidate,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        last_result = result
        if result.success:
            return result
    return last_result
````

REPLACE the whole block with:

````python
) -> tuple[Any, float]:
    """Solve from each start point, relaxing ftol only when every start point fails.

    Returns `(result, ftol_used)`. A solve that succeeds at the strict tolerance is
    returned unchanged, so relaxing never alters a result that already converged.
    """

    last_result: Any = None
    last_ftol = SOLVER_FTOL_LADDER[0]
    for ftol in SOLVER_FTOL_LADDER:
        for candidate in candidate_initial_weights:
            result = minimize(
                objective,
                x0=candidate,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500, "ftol": ftol},
            )
            last_result = result
            last_ftol = ftol
            if result.success:
                return result, ftol
    return last_result, last_ftol
````

### Step 5 — create new file `tests/unit/test_optimizer_solver_fallback.py`

New tests for the ladder. They replace the real solver with a fake that fails on purpose, so they do not depend on market data.

Create the file `tests/unit/test_optimizer_solver_fallback.py` with exactly this content (the file must end with one newline):

````python
"""Tests for the SLSQP tolerance ladder used when the strict solve cannot converge."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import OptimizeResult

from etf_portfolio.optimization import optimizer as optimizer_module
from etf_portfolio.optimization.optimizer import SOLVER_FTOL_LADDER, optimize_portfolio

ASSETS = ["VTI", "BND", "IAU", "REMX"]


def _expected_returns() -> pd.Series:
    return pd.Series({"VTI": 0.10, "BND": 0.05, "IAU": 0.07, "REMX": 0.09})


def _covariance_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [0.040, 0.004, 0.006, 0.012],
            [0.004, 0.010, 0.002, 0.003],
            [0.006, 0.002, 0.025, 0.005],
            [0.012, 0.003, 0.005, 0.060],
        ],
        index=ASSETS,
        columns=ASSETS,
    )


def _install_fake_minimize(
    monkeypatch: pytest.MonkeyPatch,
    *,
    loosest_failing_ftol: float,
) -> list[float]:
    """Make every solve fail when ftol <= loosest_failing_ftol; record each ftol tried."""

    real_minimize = optimizer_module.minimize
    seen_ftols: list[float] = []

    def fake_minimize(objective: Any, **kwargs: Any) -> Any:
        ftol = float(kwargs["options"]["ftol"])
        seen_ftols.append(ftol)
        if ftol <= loosest_failing_ftol:
            start = np.asarray(kwargs["x0"], dtype=float)
            return OptimizeResult(
                x=start,
                success=False,
                status=8,
                message="Positive directional derivative for linesearch",
                fun=float(objective(start)),
            )
        return real_minimize(objective, **kwargs)

    monkeypatch.setattr(optimizer_module, "minimize", fake_minimize)
    return seen_ftols


def _events(caplog: pytest.LogCaptureFixture, name: str) -> list[logging.LogRecord]:
    return [record for record in caplog.records if getattr(record, "event", "") == name]


def test_ladder_starts_strict_and_is_sorted_from_strict_to_loose() -> None:
    assert SOLVER_FTOL_LADDER[0] == 1e-12
    assert list(SOLVER_FTOL_LADDER) == sorted(SOLVER_FTOL_LADDER)


def test_strict_solve_success_does_not_relax_or_warn(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    seen_ftols = _install_fake_minimize(monkeypatch, loosest_failing_ftol=0.0)

    with caplog.at_level(logging.WARNING):
        weights = optimize_portfolio(
            _expected_returns(),
            _covariance_matrix(),
            method="max_sharpe",
            max_weight=0.6,
        )

    assert weights.sum() == pytest.approx(1.0)
    assert set(seen_ftols) == {SOLVER_FTOL_LADDER[0]}
    assert _events(caplog, "optimizer_tolerance_relaxed") == []


def test_relaxed_tolerance_is_used_when_strict_solve_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    seen_ftols = _install_fake_minimize(monkeypatch, loosest_failing_ftol=1e-12)

    with caplog.at_level(logging.WARNING):
        weights = optimize_portfolio(
            _expected_returns(),
            _covariance_matrix(),
            method="max_sharpe",
            max_weight=0.6,
        )

    assert weights.sum() == pytest.approx(1.0)
    assert weights.max() <= 0.6 + 1e-8
    assert seen_ftols[0] == SOLVER_FTOL_LADDER[0]
    assert seen_ftols[-1] == SOLVER_FTOL_LADDER[1]
    relaxed_events = _events(caplog, "optimizer_tolerance_relaxed")
    assert len(relaxed_events) == 1
    assert relaxed_events[0].ftol_used == SOLVER_FTOL_LADDER[1]


def test_failure_is_raised_only_after_every_tolerance_was_tried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_ftols = _install_fake_minimize(monkeypatch, loosest_failing_ftol=1.0)

    with pytest.raises(ValueError, match="Optimization failed"):
        optimize_portfolio(
            _expected_returns(),
            _covariance_matrix(),
            method="max_sharpe",
            max_weight=0.6,
        )

    assert set(seen_ftols) == set(SOLVER_FTOL_LADDER)
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
uv run pytest -q tests/unit/test_optimizer_solver_fallback.py
````

Expected: `4 passed`.

````bash
uv run pytest -q tests/unit/test_optimizer.py
````

Expected: All pass, 0 failed.

````bash
uv run python - <<'EOF'
import logging
import warnings
from pathlib import Path

import pandas as pd

from etf_portfolio import cli
from etf_portfolio.backtesting.engine import run_walk_forward_backtest
from etf_portfolio.data.ingest import load_etf_universe_metadata

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
root = Path(".")
config = cli._load_project_config("configs/base.yaml", root)
returns = pd.read_parquet(root / "data/processed/returns.parquet")
metadata = load_etf_universe_metadata(cli._metadata_path(root)).set_index("ticker")
assets = cli._asset_returns(returns, config)
constraints = cli._build_optimization_constraints(config, metadata, assets.columns)
for lookback in (252, 1008):
    result = run_walk_forward_backtest(
        assets,
        rebalance_dates=cli._rebalance_dates(assets.index, "monthly"),
        lookback_periods=lookback,
        optimization_method="max_sharpe",
        max_weight=config.optimization.default_max_weight_per_etf,
        asset_classes=constraints["asset_classes"],
        asset_class_bounds=constraints["asset_class_bounds"],
        ticker_bounds=constraints["ticker_bounds"],
        bond_assets=constraints["bond_assets"],
        min_bond_exposure=constraints["min_bond_exposure"],
        expense_ratios=constraints["expense_ratios"],
        risk_free_rate=0.03,
        covariance_method="ledoit_wolf",
    )
    print("SOLVER OK lookback", lookback, "rebalances", len(result.target_weights))
EOF
````

Expected: Exactly two lines, both starting with `SOLVER OK lookback`. No traceback. (Before this plan the same script crashed with `Optimization failed: Positive directional derivative for linesearch`.) Takes about 1 minute.

````bash
git status --short
````

Expected: Only the two files from section 3. The check script above must not have changed any file.

````bash
uv run pytest -q -x
````

Expected: Last line starts with `348 passed`. No `failed`, no `error`.

## 8. Commit

````bash
git add -A
git status --short
````

Expected: only the files listed in section 3 appear.

````bash
git commit -m "fix(optimizer): retry SLSQP on a tolerance ladder instead of aborting the run"
````

## 9. Done checklist

- [ ] `SOLVER_FTOL_LADDER` exists in `src/etf_portfolio/optimization/optimizer.py`.
- [ ] `_solve_with_retries` returns `(result, ftol_used)`.
- [ ] The real-data check printed two `SOLVER OK` lines.
- [ ] The full test suite passes.
- [ ] `git status --short` prints nothing.
- [ ] You did not run `git push`.

## 10. If something goes wrong

- **A FIND block is not in the file.** Check you are on the right branch and that every earlier plan is committed (`git log --oneline | head`). Search for the first line of the FIND block only (`grep -n "<first line>" <file>`). If the surrounding code is different from the plan, STOP and report; do not improvise.
- **`ruff format --check` reports a file would be reformatted.** Run `uv run ruff format <that file>` once, then re-run the verify commands.
- **You want to start this plan over.** Run `git checkout -- . && git clean -fd -- src tests docs configs` (this discards ONLY uncommitted work of this plan), then start again from section 4.
- **`TypeError: cannot unpack non-iterable OptimizeResult object`.** You changed the call site but not `_solve_with_retries` (or the other way round). Both edits are required.
- **The real-data check still crashes with `Optimization failed`.** Open `optimizer.py` and confirm the `minimize(` call inside `_solve_with_retries` uses `"ftol": ftol` (the loop variable), not the literal `1e-12`.
