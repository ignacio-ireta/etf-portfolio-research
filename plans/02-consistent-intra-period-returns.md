# Plan 02 — Make daily returns follow the same buy-and-hold path as the weights

- **Depends on:** Plan 01 must be committed
- **Branch:** `fix/audit-remediation`
- **Estimated size:** 1 file edited, 1 test file created, about 30 changed lines of source

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

After this plan, the walk-forward engine computes the daily portfolio returns between two
rebalance dates as the returns of holdings that are **bought once and then left alone**
(buy-and-hold). Before this plan the returns were computed as if the portfolio were reset
to its starting weights every single day, for free.

## 2. Background (why this change exists)

Audit finding: `run_walk_forward_backtest` in `src/etf_portfolio/backtesting/engine.py` was
internally inconsistent.

- Line `segment_returns = realized_window.mul(applied_weights, axis=1).sum(axis=1)` multiplies
  every day's asset returns by the **same** weights. That is the return of a portfolio that is
  rebalanced back to those weights every day, with no trading cost.
- But `_drift_weights(...)`, a few lines later, computes the end-of-period weights assuming the
  holdings were **never touched** (buy-and-hold).

A real contribution-only or monthly-rebalanced portfolio is buy-and-hold between rebalance
dates. The mismatch flattered the committed strategy by about 2 basis points per month
(about 2.6% cumulative over the backtest).

The fix replaces the one line with a call to a new helper `_buy_and_hold_returns`. The helper:

1. Builds the value of each holding day by day: `weight * cumulative growth of that asset`.
2. Sums them to a portfolio value path that starts at 1.0.
3. Returns the day-over-day change of that path.

If the weights sum to less than 1 (only possible for universes with an explicit `CASH`
column), the missing weight is treated as cash earning 0%, which is exactly what the old
line did on day one.

Note: on the **first day** of every period the old and new formulas give the same number.
They differ only from day two onward, and only slightly. Tests with one-day periods are
unaffected.

## 3. Files this plan touches

- `src/etf_portfolio/backtesting/engine.py` (edit)
- `tests/unit/test_buy_and_hold_accounting.py` (create)
- `tests/golden/backtest_metrics_optimized_strategy.json` (rewritten by the command step; do not edit it by hand)

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

Expected: The line contains `version the lookback in config` (Plan 01's commit).

## 5. Fast path — apply the prepared patch

The patch file contains every edit in this plan, already tested. Try it first.

````bash
git apply --check plans/patches/02-consistent-intra-period-returns.patch
````

- If this prints **nothing**, the patch fits. Run:

````bash
git apply plans/patches/02-consistent-intra-period-returns.patch
````

  The patch does **not** run commands. After applying it you must still run the command steps listed in section 6 (they are marked **COMMAND STEP**).

  Then jump to section 7 (Verify).

- If it prints an **error**, do not force it. Use section 6 (manual steps) instead.

## 6. Manual steps (only if the fast path failed)

### Step 1 — edit `src/etf_portfolio/backtesting/engine.py`

Use the new helper for the period's daily returns.

Open `src/etf_portfolio/backtesting/engine.py`. FIND this exact block (it occurs exactly once):

````python
        segment_returns = realized_window.mul(applied_weights, axis=1).sum(axis=1)
        transaction_cost_dollars = transaction_costs(
````

REPLACE the whole block with:

````python
        segment_returns = _buy_and_hold_returns(realized_window, applied_weights)
        transaction_cost_dollars = transaction_costs(
````

### Step 2 — edit `src/etf_portfolio/backtesting/engine.py`

Add the helper function directly above `_drift_weights`.

Open `src/etf_portfolio/backtesting/engine.py`. FIND this exact block (it occurs exactly once):

````python
def _drift_weights(
    applied_weights: pd.Series,
    segment_returns: pd.Series,
    realized_window: pd.DataFrame,
) -> pd.Series:
````

REPLACE the whole block with:

````python
def _buy_and_hold_returns(
    realized_window: pd.DataFrame,
    applied_weights: pd.Series,
) -> pd.Series:
    """Daily portfolio returns when holdings are left untouched inside the window.

    Weights are applied once, at the start of the window, and then drift with each
    asset's own return. This matches `_drift_weights`, which projects the same
    untouched holdings to the next rebalance date. Weight that is not allocated to
    any asset is treated as cash earning zero.
    """

    holding_values = (1.0 + realized_window).cumprod().mul(applied_weights, axis=1)
    cash_weight = 1.0 - float(applied_weights.sum())
    portfolio_values = holding_values.sum(axis=1) + cash_weight
    previous_values = portfolio_values.shift(1)
    previous_values.iloc[0] = 1.0
    return portfolio_values / previous_values - 1.0


def _drift_weights(
    applied_weights: pd.Series,
    segment_returns: pd.Series,
    realized_window: pd.DataFrame,
) -> pd.Series:
````

### Step 3 — create new file `tests/unit/test_buy_and_hold_accounting.py`

New tests that fail before this plan and pass after it.

Create the file `tests/unit/test_buy_and_hold_accounting.py` with exactly this content (the file must end with one newline):

````python
"""Tests that intra-period returns follow the same buy-and-hold path as the weights."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf_portfolio.backtesting.engine import _buy_and_hold_returns, run_walk_forward_backtest


def _two_asset_window() -> pd.DataFrame:
    index = pd.bdate_range("2021-01-04", periods=3)
    return pd.DataFrame(
        {
            "AAA": [0.10, 0.10, 0.10],
            "BBB": [0.00, 0.00, 0.00],
        },
        index=index,
    )


def test_buy_and_hold_returns_let_weights_drift_inside_the_window() -> None:
    window = _two_asset_window()
    weights = pd.Series({"AAA": 0.5, "BBB": 0.5})

    returns = _buy_and_hold_returns(window, weights)

    # Day 1: 0.5 * 10% = 5%. Portfolio value 1.05, of which AAA is 0.55.
    # Day 2: AAA earns 10% on 0.55 -> +0.055 on 1.05 = 5.238...%, NOT 5%.
    assert returns.iloc[0] == pytest.approx(0.05)
    assert returns.iloc[1] == pytest.approx(0.055 / 1.05)
    assert returns.iloc[2] == pytest.approx(0.0605 / 1.105)


def test_buy_and_hold_returns_compound_to_the_value_of_untouched_holdings() -> None:
    rng = np.random.default_rng(11)
    index = pd.bdate_range("2021-01-04", periods=21)
    window = pd.DataFrame(
        rng.normal(loc=0.0005, scale=0.02, size=(21, 3)),
        index=index,
        columns=["AAA", "BBB", "CCC"],
    )
    weights = pd.Series({"AAA": 0.2, "BBB": 0.3, "CCC": 0.5})

    returns = _buy_and_hold_returns(window, weights)

    expected_growth = float(((1.0 + window).prod() * weights).sum())
    assert float((1.0 + returns).prod()) == pytest.approx(expected_growth, rel=1e-12)


def test_buy_and_hold_returns_treat_unallocated_weight_as_zero_return_cash() -> None:
    window = _two_asset_window()
    weights = pd.Series({"AAA": 0.5, "BBB": 0.0})

    returns = _buy_and_hold_returns(window, weights)

    assert returns.iloc[0] == pytest.approx(0.05)
    assert returns.iloc[1] == pytest.approx(0.055 / 1.05)


def test_engine_portfolio_value_matches_untouched_holdings_between_rebalances() -> None:
    """Engine-level check: one period's compounded return equals the holdings' growth."""

    rng = np.random.default_rng(5)
    index = pd.bdate_range("2020-01-01", periods=120)
    returns = pd.DataFrame(
        rng.normal(loc=0.0004, scale=0.015, size=(120, 3)),
        index=index,
        columns=["AAA", "BBB", "CCC"],
    )
    rebalance_dates = [index[40], index[80]]

    result = run_walk_forward_backtest(
        returns,
        rebalance_dates=rebalance_dates,
        lookback_periods=30,
        optimization_method="equal_weight",
    )

    first_weights = result.applied_weights.iloc[0]
    first_window = returns.loc[(returns.index > index[40]) & (returns.index <= index[80])]
    period_returns = result.portfolio_returns.loc[first_window.index]

    expected_growth = float(((1.0 + first_window).prod() * first_weights).sum())
    assert float((1.0 + period_returns).prod()) == pytest.approx(expected_growth, rel=1e-12)
````

### Step 4 — COMMAND STEP (run this even if you used the fast path)

The golden snapshot `tests/golden/backtest_metrics_optimized_strategy.json` stores metric values produced by the OLD formula. Regenerate it. This is an intended change: the numbers move slightly (for example `Alpha` changes from about `-0.0646` to about `-0.0653`).

````bash
UPDATE_GOLDEN=1 uv run pytest -q tests/golden
````

Expected: Last line contains `1 skipped`. (The test skips itself after rewriting the snapshot file.)

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
uv run pytest -q tests/golden
````

Expected: `1 passed`.

````bash
uv run pytest -q tests/unit/test_buy_and_hold_accounting.py
````

Expected: `4 passed`.

````bash
git diff --stat -- tests/golden
````

Expected: Exactly one file listed: `tests/golden/backtest_metrics_optimized_strategy.json`.

````bash
uv run pytest -q -x
````

Expected: Last line starts with `344 passed`. No `failed`, no `error`.

## 8. Commit

````bash
git add -A
git status --short
````

Expected: only the files listed in section 3 appear, plus `tests/golden/backtest_metrics_optimized_strategy.json` (regenerated snapshot).

````bash
git commit -m "fix(backtest): compute intra-period returns as buy-and-hold, consistent with drift"
````

## 9. Done checklist

- [ ] `_buy_and_hold_returns` exists in `src/etf_portfolio/backtesting/engine.py` and is called once inside `run_walk_forward_backtest`.
- [ ] The text `realized_window.mul(applied_weights, axis=1).sum(axis=1)` no longer appears in `engine.py`.
- [ ] The golden snapshot was regenerated and `tests/golden` passes.
- [ ] The full test suite passes.
- [ ] `git status --short` prints nothing.
- [ ] You did not run `git push`.

## 10. If something goes wrong

- **A FIND block is not in the file.** Check you are on the right branch and that every earlier plan is committed (`git log --oneline | head`). Search for the first line of the FIND block only (`grep -n "<first line>" <file>`). If the surrounding code is different from the plan, STOP and report; do not improvise.
- **`ruff format --check` reports a file would be reformatted.** Run `uv run ruff format <that file>` once, then re-run the verify commands.
- **You want to start this plan over.** Run `git checkout -- . && git clean -fd -- src tests docs configs` (this discards ONLY uncommitted work of this plan), then start again from section 4.
- **A test outside `tests/golden` fails with a small numeric difference.** STOP and report the test name and the two numbers. Do not edit expected values in other tests on your own.
