# Plan 04 — Give the fixed-allocation benchmarks the same schedule and costs as the strategy

- **Depends on:** Plan 03 must be committed
- **Branch:** `fix/audit-remediation`
- **Estimated size:** 2 source files, 4 doc files and 2 test files edited, 1 test file created

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

After this plan, the fixed-allocation benchmarks from `benchmark.secondary` in the config
(for example `global_60_40` = 60% VT / 40% BND) are simulated like a real portfolio:

- between two rebalance dates the holdings drift (buy-and-hold);
- on each of the strategy's rebalance dates they are reset to the fixed weights;
- each reset pays the same transaction cost rate as the strategy, on the traded weight.

The wording in the report, README and docs is updated to say this.

## 2. Background (why this change exists)

Audit finding: the strategy pays 3 basis points on every trade and only trades monthly, but
the `60/40 Portfolio` and `simple_global_baseline` benchmarks were computed by
`_composite_benchmark_returns` in `src/etf_portfolio/cli.py` as

    returns[tickers] * fixed_weights, summed per day

That is a portfolio rebalanced to its target **every day for free**. The comparison was not
like-for-like.

This plan replaces `_composite_benchmark_returns` with `_static_mix_benchmark_returns`.

Details you must not change:

- The cost of a rebalance done at the close of date T is subtracted from the **next**
  return date. The strategy engine uses the same convention.
- No initial purchase cost is charged to these benchmarks. It would be a one-off 3 bps and
  would fall outside the reported window in most runs. The docs say so explicitly.
- The single-ETF benchmark (`Selected Benchmark ETF`, VT) is not changed: one fund that is
  bought once never needs rebalancing.
- Two existing tests assert the OLD wording. This plan updates those two assertions. That
  is intended.

## 3. Files this plan touches

- `src/etf_portfolio/cli.py` (edit)
- `src/etf_portfolio/reporting/report.py` (edit)
- `tests/unit/test_report_generation.py` (edit)
- `tests/unit/test_cli_pipeline.py` (edit)
- `tests/unit/test_static_mix_benchmark.py` (create)
- `README.md` (edit)
- `docs/how_to_read_the_report.md` (edit)
- `docs/methodology.md` (edit)
- `docs/assumptions_and_limitations.md` (edit)

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

Expected: The line contains `retry SLSQP on a tolerance ladder` (Plan 03's commit).

## 5. Fast path — apply the prepared patch

The patch file contains every edit in this plan, already tested. Try it first.

````bash
git apply --check plans/patches/04-like-for-like-static-benchmarks.patch
````

- If this prints **nothing**, the patch fits. Run:

````bash
git apply plans/patches/04-like-for-like-static-benchmarks.patch
````

  Then jump to section 7 (Verify).

- If it prints an **error**, do not force it. Use section 6 (manual steps) instead.

## 6. Manual steps (only if the fast path failed)

### Step 1 — edit `src/etf_portfolio/cli.py`

`_build_benchmark_suite`: call the new simulation for each secondary benchmark.

Open `src/etf_portfolio/cli.py`. FIND this exact block (it occurs exactly once):

````python
        suite[label] = _composite_benchmark_returns(returns, allocations)
````

REPLACE the whole block with:

````python
        suite[label] = _static_mix_benchmark_returns(
            returns,
            allocations,
            rebalance_dates=rebalance_dates,
            transaction_cost_rate=transaction_cost_rate,
        )
````

### Step 2 — edit `src/etf_portfolio/cli.py`

Replace the old helper function with the new one.

Open `src/etf_portfolio/cli.py`. FIND this exact block (it occurs exactly once):

````python
def _composite_benchmark_returns(
    returns: pd.DataFrame,
    allocations: dict[str, float],
) -> pd.Series:
    available = returns.loc[:, list(allocations)]
    weights = pd.Series(allocations, dtype=float)
    return available.mul(weights, axis=1).sum(axis=1)
````

REPLACE the whole block with:

````python
def _static_mix_benchmark_returns(
    returns: pd.DataFrame,
    allocations: dict[str, float],
    *,
    rebalance_dates: pd.DatetimeIndex,
    transaction_cost_rate: float,
) -> pd.Series:
    """Simulate a fixed-allocation benchmark on the strategy's rebalance schedule.

    Holdings drift between rebalance dates. At the close of each rebalance date the
    mix is reset to `allocations`; the cost of that trade (cost rate times traded
    weight) is charged on the next return date, the same convention as the strategy
    engine. No initial purchase cost is charged.
    """

    available = returns.loc[:, list(allocations)].dropna()
    target = pd.Series(allocations, dtype=float).reindex(available.columns)
    schedule = set(pd.DatetimeIndex(rebalance_dates))
    weights = target.copy()
    pending_cost = 0.0
    daily_returns: list[float] = []
    for date, asset_returns_row in available.iterrows():
        grown = weights * (1.0 + asset_returns_row)
        growth = float(grown.sum())
        daily_returns.append(growth - 1.0 - pending_cost)
        pending_cost = 0.0
        weights = grown / growth
        if date in schedule:
            pending_cost = transaction_cost_rate * float((target - weights).abs().sum())
            weights = target.copy()
    return pd.Series(daily_returns, index=available.index, dtype=float)
````

### Step 3 — edit `src/etf_portfolio/cli.py`

`_report_assumptions`: describe the benchmarks truthfully.

Open `src/etf_portfolio/cli.py`. FIND this exact block (it occurs exactly once):

````python
            "The selected benchmark ETF and configured secondary allocation benchmarks are "
            "external/theoretical return-series baselines; they are not simulated with "
            "contribution-only drift or strategy transaction costs."
````

REPLACE the whole block with:

````python
            "The selected benchmark ETF is a single-fund buy-and-hold return series. "
            "Configured secondary allocation benchmarks are reset to their fixed weights on "
            "the same rebalance dates as the strategy and pay the same transaction "
            "cost/slippage rate on the traded weight; they are not simulated with "
            "contribution-only drift, and no initial purchase cost is charged to them."
````

### Step 4 — edit `src/etf_portfolio/reporting/report.py`

Same wording fix in the report's built-in assumptions table.

Open `src/etf_portfolio/reporting/report.py`. FIND this exact block (it occurs exactly once):

````python
        "single-ETF and fixed-allocation benchmarks are theoretical return-series baselines.",
````

REPLACE the whole block with:

````python
        "fixed-allocation benchmarks are reset on the same rebalance dates and pay the same "
        "cost rate; the single-ETF benchmark is a buy-and-hold return series.",
````

### Step 5 — edit `tests/unit/test_report_generation.py`

This existing assertion checks the OLD sentence. Update it to the new sentence.

Open `tests/unit/test_report_generation.py`. FIND this exact block (it occurs exactly once):

````python
    assert (
        "single-ETF and fixed-allocation benchmarks are theoretical return-series baselines"
        in report_html
    )
````

REPLACE the whole block with:

````python
    assert (
        "fixed-allocation benchmarks are reset on the same rebalance dates and pay the same "
        "cost rate" in report_html
    )
````

### Step 6 — edit `tests/unit/test_cli_pipeline.py`

This existing assertion checks the OLD sentence. Update it to the new sentence.

Open `tests/unit/test_cli_pipeline.py`. FIND this exact block (it occurs exactly once):

````python
    assert "external/theoretical return-series baselines" in latest_report_html
````

REPLACE the whole block with:

````python
    assert "reset to their fixed weights on the same rebalance dates" in latest_report_html
````

### Step 7 — create new file `tests/unit/test_static_mix_benchmark.py`

New tests for the new helper.

Create the file `tests/unit/test_static_mix_benchmark.py` with exactly this content (the file must end with one newline):

````python
"""Tests for fixed-allocation benchmarks simulated on the strategy's rebalance schedule."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf_portfolio import cli


def _random_returns() -> pd.DataFrame:
    rng = np.random.default_rng(21)
    index = pd.bdate_range("2021-01-04", periods=30)
    return pd.DataFrame(
        rng.normal(loc=0.0005, scale=0.02, size=(30, 2)),
        index=index,
        columns=["AAA", "BBB"],
    )


ALLOCATIONS = {"AAA": 0.6, "BBB": 0.4}


def test_without_rebalance_dates_the_mix_is_buy_and_hold() -> None:
    returns = _random_returns()

    series = cli._static_mix_benchmark_returns(
        returns,
        ALLOCATIONS,
        rebalance_dates=pd.DatetimeIndex([]),
        transaction_cost_rate=0.0,
    )

    weights = pd.Series(ALLOCATIONS)
    expected_growth = float(((1.0 + returns).prod() * weights).sum())
    assert float((1.0 + series).prod()) == pytest.approx(expected_growth, rel=1e-12)


def test_rebalancing_every_day_for_free_equals_constant_weights() -> None:
    returns = _random_returns()

    series = cli._static_mix_benchmark_returns(
        returns,
        ALLOCATIONS,
        rebalance_dates=returns.index,
        transaction_cost_rate=0.0,
    )

    expected = returns.mul(pd.Series(ALLOCATIONS), axis=1).sum(axis=1)
    pd.testing.assert_series_equal(series, expected, check_names=False)


def test_cost_of_a_rebalance_is_charged_on_the_next_return_date() -> None:
    index = pd.bdate_range("2021-01-04", periods=3)
    returns = pd.DataFrame(
        {"AAA": [0.10, 0.0, 0.0], "BBB": [0.0, 0.0, 0.0]},
        index=index,
    )

    series = cli._static_mix_benchmark_returns(
        returns,
        {"AAA": 0.5, "BBB": 0.5},
        rebalance_dates=pd.DatetimeIndex([index[0]]),
        transaction_cost_rate=0.01,
    )

    # Day 1: 0.5 * 10% = 5%. Drifted weights: AAA 0.55/1.05, BBB 0.50/1.05.
    # Rebalancing back to 50/50 trades |0.5 - 0.55/1.05| * 2 of the portfolio.
    traded_weight = 2.0 * (0.55 / 1.05 - 0.5)
    assert series.iloc[0] == pytest.approx(0.05)
    assert series.iloc[1] == pytest.approx(-0.01 * traded_weight)
    assert series.iloc[2] == pytest.approx(0.0)


def test_rebalance_dates_outside_the_data_are_ignored() -> None:
    returns = _random_returns()

    series = cli._static_mix_benchmark_returns(
        returns,
        ALLOCATIONS,
        rebalance_dates=pd.DatetimeIndex(["1999-01-01"]),
        transaction_cost_rate=0.01,
    )

    assert len(series) == len(returns)
    assert series.index.equals(returns.index)


def test_old_costless_daily_rebalanced_helper_is_gone() -> None:
    assert not hasattr(cli, "_composite_benchmark_returns")
````

### Step 8 — edit `README.md`

README wording.

Open `README.md`. FIND this exact block (it occurs exactly once):

````markdown
- The selected benchmark ETF and configured secondary allocation benchmarks are external/theoretical return-series baselines and do not model contribution-only drift or strategy transaction costs.
````

REPLACE the whole block with:

````markdown
- The selected benchmark ETF is a single-fund buy-and-hold return series. Configured secondary allocation benchmarks (for example 60/40) drift between rebalance dates, are reset to their fixed weights on the same rebalance dates as the strategy, and pay the same transaction cost/slippage rate on the traded weight. They do not model contributions, and no initial purchase cost is charged to them.
````

### Step 9 — edit `docs/how_to_read_the_report.md`

Docs wording.

Open `docs/how_to_read_the_report.md`. FIND this exact block (it occurs exactly once):

````markdown
The selected benchmark ETF and configured secondary allocation benchmarks are different: they are external/theoretical return-series baselines and do not model contribution-only drift or strategy transaction costs.
````

REPLACE the whole block with:

````markdown
The selected benchmark ETF and configured secondary allocation benchmarks are different. The selected benchmark ETF is a single-fund buy-and-hold return series. Secondary allocation benchmarks (for example 60/40) drift between rebalance dates, are reset to their fixed weights on the same rebalance dates as the strategy, and pay the same transaction cost/slippage rate on the traded weight. Neither kind models contribution-only drift, and no initial purchase cost is charged to them.
````

### Step 10 — edit `docs/methodology.md`

Docs wording.

Open `docs/methodology.md`. FIND this exact block (it occurs exactly once):

````markdown
Constant-weight benchmarks (like 60/40, configured secondary allocation benchmarks, or single ETFs) remain theoretical return-series baselines. They do not model contribution-only drift, strategy transaction costs, or rebalance-date implementation mechanics unless a future config explicitly adds those assumptions.
````

REPLACE the whole block with:

````markdown
Fixed-allocation benchmarks (like 60/40 and the other configured secondary allocation benchmarks) are simulated on the strategy's rebalance schedule: holdings drift between rebalance dates, are reset to the fixed weights on each rebalance date, and pay the strategy's transaction cost/slippage rate on the traded weight, charged on the next return date. Single-ETF benchmarks are buy-and-hold return series. Neither kind models contribution-only drift, and no initial purchase cost is charged to them.
````

### Step 11 — edit `docs/assumptions_and_limitations.md`

Docs wording.

Open `docs/assumptions_and_limitations.md`. FIND this exact block (it occurs exactly once):

````markdown
The selected benchmark ETF and configured secondary allocation benchmarks are external/theoretical return-series baselines. They do not model contribution-only drift or strategy transaction costs.
````

REPLACE the whole block with:

````markdown
The selected benchmark ETF is a single-fund buy-and-hold return series. Configured secondary allocation benchmarks drift between rebalance dates, are reset to their fixed weights on the same rebalance dates as the strategy, and pay the same transaction cost/slippage rate on the traded weight. They do not model contribution-only drift, and no initial purchase cost is charged to them.
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
uv run pytest -q tests/unit/test_static_mix_benchmark.py
````

Expected: `5 passed`.

````bash
grep -rn "theoretical return-series" src tests README.md docs
````

Expected: Prints nothing (the old wording is gone everywhere).

````bash
uv run pytest -q -x
````

Expected: Last line starts with `353 passed`. No `failed`, no `error`.

## 8. Commit

````bash
git add -A
git status --short
````

Expected: only the files listed in section 3 appear.

````bash
git commit -m "fix(benchmarks): simulate fixed-allocation benchmarks with the strategy schedule and costs"
````

## 9. Done checklist

- [ ] `_composite_benchmark_returns` no longer exists in `src/etf_portfolio/cli.py`.
- [ ] `_static_mix_benchmark_returns` exists and is called from `_build_benchmark_suite`.
- [ ] The phrase `theoretical return-series` appears nowhere in `src`, `tests`, `README.md`, `docs`.
- [ ] The full test suite passes.
- [ ] `git status --short` prints nothing.
- [ ] You did not run `git push`.

## 10. If something goes wrong

- **A FIND block is not in the file.** Check you are on the right branch and that every earlier plan is committed (`git log --oneline | head`). Search for the first line of the FIND block only (`grep -n "<first line>" <file>`). If the surrounding code is different from the plan, STOP and report; do not improvise.
- **`ruff format --check` reports a file would be reformatted.** Run `uv run ruff format <that file>` once, then re-run the verify commands.
- **You want to start this plan over.** Run `git checkout -- . && git clean -fd -- src tests docs configs` (this discards ONLY uncommitted work of this plan), then start again from section 4.
- **`test_cli_pipeline.py` fails on the assertion you edited.** The sentence in `_report_assumptions` (cli.py) and the sentence in the test must contain the same words `reset to their fixed weights on the same rebalance dates`. Compare them character by character.
