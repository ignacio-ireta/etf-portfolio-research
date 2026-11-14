# Plan 06 — Report a confidence interval for every Sharpe-ratio comparison

- **Depends on:** Plan 05 must be committed
- **Branch:** `fix/audit-remediation`
- **Estimated size:** 1 source file created, 1 source file edited, 3 doc files edited, 1 test file created

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

1. Computes, for each benchmark, a 95% bootstrap confidence interval for
   `Sharpe(Optimized Strategy) - Sharpe(benchmark)`.
2. Writes the results to `reports/metrics/backtest_metrics.json` under the key
   `sharpe_difference_vs_benchmarks` (one entry per benchmark name).
3. Shows the interval versus the primary benchmark in the report's assumptions list (line
   starting with `Statistical uncertainty:`), with a plain statement of whether the interval
   contains zero.

## 2. Background (why this change exists)

Audit finding: the report presents `Sharpe 0.626` for the strategy next to `0.524` for VT
with no indication of how uncertain those numbers are. On about 11 years of data the 95%
interval for that difference is roughly `[-0.10, +0.30]`: it contains zero, so the data
cannot distinguish the two. The same is true versus equal-weight and risk parity.

A stationary block bootstrap is used because daily returns are not independent
(volatility clusters). It resamples blocks of consecutive days with random lengths
(21 trading days on average), using the **same** blocks for both series so their
correlation is preserved.

Design decisions (do not change them):

- The new code lives in a new module, `src/etf_portfolio/backtesting/uncertainty.py`.
- The random seed is fixed (`0`), so two runs on the same data give identical numbers.
  This keeps the pipeline deterministic, which the output contract requires.
- The bootstrap parameters are module constants, not config fields.
- If a benchmark has fewer than 3 common observations with the strategy, its numeric
  results are `null` in the JSON and no assumption line is written.

## 3. Files this plan touches

- `src/etf_portfolio/backtesting/uncertainty.py` (create)
- `src/etf_portfolio/cli.py` (edit)
- `tests/unit/test_sharpe_uncertainty.py` (create)
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

Expected: The line contains `report the realized-vs-target implementation gap` (Plan 05's commit).

## 5. Fast path — apply the prepared patch

The patch file contains every edit in this plan, already tested. Try it first.

````bash
git apply --check plans/patches/06-sharpe-difference-confidence-intervals.patch
````

- If this prints **nothing**, the patch fits. Run:

````bash
git apply plans/patches/06-sharpe-difference-confidence-intervals.patch
````

  Then jump to section 7 (Verify).

- If it prints an **error**, do not force it. Use section 6 (manual steps) instead.

## 6. Manual steps (only if the fast path failed)

### Step 1 — create new file `src/etf_portfolio/backtesting/uncertainty.py`

The new module with the bootstrap function.

Create the file `src/etf_portfolio/backtesting/uncertainty.py` with exactly this content (the file must end with one newline):

````python
"""Sampling uncertainty for backtest comparisons.

A Sharpe ratio measured on roughly a decade of daily returns is a noisy estimate.
This module puts a confidence interval around the *difference* between two Sharpe
ratios with a stationary block bootstrap (Politis and Romano, 1994): both return
series are resampled with the same randomly chosen blocks of consecutive days, so
day-to-day autocorrelation and the correlation between the two series are kept.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DEFAULT_BOOTSTRAP_RESAMPLES = 1000
DEFAULT_MEAN_BLOCK_LENGTH = 21
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_BOOTSTRAP_SEED = 0
MINIMUM_BOOTSTRAP_OBSERVATIONS = 3


def bootstrap_sharpe_difference(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    mean_block_length: int = DEFAULT_MEAN_BLOCK_LENGTH,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap the annualized Sharpe ratio of `returns` minus that of `benchmark_returns`.

    Returns a JSON-friendly dictionary. When fewer than three common observations
    exist, or either series has zero volatility, the numeric results are ``None``.
    """

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive.")
    if mean_block_length <= 0:
        raise ValueError("mean_block_length must be positive.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1.")

    paired = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    observation_count = len(paired)
    summary: dict[str, Any] = {
        "sharpe_difference": None,
        "ci_lower": None,
        "ci_upper": None,
        "probability_not_positive": None,
        "confidence_level": float(confidence_level),
        "n_observations": observation_count,
        "n_resamples": int(n_resamples),
        "mean_block_length": int(mean_block_length),
        "seed": int(seed),
    }
    if observation_count < MINIMUM_BOOTSTRAP_OBSERVATIONS:
        return summary

    strategy = paired.iloc[:, 0].to_numpy(dtype=float)
    benchmark = paired.iloc[:, 1].to_numpy(dtype=float)
    daily_risk_free = risk_free_rate / periods_per_year
    annualizer = float(np.sqrt(periods_per_year))

    strategy_volatility = float(strategy.std(ddof=1))
    benchmark_volatility = float(benchmark.std(ddof=1))
    if np.isclose(strategy_volatility, 0.0) or np.isclose(benchmark_volatility, 0.0):
        return summary

    point_estimate = (
        (strategy.mean() - daily_risk_free) / strategy_volatility
        - (benchmark.mean() - daily_risk_free) / benchmark_volatility
    ) * annualizer

    # Stationary bootstrap indices: each position either continues the current block
    # (previous index + 1, wrapping around) or jumps to a fresh random start.
    rng = np.random.default_rng(seed)
    positions = np.arange(observation_count)
    jumps = rng.random((n_resamples, observation_count)) < 1.0 / mean_block_length
    jumps[:, 0] = True
    fresh_starts = rng.integers(0, observation_count, size=(n_resamples, observation_count))
    last_jump = np.maximum.accumulate(np.where(jumps, positions, 0), axis=1)
    block_starts = np.take_along_axis(fresh_starts, last_jump, axis=1)
    indices = (block_starts + positions - last_jump) % observation_count

    strategy_samples = strategy[indices]
    benchmark_samples = benchmark[indices]
    with np.errstate(divide="ignore", invalid="ignore"):
        differences = (
            (strategy_samples.mean(axis=1) - daily_risk_free) / strategy_samples.std(axis=1, ddof=1)
            - (benchmark_samples.mean(axis=1) - daily_risk_free)
            / benchmark_samples.std(axis=1, ddof=1)
        ) * annualizer
    differences = differences[np.isfinite(differences)]
    if differences.size == 0:
        return summary

    tail = (1.0 - confidence_level) / 2.0
    summary.update(
        {
            "sharpe_difference": float(point_estimate),
            "ci_lower": float(np.quantile(differences, tail)),
            "ci_upper": float(np.quantile(differences, 1.0 - tail)),
            "probability_not_positive": float(np.mean(differences <= 0.0)),
        }
    )
    return summary


__all__ = [
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_CONFIDENCE_LEVEL",
    "DEFAULT_MEAN_BLOCK_LENGTH",
    "bootstrap_sharpe_difference",
]
````

### Step 2 — edit `src/etf_portfolio/cli.py`

Import the new function.

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
    calculate_sharpe_ratio,
    summarize_implementation_gap,
)
from etf_portfolio.backtesting.uncertainty import bootstrap_sharpe_difference
````

### Step 3 — edit `src/etf_portfolio/cli.py`

`run_backtest`: compute the interval versus every benchmark.

Open `src/etf_portfolio/cli.py`. FIND this exact block (it occurs exactly once):

````python
    primary_benchmark = aligned_benchmark_suite.get("Selected Benchmark ETF", aligned_benchmark)
````

REPLACE the whole block with:

````python
    primary_benchmark = aligned_benchmark_suite.get("Selected Benchmark ETF", aligned_benchmark)
    sharpe_uncertainty = {
        name: bootstrap_sharpe_difference(
            aligned_portfolio,
            series,
            risk_free_rate=risk_free_rate,
            periods_per_year=252,
        )
        for name, series in aligned_benchmark_suite.items()
    }
````

### Step 4 — edit `src/etf_portfolio/cli.py`

`run_backtest`: pass the primary-benchmark interval into the report assumptions.

Open `src/etf_portfolio/cli.py`. FIND this exact block (it occurs exactly once):

````python
            effective_start_date=effective_start_date,
            implementation_gap=gap_summary,
        ),
````

REPLACE the whole block with:

````python
            effective_start_date=effective_start_date,
            implementation_gap=gap_summary,
            sharpe_uncertainty=sharpe_uncertainty.get("Selected Benchmark ETF"),
        ),
````

### Step 5 — edit `src/etf_portfolio/cli.py`

`run_backtest`: write all intervals into the metrics JSON.

Open `src/etf_portfolio/cli.py`. FIND this exact block (it occurs exactly once):

````python
        "implementation_gap": gap_summary,
    }
````

REPLACE the whole block with:

````python
        "implementation_gap": gap_summary,
        "sharpe_difference_vs_benchmarks": sharpe_uncertainty,
    }
````

### Step 6 — edit `src/etf_portfolio/cli.py`

`_report_assumptions`: accept the interval.

Open `src/etf_portfolio/cli.py`. FIND this exact block (it occurs exactly once):

````python
    implementation_gap: dict[str, float | int] | None = None,
) -> dict[str, str]:
````

REPLACE the whole block with:

````python
    implementation_gap: dict[str, float | int] | None = None,
    sharpe_uncertainty: dict[str, Any] | None = None,
) -> dict[str, str]:
````

### Step 7 — edit `src/etf_portfolio/cli.py`

`_report_assumptions`: add the `statistical_uncertainty` line.

Open `src/etf_portfolio/cli.py`. FIND this exact block (it occurs exactly once):

````python
        assumptions["implementation_gap"] = gap_text
    if run_id is not None:
````

REPLACE the whole block with:

````python
        assumptions["implementation_gap"] = gap_text
    if (
        sharpe_uncertainty is not None
        and sharpe_uncertainty.get("sharpe_difference") is not None
        and sharpe_uncertainty.get("ci_lower") is not None
        and sharpe_uncertainty.get("ci_upper") is not None
    ):
        ci_lower = float(sharpe_uncertainty["ci_lower"])
        ci_upper = float(sharpe_uncertainty["ci_upper"])
        verdict = (
            "The interval contains zero: this sample cannot distinguish the two Sharpe ratios."
            if ci_lower <= 0.0 <= ci_upper
            else "The interval excludes zero: the difference is unlikely to be sampling "
            "noise alone, which is still not evidence about the future."
        )
        assumptions["statistical_uncertainty"] = (
            "Statistical uncertainty: Sharpe ratio of the strategy minus Sharpe ratio of "
            f"the primary benchmark ({config.benchmark.primary}) is "
            f"{float(sharpe_uncertainty['sharpe_difference']):+.2f}, with a "
            f"{float(sharpe_uncertainty['confidence_level']):.0%} bootstrap interval of "
            f"[{ci_lower:+.2f}, {ci_upper:+.2f}] "
            f"({int(sharpe_uncertainty['n_resamples'])} stationary-bootstrap resamples, "
            f"mean block {int(sharpe_uncertainty['mean_block_length'])} trading days, "
            f"{int(sharpe_uncertainty['n_observations'])} daily observations). {verdict}"
        )
    if run_id is not None:
````

### Step 8 — create new file `tests/unit/test_sharpe_uncertainty.py`

New tests.

Create the file `tests/unit/test_sharpe_uncertainty.py` with exactly this content (the file must end with one newline):

````python
"""Tests for the bootstrap confidence interval on Sharpe-ratio differences."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from etf_portfolio import cli
from etf_portfolio.backtesting.uncertainty import bootstrap_sharpe_difference
from etf_portfolio.config import load_config


def _noise(seed: int, periods: int = 500) -> pd.Series:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2018-01-01", periods=periods)
    return pd.Series(rng.normal(loc=0.0004, scale=0.01, size=periods), index=index)


def test_result_is_deterministic_for_a_fixed_seed() -> None:
    first = bootstrap_sharpe_difference(_noise(1), _noise(2), seed=7)
    second = bootstrap_sharpe_difference(_noise(1), _noise(2), seed=7)
    other_seed = bootstrap_sharpe_difference(_noise(1), _noise(2), seed=8)

    assert first == second
    assert first["ci_lower"] != other_seed["ci_lower"]


def test_identical_series_have_zero_difference() -> None:
    series = _noise(3)

    result = bootstrap_sharpe_difference(series, series)

    assert result["sharpe_difference"] == pytest.approx(0.0)
    assert result["ci_lower"] == pytest.approx(0.0)
    assert result["ci_upper"] == pytest.approx(0.0)
    assert result["probability_not_positive"] == pytest.approx(1.0)


def test_a_constant_daily_edge_gives_an_interval_above_zero() -> None:
    benchmark = _noise(4)
    strategy = benchmark + 0.001

    result = bootstrap_sharpe_difference(strategy, benchmark)

    assert result["sharpe_difference"] > 0.0
    assert result["ci_lower"] > 0.0
    assert result["probability_not_positive"] == pytest.approx(0.0)


def test_two_unrelated_noise_series_give_an_interval_that_contains_zero() -> None:
    result = bootstrap_sharpe_difference(_noise(5), _noise(6))

    assert result["ci_lower"] < 0.0 < result["ci_upper"]
    assert result["n_observations"] == 500
    assert result["n_resamples"] == 1000
    assert result["mean_block_length"] == 21
    assert result["confidence_level"] == pytest.approx(0.95)


def test_point_estimate_matches_the_metrics_module() -> None:
    from etf_portfolio.backtesting.metrics import sharpe_ratio

    strategy, benchmark = _noise(9), _noise(10)

    result = bootstrap_sharpe_difference(strategy, benchmark, risk_free_rate=0.03)

    expected = sharpe_ratio(strategy, risk_free_rate=0.03) - sharpe_ratio(
        benchmark, risk_free_rate=0.03
    )
    assert result["sharpe_difference"] == pytest.approx(expected)


def test_too_few_observations_return_none_results() -> None:
    index = pd.bdate_range("2021-01-04", periods=2)
    short = pd.Series([0.01, -0.01], index=index)

    result = bootstrap_sharpe_difference(short, short)

    assert result["sharpe_difference"] is None
    assert result["ci_lower"] is None
    assert result["ci_upper"] is None
    assert result["n_observations"] == 2


def test_missing_values_are_dropped_pairwise() -> None:
    strategy = _noise(11)
    benchmark = _noise(12)
    benchmark.iloc[:100] = np.nan

    result = bootstrap_sharpe_difference(strategy, benchmark)

    assert result["n_observations"] == 400


def test_invalid_arguments_are_rejected() -> None:
    series = _noise(13)
    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap_sharpe_difference(series, series, n_resamples=0)
    with pytest.raises(ValueError, match="confidence_level"):
        bootstrap_sharpe_difference(series, series, confidence_level=1.0)


def test_backtest_writes_sharpe_uncertainty_for_every_benchmark(
    tiny_project: Path, synthetic_prices: pd.DataFrame
) -> None:
    processed = tiny_project / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    synthetic_prices.to_parquet(processed / "prices_validated.parquet")
    synthetic_prices.pct_change(fill_method=None).dropna().to_parquet(processed / "returns.parquet")

    cli.run_backtest("configs/base.yaml", project_root=tiny_project, lookback_periods=5)

    metrics = json.loads(
        (tiny_project / "reports/metrics/backtest_metrics.json").read_text(encoding="utf-8")
    )
    uncertainty = metrics["sharpe_difference_vs_benchmarks"]
    assert set(uncertainty) == set(metrics["benchmarks"])
    primary = uncertainty["Selected Benchmark ETF"]
    assert primary["ci_lower"] <= primary["sharpe_difference"] <= primary["ci_upper"]

    report_html = (tiny_project / "reports/html/latest_report.html").read_text(encoding="utf-8")
    assert "Statistical uncertainty:" in report_html


def test_assumption_line_says_when_the_interval_contains_zero(tiny_project: Path) -> None:
    config = load_config(tiny_project / "configs/base.yaml")
    base = {
        "confidence_level": 0.95,
        "n_observations": 2800,
        "n_resamples": 1000,
        "mean_block_length": 21,
        "seed": 0,
        "probability_not_positive": 0.17,
    }

    inconclusive = cli._report_assumptions(
        config,
        "max_sharpe",
        sharpe_uncertainty={**base, "sharpe_difference": 0.10, "ci_lower": -0.10, "ci_upper": 0.30},
    )["statistical_uncertainty"]
    conclusive = cli._report_assumptions(
        config,
        "max_sharpe",
        sharpe_uncertainty={**base, "sharpe_difference": 0.40, "ci_lower": 0.10, "ci_upper": 0.70},
    )["statistical_uncertainty"]

    assert inconclusive.startswith("Statistical uncertainty:")
    assert "[-0.10, +0.30]" in inconclusive
    assert "contains zero" in inconclusive
    assert "excludes zero" in conclusive


def test_assumption_line_is_omitted_without_a_usable_interval(tiny_project: Path) -> None:
    config = load_config(tiny_project / "configs/base.yaml")

    assumptions = cli._report_assumptions(
        config,
        "max_sharpe",
        sharpe_uncertainty={"sharpe_difference": None, "ci_lower": None, "ci_upper": None},
    )

    assert "statistical_uncertainty" not in assumptions
````

### Step 9 — edit `docs/output_contract.md`

Document the new metrics JSON key.

Open `docs/output_contract.md`. FIND this exact block (it occurs exactly once):

````markdown
`rebalances_with_constraint_violation`), `run_record` (relative path) |
````

REPLACE the whole block with:

````markdown
`rebalances_with_constraint_violation`), `sharpe_difference_vs_benchmarks` (benchmark name → `sharpe_difference`, `ci_lower`, `ci_upper`, `probability_not_positive`, `confidence_level`, `n_observations`, `n_resamples`, `mean_block_length`, `seed`), `run_record` (relative path) |
````

### Step 10 — edit `README.md`

README: mention the interval.

Open `README.md`. FIND this exact block (it occurs exactly once):

````markdown
- Backtest reporting separates optimizer targets from realized holdings.
````

REPLACE the whole block with:

````markdown
- Every Sharpe comparison comes with a 95% stationary-bootstrap confidence interval for the Sharpe difference (report assumption line `Statistical uncertainty:`; all benchmarks in `reports/metrics/backtest_metrics.json` under `sharpe_difference_vs_benchmarks`). If the interval contains zero, the sample cannot distinguish the two portfolios.
- Backtest reporting separates optimizer targets from realized holdings.
````

### Step 11 — append to the end of `docs/how_to_read_the_report.md`

Add a reader-facing section.

Go to the very end of `docs/how_to_read_the_report.md`. Do not delete anything. Add this text after the last line:

````markdown

## Statistical uncertainty: is the Sharpe difference real?

A Sharpe ratio measured on about ten years of daily returns is a noisy number. Two
portfolios whose true Sharpe ratios are equal will routinely show a difference of 0.1 or
0.2 by luck alone.

Every backtest therefore reports, for each benchmark, a 95% confidence interval for
`Sharpe(Optimized Strategy) - Sharpe(benchmark)`:

- The assumption line that starts with `Statistical uncertainty:` gives the interval versus
  the primary benchmark.
- `reports/metrics/backtest_metrics.json` gives it for every benchmark under
  `sharpe_difference_vs_benchmarks`.

How to read it:

- **The interval contains zero** (for example `[-0.10, +0.30]`): the data cannot tell the two
  Sharpe ratios apart. Do not say the strategy "beat" that benchmark on a risk-adjusted basis.
- **The interval excludes zero** (for example `[+0.05, +0.31]`): the difference is unlikely to
  be sampling noise alone. This still says nothing about the future, about costs that are
  not modeled, or about how many configurations were tried before this one.

Method: stationary block bootstrap, 1000 resamples, mean block length 21 trading days,
fixed seed 0 (results are reproducible). Both series are resampled with the same blocks.
`probability_not_positive` is the share of resamples in which the difference was zero or
negative.
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
uv run pytest -q tests/unit/test_sharpe_uncertainty.py
````

Expected: `11 passed`.

````bash
uv run pytest -q -x
````

Expected: Last line starts with `370 passed`. No `failed`, no `error`.

## 8. Commit

````bash
git add -A
git status --short
````

Expected: only the files listed in section 3 appear.

````bash
git commit -m "feat(report): add bootstrap confidence intervals for Sharpe differences"
````

## 9. Done checklist

- [ ] `src/etf_portfolio/backtesting/uncertainty.py` exists.
- [ ] `reports/metrics/backtest_metrics.json` written by a backtest contains `sharpe_difference_vs_benchmarks` (checked by the new end-to-end test).
- [ ] `docs/how_to_read_the_report.md` ends with the section `Statistical uncertainty: is the Sharpe difference real?`.
- [ ] The full test suite passes.
- [ ] `git status --short` prints nothing.
- [ ] You did not run `git push`.

## 10. If something goes wrong

- **A FIND block is not in the file.** Check you are on the right branch and that every earlier plan is committed (`git log --oneline | head`). Search for the first line of the FIND block only (`grep -n "<first line>" <file>`). If the surrounding code is different from the plan, STOP and report; do not improvise.
- **`ruff format --check` reports a file would be reformatted.** Run `uv run ruff format <that file>` once, then re-run the verify commands.
- **You want to start this plan over.** Run `git checkout -- . && git clean -fd -- src tests docs configs` (this discards ONLY uncommitted work of this plan), then start again from section 4.
- **`ruff check` reports `I001` (import block is un-sorted) in `cli.py`.** The new import line must be directly below the `from etf_portfolio.backtesting.metrics import (...)` block and above `from etf_portfolio.config import ...`. Or run `uv run ruff check --fix src/etf_portfolio/cli.py`.
