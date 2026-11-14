# Plan 01 — Put the lookback in the config and let it warm up before the start date

- **Depends on:** nothing (this is the first plan)
- **Branch:** `fix/audit-remediation`
- **Estimated size:** 3 files edited, 1 test file created, about 150 changed lines

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

1. The estimation lookback (how many trading days of history the optimizer sees) is a
   config field, `backtest.lookback_periods`, with default `756`. It is therefore part of
   the config hash.
2. The `--lookback-periods` CLI flag still exists, but only as an optional override. When it
   is not given, the config value is used.
3. Returns dated **before** `backtest.start_date` are used to estimate the first portfolio.
   Before this plan they were thrown away, which silently delayed the first trade by three
   years.
4. Every backtest run record stores `lookback_periods`, `configured_lookback_periods` and
   `configured_backtest_start_date`.
5. The report's assumptions list states the configured start date, the real first return
   date, and the lookback.
6. A WARNING log event `backtest_start_delayed` is emitted when the real first return date
   is more than 45 days after `backtest.start_date`.

## 2. Background (why this change exists)

Audit finding: `configs/base.yaml` says `backtest.start_date: "2012-01-01"`, but the
committed run starts on `2015-02-02`. The cause is in `run_backtest` in
`src/etf_portfolio/cli.py`: the return history is cut at `backtest.start_date` **first**, and
only afterwards the 756-day lookback is taken from what is left. So the first 756 trading
days of the window can never trade.

Second finding: `756` is the constant `DEFAULT_LOOKBACK_PERIODS` in `cli.py`. It is not in the
config and not in the run record, so two runs with different lookbacks have the same
`config_hash`.

Important fact for later plans: the committed data starts on 2011-01-04, so even after this
fix the first simulated return of `configs/base.yaml` is in **February 2014** (2011 + 3 years
of lookback), not 2012. That is correct and expected. The new assumption line and warning
make this visible instead of hiding it.

How the fix works: `run_backtest` now keeps two frames.
`windowed_asset_returns` (cut to the configured window) is used ONLY to choose the rebalance
dates. `asset_returns` (everything up to `backtest.end_date`) is what the engine receives, so
the engine's existing rule "train on returns strictly before the rebalance date" can reach
back before the window. The engine itself is not changed in this plan.

## 3. Files this plan touches

- `src/etf_portfolio/config.py` (edit)
- `configs/base.yaml` (edit)
- `src/etf_portfolio/cli.py` (edit)
- `tests/unit/test_backtest_window_warmup.py` (create)
- `docs/output_contract.md` (edit)
- `README.md` (edit)

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

Expected: `fix/audit-remediation`. If it prints something else, run `git checkout -b fix/audit-remediation` (first time) or `git checkout fix/audit-remediation` (branch already exists) and run the command again.

## 5. Fast path — apply the prepared patch

The patch file contains every edit in this plan, already tested. Try it first.

````bash
git apply --check plans/patches/01-lookback-in-config-and-warmup.patch
````

- If this prints **nothing**, the patch fits. Run:

````bash
git apply plans/patches/01-lookback-in-config-and-warmup.patch
````

  Then jump to section 7 (Verify).

- If it prints an **error**, do not force it. Use section 6 (manual steps) instead.

## 6. Manual steps (only if the fast path failed)

### Step 1 — edit `src/etf_portfolio/config.py`

Add the new config field to `BacktestConfig`.

Open `src/etf_portfolio/config.py`. FIND this exact block (it occurs exactly once):

````python
    start_date: date | None = None
    end_date: date | None = None
    initial_capital: float = Field(default=100_000.0, gt=0.0)
````

REPLACE the whole block with:

````python
    start_date: date | None = None
    end_date: date | None = None
    initial_capital: float = Field(default=100_000.0, gt=0.0)
    # Number of trailing return observations used to estimate expected returns and
    # covariance at each rebalance. Returns dated before `start_date` are still used
    # for estimation, so the lookback does not delay the first rebalance when
    # enough earlier history exists.
    lookback_periods: int = Field(default=756, gt=0)
````

### Step 2 — edit `configs/base.yaml`

Make the value explicit in the base config so it is visible and versioned.

Open `configs/base.yaml`. FIND this exact block (it occurs exactly once):

````yaml
backtest:
  start_date: "2012-01-01"
  end_date: null
  initial_capital: 100000.0
````

REPLACE the whole block with:

````yaml
backtest:
  # First date on which the strategy may rebalance. Returns before this date are
  # used only to estimate the first portfolio (lookback warm-up).
  start_date: "2012-01-01"
  end_date: null
  initial_capital: 100000.0
  # Trailing trading days used for estimation at each rebalance (756 = about 3 years).
  # If the data does not reach back this far before start_date, the first trade is
  # delayed and the run logs a `backtest_start_delayed` warning.
  lookback_periods: 756
````

### Step 3 — edit `src/etf_portfolio/cli.py`

Remove the hidden constant; add the threshold for the new warning.

Open `src/etf_portfolio/cli.py`. FIND this exact block (it occurs exactly once):

````python
DEFAULT_LOOKBACK_PERIODS = 756
LOGGER = get_logger(__name__)
````

REPLACE the whole block with:

````python
BACKTEST_START_DELAY_WARNING_DAYS = 45
LOGGER = get_logger(__name__)
````

### Step 4 — edit `src/etf_portfolio/cli.py`

`run_backtest`: the lookback argument becomes an optional override.

Open `src/etf_portfolio/cli.py`. FIND this exact block (it occurs exactly once):

````python
    project_root: Path = Path("."),
    lookback_periods: int = DEFAULT_LOOKBACK_PERIODS,
    run_id: str | None = None,
    persist_metrics: bool = True,
) -> tuple[Path, Path]:
    """Run walk-forward backtesting and persist report and metrics artifacts."""

    config = _load_project_config(config_path, project_root)
````

REPLACE the whole block with:

````python
    project_root: Path = Path("."),
    lookback_periods: int | None = None,
    run_id: str | None = None,
    persist_metrics: bool = True,
) -> tuple[Path, Path]:
    """Run walk-forward backtesting and persist report and metrics artifacts."""

    config = _load_project_config(config_path, project_root)
    configured_lookback = _resolve_lookback_periods(config, lookback_periods)
````

### Step 5 — edit `src/etf_portfolio/cli.py`

`run_backtest`: choose rebalance dates from the configured window, but give the engine the full history so the lookback can use earlier returns.

Open `src/etf_portfolio/cli.py`. FIND this exact block (it occurs exactly once):

````python
    asset_returns, benchmark_returns = _apply_backtest_window(
        asset_returns,
        benchmark_returns,
        config,
    )

    rebalance_dates = _rebalance_dates(asset_returns.index, config.rebalance.frequency)
    effective_lookback = min(lookback_periods, len(asset_returns.index) - 1)
````

REPLACE the whole block with:

````python
    # Rebalance dates come from the configured backtest window. The estimation history
    # keeps every return before backtest.start_date so the trailing lookback can warm
    # up on pre-window data instead of delaying the first rebalance.
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
    effective_lookback = min(configured_lookback, len(asset_returns.index) - 1)
````

### Step 6 — edit `src/etf_portfolio/cli.py`

`run_backtest`: compute the real first return date and warn when it is late.

Open `src/etf_portfolio/cli.py`. FIND this exact block (it occurs exactly once):

````python
    aligned_portfolio = backtest_result.portfolio_returns.reindex(aligned_benchmark.index)
    in_sample_returns = asset_returns.loc[asset_returns.index < backtest_result.weights.index[-1]]
````

REPLACE the whole block with:

````python
    aligned_portfolio = backtest_result.portfolio_returns.reindex(aligned_benchmark.index)
    effective_start_date = pd.Timestamp(aligned_portfolio.index.min())
    configured_start_date = config.backtest.start_date
    if configured_start_date is not None and effective_start_date > pd.Timestamp(
        configured_start_date
    ) + pd.Timedelta(days=BACKTEST_START_DELAY_WARNING_DAYS):
        log_event(
            LOGGER,
            logging.WARNING,
            "backtest_start_delayed",
            run_id=stage_run_id,
            stage="backtest",
            config_path=str(config_path),
            configured_start_date=configured_start_date.isoformat(),
            effective_start_date=effective_start_date.date().isoformat(),
            lookback_periods=effective_lookback,
            reason=(
                "Not enough return history before backtest.start_date to fill the "
                "lookback; the first rebalance was delayed."
            ),
        )
    in_sample_returns = asset_returns.loc[asset_returns.index < backtest_result.weights.index[-1]]
````

### Step 7 — edit `src/etf_portfolio/cli.py`

`run_backtest`: pass the window facts into the report assumptions.

Open `src/etf_portfolio/cli.py`. FIND this exact block (it occurs exactly once):

````python
        assumptions=_report_assumptions(config, method, run_id=stage_run_id),
````

REPLACE the whole block with:

````python
        assumptions=_report_assumptions(
            config,
            method,
            run_id=stage_run_id,
            lookback_periods=effective_lookback,
            effective_start_date=effective_start_date,
        ),
````

### Step 8 — edit `src/etf_portfolio/cli.py`

`run_backtest`: store the lookback and configured start in the run record.

Open `src/etf_portfolio/cli.py`. FIND this exact block (it occurs exactly once):

````python
            backtest_metrics=metrics_payload["optimized_strategy"],
            pipeline_steps=["ingest", "validate", "features", "optimize", "backtest"],
        )
````

REPLACE the whole block with:

````python
            backtest_metrics=metrics_payload["optimized_strategy"],
            pipeline_steps=["ingest", "validate", "features", "optimize", "backtest"],
            extra={
                "lookback_periods": effective_lookback,
                "configured_lookback_periods": configured_lookback,
                "configured_backtest_start_date": (
                    configured_start_date.isoformat() if configured_start_date is not None else None
                ),
            },
        )
````

### Step 9 — edit `src/etf_portfolio/cli.py`

`run_report`: same optional override.

Open `src/etf_portfolio/cli.py`. FIND this exact block (it occurs exactly once):

````python
    project_root: Path = Path("."),
    lookback_periods: int = DEFAULT_LOOKBACK_PERIODS,
    run_id: str | None = None,
) -> Path:
````

REPLACE the whole block with:

````python
    project_root: Path = Path("."),
    lookback_periods: int | None = None,
    run_id: str | None = None,
) -> Path:
````

### Step 10 — edit `src/etf_portfolio/cli.py`

`run_all`: same optional override. The value is resolved here so the resume logic (`params=`) hashes the real number, not `None`.

Open `src/etf_portfolio/cli.py`. FIND this exact block (it occurs exactly once):

````python
    project_root: Path = Path("."),
    lookback_periods: int = DEFAULT_LOOKBACK_PERIODS,
    run_id: str | None = None,
    resume: bool = False,
````

REPLACE the whole block with:

````python
    project_root: Path = Path("."),
    lookback_periods: int | None = None,
    run_id: str | None = None,
    resume: bool = False,
````

### Step 11 — edit `src/etf_portfolio/cli.py`

`run_all`: resolve the lookback right after the config is loaded.

Open `src/etf_portfolio/cli.py`. FIND this exact block (it occurs exactly once):

````python
    config = _load_project_config(config_path, project_root)
    pipeline_run_id = run_id or generate_run_id("run-all")
````

REPLACE the whole block with:

````python
    config = _load_project_config(config_path, project_root)
    lookback_periods = _resolve_lookback_periods(config, lookback_periods)
    pipeline_run_id = run_id or generate_run_id("run-all")
````

### Step 12 — edit `src/etf_portfolio/cli.py`

CLI parser: the three `--lookback-periods` flags default to `None` now.

Open `src/etf_portfolio/cli.py`. FIND this exact block (it occurs exactly once):

````python
    backtest_parser = subparsers.add_parser("backtest", parents=[common])
    backtest_parser.add_argument("--lookback-periods", type=int, default=DEFAULT_LOOKBACK_PERIODS)

    report_parser = subparsers.add_parser("report", parents=[common])
    report_parser.add_argument("--lookback-periods", type=int, default=DEFAULT_LOOKBACK_PERIODS)

    run_all_parser = subparsers.add_parser("run-all", parents=[common])
    run_all_parser.add_argument("--lookback-periods", type=int, default=DEFAULT_LOOKBACK_PERIODS)
````

REPLACE the whole block with:

````python
    lookback_help = (
        "Override backtest.lookback_periods from the config for this run only "
        "(default: use the config value)."
    )
    backtest_parser = subparsers.add_parser("backtest", parents=[common])
    backtest_parser.add_argument("--lookback-periods", type=int, default=None, help=lookback_help)

    report_parser = subparsers.add_parser("report", parents=[common])
    report_parser.add_argument("--lookback-periods", type=int, default=None, help=lookback_help)

    run_all_parser = subparsers.add_parser("run-all", parents=[common])
    run_all_parser.add_argument("--lookback-periods", type=int, default=None, help=lookback_help)
````

### Step 13 — edit `src/etf_portfolio/cli.py`

`_report_assumptions`: accept the two new optional facts.

Open `src/etf_portfolio/cli.py`. FIND this exact block (it occurs exactly once):

````python
    optimization_method: str,
    *,
    run_id: str | None = None,
) -> dict[str, str]:
    assumptions = {
````

REPLACE the whole block with:

````python
    optimization_method: str,
    *,
    run_id: str | None = None,
    lookback_periods: int | None = None,
    effective_start_date: pd.Timestamp | None = None,
) -> dict[str, str]:
    assumptions = {
````

### Step 14 — edit `src/etf_portfolio/cli.py`

`_report_assumptions`: add the `backtest_window` line.

Open `src/etf_portfolio/cli.py`. FIND this exact block (it occurs exactly once):

````python
    if run_id is not None:
        assumptions["run_id"] = f"Run ID: {run_id}"
    return assumptions
````

REPLACE the whole block with:

````python
    if lookback_periods is not None and effective_start_date is not None:
        configured_start = (
            config.backtest.start_date.isoformat()
            if config.backtest.start_date is not None
            else "not set"
        )
        assumptions["backtest_window"] = (
            f"Backtest window: configured start {configured_start}; first simulated return "
            f"{effective_start_date.date().isoformat()}; trailing lookback "
            f"{lookback_periods} observations. Returns before the configured start are used "
            "for estimation only. If the first simulated return is much later than the "
            "configured start, the data history was too short to fill the lookback."
        )
    if run_id is not None:
        assumptions["run_id"] = f"Run ID: {run_id}"
    return assumptions
````

### Step 15 — edit `src/etf_portfolio/cli.py`

Add the two small helper functions, directly above `_apply_backtest_window`.

Open `src/etf_portfolio/cli.py`. FIND this exact block (it occurs exactly once):

````python
def _apply_backtest_window(
    asset_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    config: AppConfig,
) -> tuple[pd.DataFrame, pd.Series]:
````

REPLACE the whole block with:

````python
def _resolve_lookback_periods(config: AppConfig, override: int | None) -> int:
    """Return the CLI override when given, otherwise backtest.lookback_periods."""

    lookback_periods = config.backtest.lookback_periods if override is None else int(override)
    if lookback_periods <= 0:
        raise ConfigError("lookback_periods must be a positive integer.")
    return lookback_periods


def _history_through_backtest_end(asset_returns: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    """Keep estimation history before backtest.start_date; cut only at backtest.end_date."""

    if config.backtest.end_date is None:
        return asset_returns
    end = pd.Timestamp(config.backtest.end_date)
    return asset_returns.loc[asset_returns.index <= end]


def _apply_backtest_window(
    asset_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    config: AppConfig,
) -> tuple[pd.DataFrame, pd.Series]:
````

### Step 16 — create new file `tests/unit/test_backtest_window_warmup.py`

New tests that fail before this plan and pass after it.

Create the file `tests/unit/test_backtest_window_warmup.py` with exactly this content (the file must end with one newline):

````python
"""Tests for the versioned lookback and the pre-window estimation warm-up."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from etf_portfolio import cli
from etf_portfolio.config import BacktestConfig, load_config


def _write_processed_data(project_root: Path, prices: pd.DataFrame) -> None:
    processed = project_root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(processed / "prices_validated.parquet")
    prices.pct_change(fill_method=None).dropna().to_parquet(processed / "returns.parquet")


def _set_backtest_section(project_root: Path, *, start_date: str, lookback_periods: int) -> None:
    config_path = project_root / "configs/base.yaml"
    original = config_path.read_text(encoding="utf-8")
    old_block = '  start_date: "2020-01-02"\n  end_date: null\n'
    assert old_block in original
    new_block = (
        f'  start_date: "{start_date}"\n  end_date: null\n  lookback_periods: {lookback_periods}\n'
    )
    config_path.write_text(original.replace(old_block, new_block), encoding="utf-8")


def test_backtest_config_lookback_defaults_to_756() -> None:
    assert BacktestConfig().lookback_periods == 756


def test_backtest_config_rejects_non_positive_lookback() -> None:
    with pytest.raises(ValidationError):
        BacktestConfig(lookback_periods=0)


def test_resolve_lookback_prefers_cli_override(tiny_project: Path) -> None:
    _set_backtest_section(tiny_project, start_date="2020-01-02", lookback_periods=30)
    config = load_config(tiny_project / "configs/base.yaml")

    assert cli._resolve_lookback_periods(config, None) == 30
    assert cli._resolve_lookback_periods(config, 7) == 7


def test_history_before_start_date_warms_up_the_lookback(
    tiny_project: Path, synthetic_prices: pd.DataFrame
) -> None:
    """The first month-end inside the window must trade, using pre-window history."""

    _write_processed_data(tiny_project, synthetic_prices)
    _set_backtest_section(tiny_project, start_date="2020-03-02", lookback_periods=30)

    cli.run_backtest("configs/base.yaml", project_root=tiny_project)

    metrics = json.loads(
        (tiny_project / "reports/metrics/backtest_metrics.json").read_text(encoding="utf-8")
    )
    record = json.loads((tiny_project / metrics["run_record"]).read_text(encoding="utf-8"))

    # Rebalance on 2020-03-31 (first month-end on/after the configured start);
    # the first simulated return is the next trading day.
    assert record["start_date"] == "2020-04-01"
    assert record["lookback_periods"] == 30
    assert record["configured_lookback_periods"] == 30
    assert record["configured_backtest_start_date"] == "2020-03-02"


def test_report_assumptions_state_the_effective_window(tiny_project: Path) -> None:
    config = load_config(tiny_project / "configs/base.yaml")

    assumptions = cli._report_assumptions(
        config,
        "equal_weight",
        lookback_periods=30,
        effective_start_date=pd.Timestamp("2020-04-01"),
    )

    assert "2020-01-02" in assumptions["backtest_window"]
    assert "2020-04-01" in assumptions["backtest_window"]
    assert "30" in assumptions["backtest_window"]
````

### Step 17 — edit `docs/output_contract.md`

Document the new run-record fields.

Open `docs/output_contract.md`. FIND this exact block (it occurs exactly once):

````markdown
`output_artifacts` (name → {path, exists, sha256}), method metadata |
````

REPLACE the whole block with:

````markdown
`output_artifacts` (name → {path, exists, sha256}), method metadata; backtest records also carry `lookback_periods` (observations actually used), `configured_lookback_periods`, and `configured_backtest_start_date` |
````

### Step 18 — edit `README.md`

Correct the README sentence that described the old (wrong) window behavior.

Open `README.md`. FIND this exact block (it occurs exactly once):

````markdown
- `backtest.start_date` and `backtest.end_date` are applied to aligned asset and benchmark return series before eligible rebalance dates are generated.
````

REPLACE the whole block with:

````markdown
- `backtest.start_date` and `backtest.end_date` select the dates on which the strategy may rebalance. Returns before `backtest.start_date` are still used to estimate the first portfolio. The estimation lookback is `backtest.lookback_periods` (default 756 trading days); `--lookback-periods` overrides it for one run. If the data history is too short to fill the lookback at `backtest.start_date`, the first rebalance is delayed, the run logs `backtest_start_delayed`, and the report's assumptions list shows the real first return date.
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
uv run pytest -q tests/unit/test_backtest_window_warmup.py
````

Expected: `5 passed` (warnings are fine).

````bash
uv run pytest -q -x
````

Expected: Last line starts with `340 passed`. No `failed`, no `error`. Takes about 4 minutes.

````bash
grep -c DEFAULT_LOOKBACK_PERIODS src/etf_portfolio/cli.py
````

Expected: `0`

## 8. Commit

````bash
git add -A
git status --short
````

Expected: only the files listed in section 3 appear.

````bash
git commit -m "fix(backtest): version the lookback in config and warm it up before start_date"
````

## 9. Done checklist

- [ ] `backtest.lookback_periods` exists in `src/etf_portfolio/config.py` and `configs/base.yaml`.
- [ ] `DEFAULT_LOOKBACK_PERIODS` no longer appears anywhere in `src/`.
- [ ] `tests/unit/test_backtest_window_warmup.py` exists and its 5 tests pass.
- [ ] The full test suite passes.
- [ ] `git status --short` prints nothing.
- [ ] You did not run `git push`.

## 10. If something goes wrong

- **A FIND block is not in the file.** Check you are on the right branch and that every earlier plan is committed (`git log --oneline | head`). Search for the first line of the FIND block only (`grep -n "<first line>" <file>`). If the surrounding code is different from the plan, STOP and report; do not improvise.
- **`ruff format --check` reports a file would be reformatted.** Run `uv run ruff format <that file>` once, then re-run the verify commands.
- **You want to start this plan over.** Run `git checkout -- . && git clean -fd -- src tests docs configs` (this discards ONLY uncommitted work of this plan), then start again from section 4.
- **`test_history_before_start_date_warms_up_the_lookback` fails with `Backtest produced no rebalance periods`.** The step that edits the `_apply_backtest_window(` call inside `run_backtest` was not applied correctly. `rebalance_dates` must be built from `windowed_asset_returns.index`, and `asset_returns` must be the result of `_history_through_backtest_end(...)`.
- **`NameError: name 'configured_lookback' is not defined`.** The step that adds `configured_lookback = _resolve_lookback_periods(...)` at the top of `run_backtest` is missing.
