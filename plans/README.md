# Audit remediation plans

Ten plans that fix the findings of the repository audit. Each plan is a self-contained file
that a small model (or a person) can execute without understanding the rest of the project:
every edit is given as an exact FIND block and an exact REPLACE block, every command has
its expected output, and every plan ends with a commit.

Plans 01 to 09 also ship a **tested patch file** in `plans/patches/`. Applying the patch
is the fast path; the manual FIND/REPLACE steps in the plan are the fallback.

## The plans, in the order they must be executed

| # | Plan file | What it fixes | New tests | Suite total after |
| --- | --- | --- | --- | --- |
| 01 | `01-lookback-in-config-and-warmup.md` | Lookback was a hidden CLI default; backtest silently started 3 years after the configured start | 5 | 340 |
| 02 | `02-consistent-intra-period-returns.md` | Returns assumed a free daily reset while weights assumed buy-and-hold | 4 | 344 |
| 03 | `03-solver-tolerance-ladder.md` | One non-converged SLSQP solve aborted the whole backtest | 4 | 348 |
| 04 | `04-like-for-like-static-benchmarks.md` | 60/40-style benchmarks were rebalanced daily at zero cost | 5 | 353 |
| 05 | `05-report-implementation-gap.md` | The headline portfolio is not the optimizer's portfolio, and nothing said so | 6 | 359 |
| 06 | `06-sharpe-difference-confidence-intervals.md` | Sharpe comparisons had no uncertainty estimate | 11 | 370 |
| 07 | `07-sensitivity-command.md` | No way to see how results move with lookback, risk-free rate, rebalance mode | 3 | 373 |
| 08 | `08-label-dead-config-drop-unused-dependency.md` | Config that does nothing; declared-but-unused `pyportfolioopt` | 5 | 378 |
| 09 | `09-research-log-tooling.md` | The research log was never used | 8 | 386 |
| 10 | `10-regenerate-artifacts-and-log-result.md` | Committed (and published) reports are stale | 0 | 386 |

Each plan depends on all plans before it. **Never skip a plan and never change the order.**
The FIND blocks of a later plan contain text written by an earlier plan.

## How these plans were produced and checked

- Baseline: commit `4b35adc` ("feat(report): restyle the report, frontier page and figures
  as a research note"), whose test suite has **335** tests.
- The edits were developed on the previous commit (`0781b7a`) and then re-verified on
  `4b35adc`: all nine patches apply there in order, and after each one `ruff check`,
  `ruff format --check` and the **full** test suite were run. The totals in the table are
  the results of those runs.
- The plan text and the patch come from one source. Every FIND block was machine-checked to
  occur exactly once in its file, and applying only the FIND/REPLACE blocks from the
  Markdown gives exactly the same files as applying the patches.
- Plan 10 (commands only) was dry-run end to end.

## Before the first plan: one-time setup

Do these once, in order, from the repository root.

1. Make sure nothing is uncommitted except this `plans/` folder:

   ```bash
   git status --short
   ```

   Expected: exactly one line, `?? plans/`.

   **If other files are listed, STOP.** Someone has work in progress. It must be committed
   (or stashed) by its owner first. Do not commit, stash or discard other people's changes
   yourself.

2. Create the working branch and commit the plans on it:

   ```bash
   git checkout -b fix/audit-remediation
   git add plans
   git commit -m "docs: add audit remediation plans"
   ```

3. Install the environment and confirm the baseline:

   ```bash
   uv sync --group dev
   uv run pytest -q
   ```

   Expected: the last line starts with `335 passed`. Write this number down. If it is
   different, read "Test counts" below before you continue.

Then execute Plan 01, Plan 02, ... Plan 10. One plan = one commit.

## Rules that apply to every plan

1. Steps in order. No skipping. No reordering.
2. Change only what a step tells you to change. No extra clean-up, renaming or reformatting.
3. All commands run from the repository root. Python always through `uv run`.
   Never `pip`, never a hand-made virtual environment.
4. A FIND block must match character for character and occurs exactly once. If it does not
   match, stop and report; do not improvise.
5. Never `git push`. Never merge into `main`. Never `git reset --hard`.
6. If any verification does not give the expected result, stop at that plan and report the
   command and its complete output. Do not start the next plan.

## Do not run whitespace fixers on `plans/patches/`

Patch files contain lines that must end with a space. The repository's pre-commit
configuration has `trailing-whitespace` and `end-of-file-fixer` hooks that would silently
corrupt them. The hooks are not installed in this clone, so a normal `git commit` is safe.
Do **not** run `make precommit` or `uv run pre-commit run --all-files` while
`plans/patches/` exists.

## Test counts

The "Suite total after" column assumes the baseline of 335 tests. If your baseline (setup
step 3) is a different number `B`, add `B - 335` to every total in the table and in the
plans. What must always hold after each plan:

- zero failed tests, zero errors;
- the total grew by exactly the "New tests" number of that plan.

## If the repository has moved on since these plans were written

The plans were verified against commit `4b35adc`. If `git log --oneline | head -1` shows a
newer commit when you start:

- Try the fast path of each plan anyway; `git apply --check` tells you safely whether the
  patch still fits.
- If it does not fit, use the manual steps of that plan.
- If a FIND block is also missing, the code that plan changes has been rewritten. Stop and
  report; do not improvise.
- The baseline test count may differ from 335. Use the rule in "Test counts".

The plans deliberately avoid editing `src/etf_portfolio/reporting/plots.py` and
`src/etf_portfolio/reporting/layout.py`, which were being reworked while the plans were
written.

## What happens after Plan 10

Nothing is pushed by these plans. The owner must review the branch, because merging to
`main` and pushing triggers `.github/workflows/deploy.yml`, which publishes `reports/` to
the public showcase site. After the remediation the report itself states that the
strategy's Sharpe advantage over the benchmark is not distinguishable from noise, and the
generated research-log entry records the decision `revisit later`.

## Audit findings that are deliberately NOT covered by these plans

These need a decision by the owner, or are too large to hand to a small model safely.
Do not attempt them as part of this series.

| Finding | Why it is not planned here |
| --- | --- |
| Remove the disabled ML scaffold (`src/etf_portfolio/ml/`, `run_ml`, `ml:` config, model card) | Touches about 20 files. `cli.py` and `tracking.py` import JSON helpers from `ml/train.py`, every test config contains an `ml:` block, and `AppConfig` forbids unknown keys. Needs its own prepared and tested patch. |
| Make `contribution_only` not the default, or rename "Optimized Strategy" | A research-design decision. Plan 05 makes the gap visible instead of deciding for the owner. |
| Universe and bounds chosen with hindsight (QQQ, REMX, per-ticker caps) | Cannot be fixed in code. It needs a pre-registered universe or an out-of-sample period, and an entry in the research log. |
| Constant 3% risk-free rate | Needs a new data source (for example a T-bill series). Plan 07 shows the sensitivity to it. |
| Extend the data before 2011 so the backtest can start in 2012 | Needs a new download (`ingest`), which changes the committed data. Note that REMX only exists since late 2010, so a 756-day lookback cannot start before late 2013 with the current universe. |
| mypy (94 errors) and `pip-audit` are advisory in CI | Making mypy blocking means fixing 94 type errors first. |
| Stale `handoff/` logs and 9 MB of `legacy/` files in git | Housekeeping; needs the owner's decision on what to keep. |
| Golden test covers only an equal-weight run on synthetic data | Plan 03 adds solver tests; a real-data optimizer snapshot is platform-sensitive and needs a tolerance policy. |

## Ready-to-paste instruction for an executing model

Give the model one plan at a time, with this instruction:

> You are working in the repository at the current directory. Execute the plan in
> `plans/<FILE>` exactly as written. Read the whole file first. Then do section 4, then
> section 5 (fast path); only if section 5 fails, do section 6 (manual steps). Even when
> section 5 succeeds, run every step in section 6 that is marked COMMAND STEP. Then do
> sections 7 and 8. Do not change anything the plan does not mention. If any command gives a result
> different from the "Expected" text, stop immediately and show me the command and its full
> output. When you finish, show me the output of `git log --oneline | head -3` and
> `git status --short`.

For Plan 10 the sections are: 4 (before you start), 5 (steps), 6 (verify), 7 (commit).
