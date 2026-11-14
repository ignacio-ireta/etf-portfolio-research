# Plan 10 — Regenerate the published artifacts and log the result

- **Depends on:** Plans 01 to 09 must all be committed
- **Branch:** `fix/audit-remediation`
- **Estimated size:** no source edits; 5 commands, then one commit of regenerated files

## 0. Rules for whoever executes this plan

You are executing one plan out of a numbered series. Follow these rules exactly.

1. Do the steps in order, top to bottom. Do not skip a step. Do not reorder steps.
2. This plan has **no manual file edits**. Every changed file is written by a command.
   Do not open any file in an editor. Do not "fix" a number, a date, or a sentence by hand.
3. Every command is run from the repository root (the folder that contains `pyproject.toml`).
4. Use `uv run ...` for every Python command. Never use `pip`, `python -m venv`, or `conda`.
5. **Never run `etf-portfolio run-all` or `etf-portfolio ingest` in this plan.** They download
   new prices from the internet and replace the committed data. This plan must reuse the
   committed files in `data/processed/`.
6. Never run `git push`. Never merge into `main`. Never run `git reset --hard`.
7. If a command does not give the expected result, STOP. Report the command and its full
   output.

## 1. Goal

After this plan, the files under `reports/` are produced by the corrected code (Plans 01
to 09), a sensitivity grid exists, and `docs/research_log.md` contains a real entry for the
run, generated from the pipeline's own outputs.

## 2. Background (why this change exists)

Audit findings:

- The committed reports were produced in April 2026 by older code. They lack the
  `schema_version` field the current code writes, and they contain the errors fixed by
  Plans 01, 02 and 04.
- `.github/workflows/deploy.yml` copies `reports/**` to a public website whenever a commit
  that touches `reports/` reaches `main`. Stale or wrong reports are therefore public.
- The research log has no entry for any backtest run.

What the new numbers will look like (reference values from a trial run with the committed
data; your values may differ in the last digit, that is fine):

| Quantity | Reference value |
| --- | --- |
| First simulated return date | `2014-02-03` |
| Strategy Sharpe ratio | about `0.66` |
| Primary benchmark (VT) Sharpe ratio | about `0.51` |
| Sharpe difference vs VT, 95% interval | about `[-0.06, +0.36]` (contains zero) |
| Implementation gap | mean about `26%`, constraints broken at `146 of 147` rebalances |
| Sensitivity grid | `16` runs, all `ok` |
| Research-log decision | `revisit later` |

Two WARNING log lines are **expected** during the backtest. They are not errors:

- `backtest_start_delayed` (the data starts in 2011, so a 756-day lookback cannot start in 2012);
- `implementation_gap_large` (the realized portfolio is far from the optimizer target).

The decision `revisit later` is the correct, honest outcome. Do not try to change it.

## 3. Files this plan touches

All of these are written by commands, never by hand:

- `reports/excel/optimized_portfolios.xlsx`, `reports/excel/portfolio_results.xlsx` (rewritten)
- `reports/figures/*.png` (rewritten)
- `reports/html/frontier.html`, `reports/html/latest_report.html`, `reports/html/backtest_report.html` (rewritten)
- `reports/metrics/backtest_metrics.json` (rewritten)
- `reports/metrics/sensitivity.json` (new)
- `reports/runs/optimize_optimize-<timestamp>-<id>.json` (new, if the optimize stage writes one)
- `reports/runs/backtest_backtest-<timestamp>-<id>.json` (new)
- `docs/research_log.md` (one entry appended at the end)

No file under `src/`, `tests/`, `configs/`, `data/` may change.

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

Expected: The line contains `add research-log entry script` (Plan 09's commit).

````bash
ls data/processed/returns.parquet data/processed/prices_validated.parquet
````

Expected: Both paths are printed, no `No such file` error.

## 5. Steps

### Step 1 — regenerate the optimizer outputs

````bash
uv run etf-portfolio optimize --config configs/base.yaml; echo "EXIT=$?"
````

Expected: many JSON log lines, then a last line `EXIT=0`. Takes a few seconds.

### Step 2 — regenerate the backtest report and metrics

````bash
uv run etf-portfolio backtest --config configs/base.yaml; echo "EXIT=$?"
````

Expected: many JSON log lines, including the two expected WARNING events described in
section 2, then a last line `EXIT=0`. Takes about one minute.

If the last line is not `EXIT=0`, STOP and report the last 20 lines of output.

### Step 3 — check what was written

````bash
uv run python - <<'EOF'
import json
from pathlib import Path

metrics = json.loads(Path("reports/metrics/backtest_metrics.json").read_text(encoding="utf-8"))
record = json.loads(Path(metrics["run_record"]).read_text(encoding="utf-8"))
interval = metrics["sharpe_difference_vs_benchmarks"]["Selected Benchmark ETF"]
gap = metrics["implementation_gap"]
print("schema_version:", metrics["schema_version"])
print("first return date:", record["start_date"])
print("lookback:", record["lookback_periods"])
print("strategy sharpe:", round(metrics["optimized_strategy"]["Sharpe Ratio"], 2))
print("interval:", round(interval["ci_lower"], 2), round(interval["ci_upper"], 2))
print("gap:", round(gap["mean_one_way_gap"], 2), gap["rebalances_with_constraint_violation"],
      "of", gap["rebalance_count"])
print("has target row:", "Optimizer Target (Full Rebalance)" in metrics["benchmarks"])
EOF
````

Expected (compare line by line):

- `schema_version: 1.0`
- `first return date:` a date in February 2014 (reference `2014-02-03`)
- `lookback: 756`
- `strategy sharpe:` between `0.60` and `0.72` (reference `0.66`)
- `interval:` two numbers, the first negative and the second positive (reference `-0.06 0.36`)
- `gap:` first number between `0.20` and `0.32`; then two counts (reference `146 of 147`)
- `has target row: True`

If `first return date` is in 2015, Plan 01 is not in effect: STOP.
If `strategy sharpe` is outside the range, STOP and report all seven lines.

### Step 4 — run the sensitivity grid

````bash
uv run etf-portfolio sensitivity --config configs/base.yaml; echo "EXIT=$?"
````

Expected: last line `EXIT=0`. Takes well under a minute.

````bash
uv run python -c "import json; s=json.load(open('reports/metrics/sensitivity.json'))['summary']; print(s['run_count'], s['ok_count'])"
````

Expected: `16 16`

### Step 5 — append the research-log entry

Copy this command exactly. Do not change the hypothesis sentence.

````bash
uv run python scripts/append_research_log_entry.py --hypothesis "After the audit remediation (plans 01 to 09), the max-Sharpe contribution-only strategy has a risk-adjusted advantage over VT that is distinguishable from noise and that comes from the optimizer."; echo "EXIT=$?"
````

Expected: the new entry is printed (sections `### Date`, `### Hypothesis`, `### Config`,
`### Result`, `### Decision`), then `EXIT=0`. With the committed data the decision line is
`revisit later`.

If it prints `ERROR: docs/research_log.md already has an entry for run ...`, the entry was
already appended earlier. Do **not** run it again; continue with section 6.

## 6. Verify

````bash
tail -5 docs/research_log.md
````

Expected: the end of the new entry: a decision (`adopt`, `reject` or `revisit later`) and a
line starting with `Reason:`.

````bash
git status --short -- src tests configs data
````

Expected: Prints nothing. (No source, test, config or data file changed.)

````bash
git status --short | grep -v "^ M reports/" | grep -v "^?? reports/"
````

Expected: exactly one line: ` M docs/research_log.md`

````bash
uv run ruff check .
````

Expected: `All checks passed!`

````bash
uv run pytest -q -x
````

Expected: Last line starts with `386 passed`. No `failed`, no `error`. (If your baseline
test count was different, see `plans/README.md`, section "Test counts".)

## 7. Commit

````bash
git add -A
git status --short
````

Expected: only files under `reports/` and `docs/research_log.md`.

````bash
git commit -m "chore(reports): regenerate artifacts after the audit remediation and log the result"
````

## 8. Done checklist

- [ ] `reports/metrics/backtest_metrics.json` has `schema_version`, `implementation_gap` and
      `sharpe_difference_vs_benchmarks`.
- [ ] `reports/metrics/sensitivity.json` exists.
- [ ] `docs/research_log.md` ends with a generated entry for this run.
- [ ] The full test suite passes.
- [ ] `git status --short` prints nothing.
- [ ] You did not run `git push` and did not merge into `main`.

## 9. Hand back to the repository owner

Stop here and report to the owner, in these words:

> Plans 01 to 10 are committed on branch `fix/audit-remediation`. Nothing was pushed.
> Merging this branch into `main` and pushing will trigger `.github/workflows/deploy.yml`,
> which publishes the regenerated `reports/` to the public showcase site. The regenerated
> report says the Sharpe advantage over VT is not distinguishable from noise (research-log
> decision: see the last entry of `docs/research_log.md`). Please review before publishing.

## 10. If something goes wrong

- **`backtest` exits with a provenance error (`Run tracking requires a real git commit`).**
  You are not inside the repository root, or the repository has no commit. `cd` to the
  folder that contains `pyproject.toml` and try again.
- **`backtest` fails while writing PNG figures (errors mentioning `kaleido` or `Chrome`).**
  STOP and report. Figure export needs a local Chrome/Chromium; installing software is the
  owner's decision.
- **You ran `run-all` or `ingest` by mistake.** Run `git checkout -- data` to restore the
  committed data, then `git status --short` and report what else changed.
- **You want to start this plan over.** Run
  `git checkout -- reports docs/research_log.md && git clean -fd -- reports`
  (this discards ONLY the uncommitted output of this plan), then start again from section 4.
