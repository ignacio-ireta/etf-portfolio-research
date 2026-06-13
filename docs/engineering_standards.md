# Engineering Standards

This document adapts a general engineering-standards rubric (sections **A–S**,
originally written for a document-to-Markdown extraction tool) to this project,
**etf-portfolio-research** — a reproducible ETF portfolio research pipeline. It
states what each standard means *here* and tracks compliance.

The rubric was written around a tool that walks a folder of documents and emits
normalized Markdown. The concepts map cleanly onto a research pipeline:

| Rubric concept (docmd) | This project |
| --- | --- |
| Folder of input documents / per-file loop | The ETF universe + the ordered **pipeline stages** (ingest → validate → features → optimize → backtest; optional ml) |
| Format extractors (docx/pdf/…) | Data providers (`data/providers.py`) + estimators / optimizers / rebalancers |
| Intermediate `Document` model | Validated price/return DataFrames + structured result objects, separated from rendering |
| Markdown output contract | The HTML / Excel / JSON **report bundle** contract (`docs/output_contract.md`, `metric_dictionary`) |
| Per-file manifest with SHA-256 | Per-stage **run records** (`reports/runs/*.json`) + the **pipeline state manifest** |
| Skip unsupported/corrupt files, don't crash | Skip / mark failed **stages**; structured errors; bounded failure |
| OCR opt-in / networked-off-by-default | Provider network (yfinance default, Tiingo opt-in) + MLflow opt-in; no auto-send to external LLMs |
| Golden Markdown files | Golden **metrics JSON** snapshot (`tests/golden/`) |

## A. Product definition and scope

The tool ingests market data for a configured ETF **universe**, validates it,
computes returns, optimizes a portfolio, runs a walk-forward **backtest**, and
writes a normalized **report bundle** (HTML + Excel + JSON metrics + figures)
plus a provenance **run record**, suitable for downstream analysis and LLM
ingestion. An optional ML track trains and governs baseline models.

- **Success:** given a config, produce the report bundle for the universe plus a
  run record; the same config + data + seed reproduces identical numeric outputs
  (except timestamps / run ids).
- **Failure handling:** unsupported configs, infeasible constraints, insufficient
  history, or corrupt data are surfaced as structured domain errors with
  meaningful exit codes — never silent crashes or half-written outputs.

## B. Repository structure

`src/` layout with the importable package under `src/etf_portfolio`, tests
outside the package, `pyproject.toml` + committed `uv.lock`. The pipeline is
formalized under `src/etf_portfolio/pipeline/` (runner, steps via the CLI stage
list, state, progress). Governance files (`LICENSE`, `CHANGELOG.md`,
`CONTRIBUTING.md`, `SECURITY.md`, `.env.example`) live at the root.

## C. Version control

Git from day one; `main` is the integration branch; feature branches for changes.
Small commits with conventional-commit-style messages
(`feat(pipeline): …`, `fix(report): …`). `CHANGELOG.md` follows *Keep a Changelog*.
The package version is dynamic (`src/etf_portfolio/__init__.py::__version__`,
exposed via `etf-portfolio --version`); tag stable versions `vX.Y.Z`. Secrets are
never committed; generated `data/` and `reports/` artifacts are versioned
**intentionally** for the offline handoff bundle (see accepted deviations).

## D. Dependency and environment management

Declared in `pyproject.toml`, isolated and reproducible via `uv` + committed
`uv.lock`. Python is pinned (`>=3.11,<3.14`). Dev dependencies are split by
purpose into PEP 735 groups — `test`, `lint`, `typecheck`, `security`, `docs` —
aggregated by `dev`. Optional capabilities are extras: `ml` (MLflow) is
off-by-default. Native dependencies: none beyond Python wheels (`kaleido` renders
figures headlessly).

## E. Configuration

Config is fully separated from code: typed, validated YAML via pydantic
(`config.py`, `configs/*.yaml`). Runtime/environment-varying behavior is set by
CLI flags: `--config`, `--log-level`, `--log-file`, `--resume`,
`--fail-fast/--continue`, `--lookback-periods`, `--version`.

## F. Architecture and modularity

Discovery/ingest → validation → intermediate model (validated prices/returns +
typed result objects) → optimization/backtest → report writing → run
state/reporting. Extraction is separated from rendering: `reporting/` converts
result objects into HTML/Excel/figures, so extraction bugs and rendering bugs are
independently testable. Orchestration lives in `pipeline/runner.py`, not in each
stage.

## G. Input/output contracts

Inputs (ETF metadata, prices, returns) and outputs (HTML report, Excel workbook,
metrics JSON, run records) are documented in `docs/output_contract.md`,
`docs/data_dictionary.md`, and `metric_dictionary`. JSON outputs carry a
`schema_version`. Output is deterministic for a fixed config + data + seed,
excluding timestamps and run ids.

## H. Error handling

Domain-specific exceptions in `errors.py` (`ConfigError`, `DataValidationError`,
`InfeasibleConstraintsError`, `InsufficientHistoryError`, `ProvenanceError`,
`MLDisabledError`, `PipelineInterrupted`) each carry a stable `code` and a process
`exit_code`. The CLI maps them to **meaningful exit codes** (0 ok, 2 config,
3 data, 4 infeasible/insufficient, 5 provenance, 130 interrupted, 1 unexpected).
A run that fails or is interrupted writes a structured `reports/runs/<run_id>/errors.json`.
One failed stage does not silently corrupt outputs (atomic writes); `run-all`
defaults to fail-fast, with `--continue` to attempt independent stages.

## I. Logging and observability

Structured single-line JSON logs (`logging_config.py`) to stderr, with
`--log-level` control and tracebacks only at debug. `run-all` writes a per-run log
to `reports/runs/<run_id>/run.log` (or `--log-file`). A human-readable console
summary (per-stage status + counts) prints to stdout. Persistent observability —
run records, pipeline state, errors.json — matters more than the terminal output.

## J. Resumability and idempotency

`run-all` persists a state manifest (`reports/runs/pipeline_state.json`) with
per-stage status, input hash, and output SHA-256s. `--resume` skips stages whose
**input hash** (relevant config sections + input-file SHA-256s + params) is
unchanged and whose outputs are intact; changes cascade downstream. Output writes
are **atomic** (temp file + `os.replace`), so an interrupted write never leaves a
half-written final file. Re-running an unchanged stage is idempotent.

## K. Testing strategy

Unit (functions/classes), integration (multi-component on fixtures), e2e (drive
the CLI like a user), golden (snapshot the metrics contract), regression
(no-lookahead / leakage), and property/fuzz (Hypothesis). Markers segment the
suite (`slow`, `integration`, `e2e`, `golden`, `property`). See `docs/testing.md`.

## L. Testing documentation

`docs/testing.md` explains how to run the whole suite or a subset, how fixtures
are organized (`tests/conftest.py`), how to regenerate golden files
(`UPDATE_GOLDEN=1`), what runs in CI, and what is intentionally slow.

## M. Code quality gates

`ruff` (lint + format, expanded rule set), `mypy` (lenient/gradual on `src`),
`pytest` + `pytest-cov`, `pip-audit`, and `pre-commit`. Run locally with
`make check` or `uv run pre-commit run --all-files`.

> **Type-checking maturity:** mypy runs in lenient mode (no `disallow_untyped_defs`)
> and currently reports a known baseline of pandas-stubs findings. New modules are
> kept clean; the baseline is reduced incrementally. CI runs mypy as advisory
> (non-blocking) until the baseline is clear, then it flips to blocking.

## N. CI/CD

GitHub Actions runs on push to `main` and on PRs across Python 3.11–3.13:
install → ruff lint → ruff format check → **mypy (advisory)** → pytest **with
coverage** → **pip-audit (advisory)** → upload coverage. Actions are pinned.
Dependabot watches `uv` and `github-actions`. CD/publish is intentionally
deferred (CI-first maturity).

## O. Documentation

`README.md` (what/install/quickstart/CLI), `docs/` (architecture, methodology,
data & metric dictionaries, output contract, testing, troubleshooting,
interpretation, trust & safety, model card, …), `CHANGELOG.md`, `CONTRIBUTING.md`,
`SECURITY.md`, `.env.example`.

## P. Security and privacy

Local-first, non-networked by default; only `ingest` reaches the network (public
prices). Secrets are read from environment variables, never committed or logged.
Document/market content is not logged (only aggregate metrics + paths). Office
macros / arbitrary code are never executed. External APIs and ML/cloud tracking
are opt-in. Dependencies are pinned and scanned (`pip-audit`, Dependabot). See
`SECURITY.md`.

## Q. Scalability

The pipeline handles a large universe and long histories by slicing trailing
lookback windows per rebalance (bounded memory rather than materializing all
windows), skipping unchanged stages on resume, isolating per-stage failures, and
writing outputs atomically. Work is currently single-process and deterministic;
parallelism, if added, will be bounded and opt-in. Concretely: it processes the
configured universe across a multi-year daily history with a manifest-driven
resumable run and reproducible, hashed artifacts — not "it is scalable."

## R. Graceful interruption

SIGINT/SIGTERM are caught by the runner and checked **between stages**: the
in-flight stage finishes (stages are not safely abortable mid-compute), remaining
stages are marked `interrupted`, a summary + resume hint is printed, `errors.json`
is written, and the process exits **130**. The next `run-all --resume` continues.

## S. Traceability

Each stage writes a run record (`reports/runs/<stage>_<run_id>.json`) with
`schema_version`, `pipeline_steps`, git commit hash, config hash, `universe_id`,
input/output SHA-256s, and method metadata. Source → output, step, dependency,
error (`errors.json`), content (metric/section provenance in the report), and run
(config hash + git) traceability are all captured. This is the project's
strongest area and predates these standards.

---

## Compliance status

| § | Area | Status |
| --- | --- | --- |
| A | Product definition & scope | Met |
| B | Repository structure | Met |
| C | Version control | Met (tagging is a manual release step) |
| D | Dependency & environment management | Met |
| E | Configuration | Met |
| F | Architecture & modularity | Met |
| G | I/O contracts | Met |
| H | Error handling | Met |
| I | Logging & observability | Met |
| J | Resumability & idempotency | Met |
| K | Testing strategy | Met |
| L | Testing documentation | Met |
| M | Code quality gates | Met (mypy lenient, baseline being reduced) |
| N | CI/CD | Met (CD deferred; new gates advisory first) |
| O | Documentation | Met |
| P | Security & privacy | Met |
| Q | Scalability | Met (single-process; parallelism is future work) |
| R | Graceful interruption | Met |
| S | Traceability | Met |

## Accepted deviations

- **argparse, not typer/rich.** The CLI uses the standard library; progress is a
  textual summary, not a live TTY bar. Functionally equivalent, less churn.
- **`data/` and `reports/` are committed.** This is intentional for the offline,
  reproducible handoff bundle — not stray generated files. `check-added-large-files`
  excludes those paths. Git LFS is a possible future move.
- **CD/publish deferred.** The project is CI-first; there is no PyPI/Docker release
  yet.
- **mypy lenient + advisory CI.** Static typing is adopted gradually; see §M.
