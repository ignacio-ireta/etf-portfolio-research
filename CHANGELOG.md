# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

Brought the project into compliance with the adapted engineering standards
(`docs/engineering_standards.md`).

### Added
- Domain error taxonomy (`errors.py`) with stable codes and meaningful CLI exit
  codes (config=2, data=3, infeasible/insufficient=4, provenance=5, interrupted=130).
- Resumable pipeline runner (`pipeline/` package): per-stage state manifest
  (`reports/runs/pipeline_state.json`), `--resume` with skip-unchanged-by-input-hash,
  graceful SIGINT/SIGTERM handling between stages, and a console summary.
- Atomic output writes (`io_utils.py`) for all generated artifacts (temp + `os.replace`).
- CLI flags: `--version`, `--log-level`, `--log-file`, `--resume`,
  `--fail-fast`/`--continue`; per-run log file for `run-all`.
- `schema_version` and `pipeline_steps` on run records and JSON outputs;
  `docs/output_contract.md`.
- Per-run `errors.json` for failed/interrupted runs.
- Test breadth: `tests/integration/`, `tests/e2e/`, `tests/golden/`, property/fuzz
  tests, a shared `tests/conftest.py`, and pytest markers.
- Tooling: mypy (lenient), `pytest-cov`, `pip-audit`, `pre-commit`, split PEP 735
  dependency groups, and an optional `ml` extra.
- CI: advisory mypy/coverage/pip-audit steps + coverage artifact; Dependabot config.
- Governance docs: `docs/engineering_standards.md`, `docs/testing.md`,
  `docs/troubleshooting.md`, `LICENSE` (MIT), `CONTRIBUTING.md`, `SECURITY.md`,
  populated `.env.example`.

### Changed
- Data validation and provenance failures now raise typed domain exceptions
  (still subclassing `ValueError`/`RuntimeError` for backward compatibility).
- Package version is dynamic, sourced from `etf_portfolio.__version__`.
- Expanded ruff rule set (bugbear, comprehensions, simplify, ruff-specific).

## [0.1.0]

- Initial reproducible ETF portfolio research pipeline: universe definition, data
  ingestion + validation, returns, optimization, walk-forward backtesting,
  reporting, run records, and an optional ML track.
