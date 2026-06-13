# Security Policy

## Reporting a vulnerability

This is a research project, not a production service. If you find a security or
privacy issue (for example a dependency vulnerability, a secret-handling flaw,
or a way the tool leaks document/market content), please open a private report:

- Email: squeezedgrape@gmail.com with the subject `SECURITY: etf-portfolio-research`.
- Do not open a public issue for an unfixed vulnerability.

Please include reproduction steps and the affected commit/version. Expect an
acknowledgement within a few days.

## Security & privacy posture

The tool is **local-first and non-networked by default**. The only stage that
reaches the network is `ingest`, which fetches public market data from the
configured provider (yfinance by default). Nothing is sent to an external LLM or
cloud service unless you explicitly opt in.

| Area | Default |
| --- | --- |
| Secrets | Read from environment variables only (e.g. `TIINGO_API_KEY`); never committed or logged |
| Document/market content | Not logged; only aggregate metrics and file paths appear in logs |
| External APIs | Off by default; `ingest` fetches public prices; Tiingo requires an opt-in key |
| ML experiment tracking (MLflow) | Off by default; local `./mlruns` unless a backend is configured |
| Untrusted inputs | Config is validated by pydantic; price data is validated before use |
| Dependencies | Pinned via `uv.lock`; scanned by `pip-audit` (CI) and Dependabot |

### Handling secrets

- Copy `.env.example` to `.env` (git-ignored) for local credentials.
- API keys are read from environment variables at call time and are never
  written to run records, logs, or report artifacts.

### Reproducibility vs. confidentiality

Run records (`reports/runs/*.json`) capture SHA-256 hashes of inputs/outputs,
the git commit, and a config hash — but never raw credentials or raw market
content. Review a config before sharing run artifacts if your universe or
provider choice is itself sensitive.

## Supported versions

The project is pre-1.0. Only the latest commit on `main` is supported.
