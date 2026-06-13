"""Shared test fixtures consolidated for unit, integration, e2e, and golden tests.

Historically each CLI test redefined its own ``FakePriceProvider`` and synthetic
price helpers. New tests should import these fixtures instead. See
docs/testing.md for how the suite is organized.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from etf_portfolio.config import AppConfig, load_config
from etf_portfolio.data.providers import PriceDataProvider
from etf_portfolio.logging_config import reset_logging


@pytest.fixture(autouse=True)
def _reset_logging_handlers() -> Iterator[None]:
    """Detach every handler ``configure_logging`` attached during a test.

    ``configure_logging`` adds a shared stderr handler (and, with ``log_file=...``,
    a FileHandler) to the root logger. Left attached across tests they accumulate,
    reference deleted tmp paths, and — for the stderr handler — write to pytest's
    closed per-test captured stream, flooding the suite with ``ValueError: I/O
    operation on closed file``. Clearing them on teardown keeps each test isolated;
    ``configure_logging`` re-adds them on demand in the next test.
    """

    yield
    reset_logging()


# A small but realistic universe (3 assets + 1 benchmark) used across tests.
TINY_CONFIG_YAML = """
project:
  name: etf_portfolio_research
  base_currency: USD
universe:
  tickers:
    - VTI
    - BND
    - IAU
benchmark:
  primary: VT
  secondary:
    global_60_40:
      VT: 0.60
      BND: 0.40
data:
  provider: yfinance
  start_date: "2020-01-01"
  end_date: null
  price_field: adjusted_close
investor_profile:
  horizon_years: 35
  objective: long_term_accumulation
  tax_preference: minimize_realized_gains
optimization:
  long_only: true
  default_max_weight_per_etf: 0.5
  risk_model: sample
  expected_return_estimator: historical_mean
  active_objective: equal_weight
  benchmark_objectives:
    - inverse_volatility
    - min_variance
constraints:
  asset_class_bounds: {}
  ticker_bounds: {}
rebalance:
  mode: contribution_only
  frequency: monthly
  fallback_sell_allowed: false
  fallback:
    sell_allowed_if_absolute_drift_exceeds: 0.10
  contribution_amount: 500.0
  tolerance_bands:
    per_ticker_abs_drift: 0.05
    per_asset_class_abs_drift: 0.10
backtest:
  start_date: "2020-01-02"
  end_date: null
  initial_capital: 50000.0
costs:
  transaction_cost_bps: 2
  slippage_bps: 1
tracking:
  artifact_dir: reports/runs
  require_git_commit: true
ml:
  enabled: false
""".strip()

ETF_METADATA_CSV = "\n".join(
    [
        (
            "ticker,name,asset_class,region,currency,expense_ratio,"
            "benchmark_index,is_leveraged,is_inverse,inception_date,role"
        ),
        (
            "VTI,Vanguard Total Stock Market ETF,equity,US,USD,0.0003,"
            "CRSP US Total Market Index,false,false,2001-05-24,core"
        ),
        (
            "BND,Vanguard Total Bond Market ETF,fixed_income,US,USD,0.0003,"
            "Bloomberg US Aggregate Float Adjusted Index,false,false,2007-04-03,core"
        ),
        (
            "IAU,iShares Gold Trust,commodity,Global,USD,0.0025,"
            "LBMA Gold Price,false,false,2005-01-21,core"
        ),
        (
            "VT,Vanguard Total World Stock ETF,equity,Global,USD,0.0006,"
            "FTSE Global All Cap Index,false,false,2008-06-24,benchmark"
        ),
    ]
)


class FakePriceProvider(PriceDataProvider):
    """A deterministic in-memory price provider for tests (no network)."""

    def __init__(self, prices: pd.DataFrame) -> None:
        self._prices = prices

    def get_prices(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str | None,
    ) -> pd.DataFrame:
        return self._prices.loc[:, tickers]


def make_synthetic_prices(*, periods: int = 80) -> pd.DataFrame:
    """Build a deterministic adjusted-price panel for VTI/BND/IAU/VT."""

    index = pd.bdate_range("2020-01-02", periods=periods)
    columns = ["VTI", "BND", "IAU", "VT"]
    data = {}
    for position, ticker in enumerate(columns, start=1):
        base_returns = np.full(len(index), 0.0004 * position)
        base_returns += 0.0008 * np.sin(np.arange(len(index)) / (2 + position))
        shock_mask = np.arange(len(index)) % (9 + position) == 0
        base_returns[shock_mask] -= 0.01 + 0.001 * position
        data[ticker] = 100 * np.cumprod(1 + base_returns)
    return pd.DataFrame(data, index=index)


def init_git_repo(project_root: Path) -> None:
    """Initialize a committed git repo so provenance resolution succeeds."""

    subprocess.run(["git", "init"], cwd=project_root, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "tests@example.com"], cwd=project_root, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test Runner"], cwd=project_root, check=True)
    subprocess.run(
        ["git", "add", "configs/base.yaml", "data/metadata/etf_universe.csv"],
        cwd=project_root,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial project files"],
        cwd=project_root,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    return make_synthetic_prices()


@pytest.fixture
def tiny_config(tmp_path: Path) -> AppConfig:
    """A valid :class:`AppConfig` loaded from the tiny test config."""

    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(TINY_CONFIG_YAML, encoding="utf-8")
    return load_config(config_path)


@pytest.fixture
def tiny_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A ready-to-run project: config + metadata + git repo, with a fake provider.

    The working directory is changed to the project root and
    ``cli.build_price_provider`` is patched to a deterministic in-memory provider,
    so ``cli.main([...])`` runs the full pipeline offline.
    """

    from etf_portfolio import cli

    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data/metadata").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs/base.yaml").write_text(TINY_CONFIG_YAML, encoding="utf-8")
    (tmp_path / "data/metadata/etf_universe.csv").write_text(ETF_METADATA_CSV, encoding="utf-8")
    init_git_repo(tmp_path)

    prices = make_synthetic_prices()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "build_price_provider", lambda config: FakePriceProvider(prices))
    return tmp_path
