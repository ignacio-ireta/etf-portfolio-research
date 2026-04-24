from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from etf_portfolio import cli
from etf_portfolio.config import load_config
from etf_portfolio.data.providers import PriceDataProvider


class FakePriceProvider(PriceDataProvider):
    def __init__(self, prices: pd.DataFrame) -> None:
        self._prices = prices

    def get_prices(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str | None,
    ) -> pd.DataFrame:
        return self._prices.loc[:, tickers]


def test_cli_pipeline_stages_create_expected_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_project_files(tmp_path)
    prices = _make_prices()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "build_price_provider", lambda config: FakePriceProvider(prices))

    assert cli.main(["ingest", "--config", "configs/base.yaml"]) == 0
    assert cli.main(["validate", "--config", "configs/base.yaml"]) == 0
    assert cli.main(["features", "--config", "configs/base.yaml"]) == 0
    assert cli.main(["optimize", "--config", "configs/base.yaml"]) == 0
    assert cli.main(["backtest", "--config", "configs/base.yaml", "--lookback-periods", "5"]) == 0
    assert cli.main(["report", "--config", "configs/base.yaml", "--lookback-periods", "5"]) == 0
    assert cli.main(["ml", "--config", "configs/base.yaml"]) == 0

    assert (tmp_path / "data/raw/prices.parquet").exists()
    assert (tmp_path / "data/processed/prices_validated.parquet").exists()
    assert (tmp_path / "data/processed/returns.parquet").exists()
    assert (tmp_path / "reports/excel/optimized_portfolios.xlsx").exists()
    assert (tmp_path / "reports/excel/portfolio_results.xlsx").exists()
    assert (tmp_path / "reports/figures").exists()
    assert (tmp_path / "reports/html/frontier.html").exists()
    assert (tmp_path / "reports/html/latest_report.html").exists()
    assert (tmp_path / "reports/html/backtest_report.html").exists()
    assert (tmp_path / "reports/metrics/backtest_metrics.json").exists()
    assert (tmp_path / "reports/runs").exists()
    assert (tmp_path / "reports/ml/dataset.parquet").exists()
    assert (tmp_path / "reports/ml/predictions.parquet").exists()
    assert (tmp_path / "reports/ml/model.pkl").exists()
    assert (tmp_path / "reports/ml/governance.json").exists()
    assert (tmp_path / "reports/ml/model_card.md").exists()
    assert (tmp_path / "reports/ml/metrics.json").exists()

    backtest_metrics_text = (tmp_path / "reports/metrics/backtest_metrics.json").read_text(
        encoding="utf-8"
    )
    assert "NaN" not in backtest_metrics_text
    metrics_payload = json.loads(backtest_metrics_text)
    assert metrics_payload["run_id"].startswith("backtest-")
    assert Path(metrics_payload["run_record"]).exists()
    assert "optimized_strategy" in metrics_payload
    assert "benchmarks" in metrics_payload
    assert "Selected Benchmark ETF" in metrics_payload["benchmarks"]
    assert "Equal-Weight ETF Universe" in metrics_payload["benchmarks"]
    assert "60/40 Portfolio" in metrics_payload["benchmarks"]
    assert "Inverse-Volatility Portfolio" in metrics_payload["benchmarks"]
    assert "Previous Optimized Strategy" in metrics_payload["benchmarks"]
    assert any((tmp_path / "reports/figures").glob("*.png"))

    latest_report_html = (tmp_path / "reports/html/latest_report.html").read_text(encoding="utf-8")
    assert "ETF Universe Summary" in latest_report_html
    assert "Portfolio Weights" in latest_report_html
    assert "Assumptions and Limitations" in latest_report_html

    ml_metrics_text = (tmp_path / "reports/ml/metrics.json").read_text(encoding="utf-8")
    assert "NaN" not in ml_metrics_text
    ml_metrics_payload = json.loads(ml_metrics_text)
    assert ml_metrics_payload["run_id"].startswith("ml-")
    assert Path(ml_metrics_payload["run_record"]).exists()
    assert ml_metrics_payload["governance"]["approval_status"] in {"eligible", "research_only"}
    assert "passes_leakage_checks" in ml_metrics_payload["governance"]["checks"]


def test_run_all_executes_reproducible_cli_pipeline(tmp_path: Path, monkeypatch) -> None:
    _write_project_files(tmp_path)
    prices = _make_prices()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "build_price_provider", lambda config: FakePriceProvider(prices))

    assert cli.main(["run-all", "--config", "configs/base.yaml", "--lookback-periods", "5"]) == 0

    assert (tmp_path / "data/raw/prices.parquet").exists()
    assert (tmp_path / "data/processed/prices_validated.parquet").exists()
    assert (tmp_path / "data/processed/returns.parquet").exists()
    assert (tmp_path / "reports/excel/optimized_portfolios.xlsx").exists()
    assert (tmp_path / "reports/html/latest_report.html").exists()
    assert (tmp_path / "reports/metrics/backtest_metrics.json").exists()


def _write_project_files(project_root: Path) -> None:
    (project_root / "configs").mkdir(parents=True, exist_ok=True)
    (project_root / "data/metadata").mkdir(parents=True, exist_ok=True)

    (project_root / "configs/base.yaml").write_text(
        """
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
  max_weight_per_etf: 0.5
  risk_model: sample
  expected_return_estimator: historical_mean
  active_objective: equal_weight
  benchmark_objectives:
    - inverse_volatility
    - min_variance
    - risk_parity
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
ml:
  enabled: true
  task: regression
  target: forward_return
  horizon_periods: 5
  models:
    - historical_mean
    - ridge
  features:
    lag_periods: [1, 5]
    momentum_periods: [5, 10]
    volatility_windows: [5]
    drawdown_windows: [5]
    correlation_windows: [5]
    moving_average_windows: [5]
  validation:
    train_window_periods: 20
    test_window_periods: 5
    step_periods: 5
    min_train_periods: 20
  tracking:
    enable_mlflow: false
    experiment_name: etf_portfolio_research
    artifact_dir: reports/ml
    dataset_version: data/processed/returns.parquet
    feature_version: test
  governance:
    minimum_fold_win_rate: 0.5
    minimum_folds_for_stability: 2
    require_baseline_outperformance: true
    require_leakage_checks: true
""".strip(),
        encoding="utf-8",
    )

    (project_root / "data/metadata/etf_universe.csv").write_text(
        "\n".join(
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
        ),
        encoding="utf-8",
    )


def _make_prices() -> pd.DataFrame:
    index = pd.bdate_range("2020-01-02", periods=80)
    columns = ["VTI", "BND", "IAU", "VT"]
    data = {}
    for position, ticker in enumerate(columns, start=1):
        base_returns = np.full(len(index), 0.0004 * position)
        base_returns += 0.0008 * np.sin(np.arange(len(index)) / (2 + position))
        shock_mask = np.arange(len(index)) % (9 + position) == 0
        base_returns[shock_mask] -= 0.01 + 0.001 * position
        data[ticker] = 100 * np.cumprod(1 + base_returns)

    return pd.DataFrame(data, index=index)


@pytest.mark.parametrize(
    ("active_objective", "expected_method"),
    [
        ("equal_weight", "equal_weight"),
        ("inverse_volatility", "inverse_volatility"),
        ("min_variance", "min_volatility"),
        ("max_sharpe", "max_sharpe"),
        ("risk_parity", "risk_parity"),
    ],
)
def test_select_optimization_method_uses_active_objective_only(
    tmp_path: Path,
    active_objective: str,
    expected_method: str,
) -> None:
    _write_project_files(tmp_path)
    config_path = tmp_path / "configs/base.yaml"
    config_text = config_path.read_text(encoding="utf-8")
    config_text = config_text.replace(
        "active_objective: equal_weight",
        f"active_objective: {active_objective}",
    )
    config_text = config_text.replace(
        """  benchmark_objectives:
    - inverse_volatility
    - min_variance
    - risk_parity
""",
        "  benchmark_objectives: []\n",
    )
    config_path.write_text(config_text, encoding="utf-8")
    config = load_config(config_path)

    assert cli._select_optimization_method(config) == expected_method


def test_apply_backtest_window_filters_asset_and_benchmark_returns(tmp_path: Path) -> None:
    _write_project_files(tmp_path)
    config_path = tmp_path / "configs/base.yaml"
    config_text = config_path.read_text(encoding="utf-8").replace(
        '  start_date: "2020-01-02"\n  end_date: null\n',
        '  start_date: "2020-02-10"\n  end_date: "2020-03-10"\n',
    )
    config_path.write_text(config_text, encoding="utf-8")
    config = load_config(config_path)

    returns = _make_prices().pct_change().dropna()
    asset_returns = cli._asset_returns(returns, config)
    benchmark_returns = cli._benchmark_returns(returns, config)

    windowed_assets, windowed_benchmark = cli._apply_backtest_window(
        asset_returns,
        benchmark_returns,
        config,
    )

    assert windowed_assets.index.min() == pd.Timestamp("2020-02-10")
    assert windowed_assets.index.max() == pd.Timestamp("2020-03-10")
    assert windowed_benchmark.index.min() == pd.Timestamp("2020-02-10")
    assert windowed_benchmark.index.max() == pd.Timestamp("2020-03-10")


def test_backtest_start_date_changes_first_eligible_rebalance_date(tmp_path: Path) -> None:
    _write_project_files(tmp_path)
    returns = _make_prices().pct_change().dropna()
    config_path = tmp_path / "configs/base.yaml"

    early_config = load_config(config_path)
    early_assets, early_benchmark = cli._apply_backtest_window(
        cli._asset_returns(returns, early_config),
        cli._benchmark_returns(returns, early_config),
        early_config,
    )
    early_rebalance_dates = cli._rebalance_dates(
        early_assets.index,
        early_config.rebalance.frequency,
    )

    late_config_text = config_path.read_text(encoding="utf-8").replace(
        '  start_date: "2020-01-02"\n  end_date: null\n',
        '  start_date: "2020-03-16"\n  end_date: null\n',
    )
    config_path.write_text(late_config_text, encoding="utf-8")
    late_config = load_config(config_path)
    late_assets, late_benchmark = cli._apply_backtest_window(
        cli._asset_returns(returns, late_config),
        cli._benchmark_returns(returns, late_config),
        late_config,
    )
    late_rebalance_dates = cli._rebalance_dates(
        late_assets.index,
        late_config.rebalance.frequency,
    )

    assert early_benchmark.index.min() == pd.Timestamp("2020-01-03")
    assert late_benchmark.index.min() == pd.Timestamp("2020-03-16")
    assert early_rebalance_dates[0] == pd.Timestamp("2020-01-31")
    assert late_rebalance_dates[0] == pd.Timestamp("2020-03-31")
    assert early_rebalance_dates[0] < late_rebalance_dates[0]


def test_run_optimize_respects_shared_metadata_constraints(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_constrained_optimize_project_files(tmp_path)
    returns_path = tmp_path / "data/processed/returns.parquet"
    returns_path.parent.mkdir(parents=True, exist_ok=True)
    _make_constrained_optimize_returns().to_parquet(returns_path)

    expected_returns = pd.Series(
        {
            "VTI": 0.03,
            "BND": 0.005,
            "IAU": 0.02,
            "VNQ": 0.04,
            "REMX": 0.12,
        },
        dtype=float,
    )
    covariance_matrix = pd.DataFrame(
        np.diag([0.06**2, 0.03**2, 0.05**2, 0.07**2, 0.09**2]),
        index=expected_returns.index,
        columns=expected_returns.index,
        dtype=float,
    )
    frontier_calls: list[dict[str, object]] = []

    monkeypatch.setattr(cli, "estimate_expected_returns", lambda *args, **kwargs: expected_returns)
    monkeypatch.setattr(
        cli,
        "calculate_covariance_matrix",
        lambda *args, **kwargs: covariance_matrix,
    )

    def fake_build_efficient_frontier_figure(*args, **kwargs):
        frontier_calls.append(kwargs.copy())
        return go.Figure()

    monkeypatch.setattr(
        cli,
        "build_efficient_frontier_figure",
        fake_build_efficient_frontier_figure,
    )

    workbook_path, frontier_path = cli.run_optimize("configs/base.yaml", project_root=tmp_path)

    weights = pd.read_excel(workbook_path, sheet_name="weights", index_col=0)["weight"]
    metadata = pd.read_csv(tmp_path / "data/metadata/etf_universe.csv").set_index("ticker")
    asset_class_weights = weights.groupby(metadata.reindex(weights.index)["asset_class"]).sum()

    assert frontier_path.exists()
    assert weights["REMX"] <= 0.05 + 1e-8
    assert weights["VTI"] >= 0.20 - 1e-8
    assert asset_class_weights["fixed_income"] >= 0.10 - 1e-8
    assert asset_class_weights["real_estate"] <= 0.10 + 1e-8
    assert frontier_calls
    assert frontier_calls[0]["ticker_bounds"] == {"VTI": (0.20, 0.60), "REMX": (0.0, 0.05)}
    assert frontier_calls[0]["asset_class_bounds"] is not None
    assert frontier_calls[0]["asset_class_bounds"]["fixed_income"] == (0.10, 0.45)
    assert frontier_calls[0]["asset_class_bounds"]["real_estate"] == (0.0, 0.10)


def _write_constrained_optimize_project_files(project_root: Path) -> None:
    (project_root / "configs").mkdir(parents=True, exist_ok=True)
    (project_root / "data/metadata").mkdir(parents=True, exist_ok=True)

    (project_root / "configs/base.yaml").write_text(
        """
project:
  name: etf_portfolio_research
  base_currency: USD
universe:
  tickers:
    - VTI
    - BND
    - IAU
    - VNQ
    - REMX
benchmark:
  primary: VT
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
  max_weight_per_etf: 0.8
  risk_model: sample
  expected_return_estimator: historical_mean
  active_objective: max_sharpe
  benchmark_objectives:
    - equal_weight
constraints:
  asset_class_bounds:
    fixed_income:
      min: 0.10
      max: 0.45
    real_estate:
      min: 0.00
      max: 0.10
  ticker_bounds:
    VTI:
      min: 0.20
      max: 0.60
    REMX:
      min: 0.00
      max: 0.05
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
ml:
  enabled: false
""".strip(),
        encoding="utf-8",
    )

    (project_root / "data/metadata/etf_universe.csv").write_text(
        "\n".join(
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
                    "VNQ,Vanguard Real Estate ETF,real_estate,US,USD,0.0013,"
                    "MSCI US Investable Market Real Estate 25/50 Index,false,false,2004-09-23,core"
                ),
                (
                    "REMX,VanEck Rare Earth and Strategic Metals ETF,equity,Global,USD,0.0047,"
                    "MVIS Global Rare Earth/Strategic Metals Index,false,false,2010-10-27,satellite"
                ),
                (
                    "VT,Vanguard Total World Stock ETF,equity,Global,USD,0.0006,"
                    "FTSE Global All Cap Index,false,false,2008-06-24,benchmark"
                ),
            ]
        ),
        encoding="utf-8",
    )


def _make_constrained_optimize_returns() -> pd.DataFrame:
    index = pd.bdate_range("2020-01-02", periods=80)
    columns = ["VTI", "BND", "IAU", "VNQ", "REMX"]
    data = {}
    for position, ticker in enumerate(columns, start=1):
        base_returns = np.full(len(index), 0.0002 * position)
        base_returns += 0.0001 * np.sin(np.arange(len(index)) / (1 + position))
        data[ticker] = 100 * np.cumprod(1 + base_returns)

    return pd.DataFrame(data, index=index)
