"""Command-line entrypoints for the research pipeline."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from plotly.io import write_html

from etf_portfolio.backtesting.engine import run_walk_forward_backtest
from etf_portfolio.backtesting.metrics import calculate_sharpe_ratio
from etf_portfolio.config import AppConfig, OptimizationObjective, load_config
from etf_portfolio.data.ingest import ingest_price_data, load_etf_universe_metadata
from etf_portfolio.data.providers import (
    PriceDataProvider,
    TiingoPriceProvider,
    YFinancePriceProvider,
)
from etf_portfolio.data.validate import validate_price_data
from etf_portfolio.features.estimators import (
    calculate_covariance_matrix,
    estimate_expected_returns,
)
from etf_portfolio.features.returns import simple_returns
from etf_portfolio.features.risk_free import get_risk_free_rate
from etf_portfolio.logging_config import configure_logging, get_logger, log_event
from etf_portfolio.ml.dataset import build_ml_dataset
from etf_portfolio.ml.evaluate import chronological_train_test_split, walk_forward_evaluate
from etf_portfolio.ml.governance import (
    evaluate_leakage_checks,
    evaluate_model_governance,
    write_model_card,
)
from etf_portfolio.ml.train import (
    fit_model,
    log_mlflow_run,
    save_model_bundle,
    write_metrics_json,
)
from etf_portfolio.optimization.optimizer import OptimizationMethod, optimize_portfolio
from etf_portfolio.reporting.plots import build_efficient_frontier_figure
from etf_portfolio.reporting.report import generate_report_bundle
from etf_portfolio.reporting.tables import build_metrics_table
from etf_portfolio.tracking import (
    build_run_record,
    generate_run_id,
    relative_to_project_root,
    resolve_run_provenance,
    write_run_record,
)

DEFAULT_LOOKBACK_PERIODS = 756
LOGGER = get_logger(__name__)
FIXED_INCOME_ASSET_CLASS = "fixed_income"
ML_MODEL_TRAINING_SCOPE = "train_split"
ML_MODEL_TRAINING_SCOPE_DESCRIPTION = (
    "chronological train split only; the final test window remains held out for "
    "evaluation and governance"
)


@dataclass(frozen=True)
class BenchmarkSuiteArtifacts:
    returns: dict[str, pd.Series]
    weights: dict[str, pd.DataFrame]


def main(argv: list[str] | None = None) -> int:
    """Run the configured research pipeline stage."""

    configure_logging()
    parser = _build_parser()
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    project_root = Path.cwd()
    run_id = generate_run_id(args.command)

    try:
        if args.command == "ingest":
            run_ingest(config_path, project_root=project_root, run_id=run_id)
        elif args.command == "validate":
            run_validate(config_path, project_root=project_root, run_id=run_id)
        elif args.command == "features":
            run_features(config_path, project_root=project_root, run_id=run_id)
        elif args.command == "optimize":
            run_optimize(config_path, project_root=project_root, run_id=run_id)
        elif args.command == "report":
            run_report(
                config_path,
                project_root=project_root,
                lookback_periods=args.lookback_periods,
                run_id=run_id,
            )
        elif args.command == "run-all":
            run_all(
                config_path,
                project_root=project_root,
                lookback_periods=args.lookback_periods,
                run_id=run_id,
            )
        elif args.command == "ml":
            run_ml(config_path, project_root=project_root, run_id=run_id)
        else:
            run_backtest(
                config_path,
                project_root=project_root,
                lookback_periods=args.lookback_periods,
                run_id=run_id,
            )
    except Exception as exc:
        log_event(
            LOGGER,
            logging.ERROR,
            "pipeline_stage_failed",
            run_id=run_id,
            stage=args.command,
            config_path=str(config_path),
            error_type=type(exc).__name__,
            reason=str(exc),
        )
        raise

    return 0


def run_ingest(
    config_path: str | Path,
    *,
    project_root: Path = Path("."),
    run_id: str | None = None,
) -> Path:
    """Fetch and persist raw price history for universe and benchmark assets."""

    config = _load_project_config(config_path, project_root)
    stage_run_id = run_id or generate_run_id("ingest")
    provider = build_price_provider(config)
    metadata = load_etf_universe_metadata(_metadata_path(project_root))
    tickers = _required_tickers(config)
    log_event(
        LOGGER,
        logging.INFO,
        "pipeline_stage_started",
        run_id=stage_run_id,
        stage="ingest",
        config_path=str(config_path),
        tickers_requested=tickers,
        start_date=config.data.start_date.isoformat(),
        end_date=config.data.end_date.isoformat() if config.data.end_date is not None else None,
    )
    cross_check_provider: PriceDataProvider | None = None
    if config.data.cross_check.enabled and config.data.cross_check.provider is not None:
        cross_check_provider = build_named_price_provider(
            config.data.cross_check.provider,
        )
    artifacts = ingest_price_data(
        provider,
        tickers,
        start_date=config.data.start_date.isoformat(),
        end_date=config.data.end_date.isoformat() if config.data.end_date is not None else None,
        metadata=metadata,
        benchmark_ticker=config.benchmark.primary,
        raw_dir=project_root / "data/raw",
        cross_check_provider=cross_check_provider,
        cross_check_max_relative_divergence=(config.data.cross_check.max_relative_divergence),
        cross_check_min_overlap=config.data.cross_check.min_overlap_observations,
    )
    log_event(
        LOGGER,
        logging.INFO,
        "pipeline_stage_completed",
        run_id=stage_run_id,
        stage="ingest",
        config_path=str(config_path),
        tickers_requested=tickers,
        tickers_loaded=artifacts.raw_prices.columns.tolist(),
        start_date=artifacts.raw_prices.index.min().date().isoformat(),
        end_date=artifacts.raw_prices.index.max().date().isoformat(),
        missing_data_ratio=float(artifacts.validation_result.missing_data_fraction.max()),
        output_path=str(artifacts.raw_prices_path),
    )
    return artifacts.raw_prices_path


def run_validate(
    config_path: str | Path,
    *,
    project_root: Path = Path("."),
    run_id: str | None = None,
) -> Path:
    """Validate raw prices and persist the cleaned price frame."""

    config = _load_project_config(config_path, project_root)
    stage_run_id = run_id or generate_run_id("validate")
    metadata = load_etf_universe_metadata(_metadata_path(project_root))
    raw_prices = pd.read_parquet(project_root / "data/raw/prices.parquet")
    log_event(
        LOGGER,
        logging.INFO,
        "pipeline_stage_started",
        run_id=stage_run_id,
        stage="validate",
        config_path=str(config_path),
        tickers_loaded=raw_prices.columns.tolist(),
        start_date=raw_prices.index.min().date().isoformat(),
        end_date=raw_prices.index.max().date().isoformat(),
    )
    validation_result = validate_price_data(
        raw_prices,
        metadata=metadata,
        benchmark_ticker=config.benchmark.primary,
    )
    validated_prices = raw_prices.sort_index()
    output_path = project_root / "data/processed/prices_validated.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validated_prices.to_parquet(output_path)
    _write_validation_summary(
        validation_result,
        project_root / "reports/metrics/validation_summary.json",
    )
    if not validation_result.suspicious_jumps.empty:
        log_event(
            LOGGER,
            logging.WARNING,
            "price_validation_suspicious_jumps_detected",
            run_id=stage_run_id,
            stage="validate",
            config_path=str(config_path),
            suspicious_jump_count=int(len(validation_result.suspicious_jumps)),
            missing_data_ratio=float(validation_result.missing_data_fraction.max()),
        )
    log_event(
        LOGGER,
        logging.INFO,
        "pipeline_stage_completed",
        run_id=stage_run_id,
        stage="validate",
        config_path=str(config_path),
        tickers_loaded=validated_prices.columns.tolist(),
        start_date=validated_prices.index.min().date().isoformat(),
        end_date=validated_prices.index.max().date().isoformat(),
        missing_data_ratio=float(validation_result.missing_data_fraction.max()),
        output_path=str(output_path),
    )
    return output_path


def run_features(
    config_path: str | Path,
    *,
    project_root: Path = Path("."),
    run_id: str | None = None,
) -> Path:
    """Create periodic returns from validated prices."""

    _ = _load_project_config(config_path, project_root)
    stage_run_id = run_id or generate_run_id("features")
    validated_prices = pd.read_parquet(project_root / "data/processed/prices_validated.parquet")
    log_event(
        LOGGER,
        logging.INFO,
        "pipeline_stage_started",
        run_id=stage_run_id,
        stage="features",
        config_path=str(config_path),
        tickers_loaded=validated_prices.columns.tolist(),
        start_date=validated_prices.index.min().date().isoformat(),
        end_date=validated_prices.index.max().date().isoformat(),
    )
    returns = simple_returns(validated_prices)
    output_path = project_root / "data/processed/returns.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    returns.to_parquet(output_path)
    log_event(
        LOGGER,
        logging.INFO,
        "pipeline_stage_completed",
        run_id=stage_run_id,
        stage="features",
        config_path=str(config_path),
        tickers_loaded=returns.columns.tolist(),
        start_date=returns.index.min().date().isoformat(),
        end_date=returns.index.max().date().isoformat(),
        output_path=str(output_path),
    )
    return output_path


def run_optimize(
    config_path: str | Path,
    *,
    project_root: Path = Path("."),
    run_id: str | None = None,
) -> tuple[Path, Path]:
    """Optimize a portfolio and persist workbook and frontier chart artifacts."""

    config = _load_project_config(config_path, project_root)
    stage_run_id = run_id or generate_run_id("optimize")
    returns = pd.read_parquet(project_root / "data/processed/returns.parquet")
    asset_returns = _asset_returns(returns, config)
    log_event(
        LOGGER,
        logging.INFO,
        "pipeline_stage_started",
        run_id=stage_run_id,
        stage="optimize",
        config_path=str(config_path),
        tickers_requested=config.universe.tickers,
        tickers_loaded=asset_returns.columns.tolist(),
        start_date=asset_returns.index.min().date().isoformat(),
        end_date=asset_returns.index.max().date().isoformat(),
    )
    expected_returns = estimate_expected_returns(
        asset_returns,
        method=config.optimization.expected_return_estimator,
    )
    covariance_matrix = calculate_covariance_matrix(
        asset_returns,
        method=config.optimization.risk_model,
    )
    metadata = load_etf_universe_metadata(_metadata_path(project_root)).set_index("ticker")
    optimization_constraints = _build_optimization_constraints(
        config,
        metadata,
        asset_returns.columns,
    )

    method = _select_optimization_method(config)
    weights = optimize_portfolio(
        expected_returns,
        covariance_matrix,
        method=method,
        max_weight=config.optimization.default_max_weight_per_etf,
        risk_free_rate=get_risk_free_rate(config),
        **optimization_constraints,
    )

    workbook_path = project_root / "reports/excel/optimized_portfolios.xlsx"
    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        weights.rename("weight").to_frame().to_excel(writer, sheet_name="weights")
        expected_returns.rename("expected_return").to_frame().to_excel(
            writer,
            sheet_name="expected_returns",
        )
        covariance_matrix.to_excel(writer, sheet_name="covariance")

    frontier_path = project_root / "reports/html/frontier.html"
    frontier_path.parent.mkdir(parents=True, exist_ok=True)
    write_html(
        build_efficient_frontier_figure(
            expected_returns,
            covariance_matrix,
            portfolio_weights=weights,
            max_weight=config.optimization.default_max_weight_per_etf,
            risk_free_rate=get_risk_free_rate(config),
            **optimization_constraints,
        ),
        file=str(frontier_path),
        full_html=True,
        include_plotlyjs="inline",
    )
    portfolio_return = float(weights.dot(expected_returns))
    portfolio_volatility = float((weights.T @ covariance_matrix @ weights) ** 0.5)
    portfolio_sharpe = calculate_sharpe_ratio(
        portfolio_return,
        portfolio_volatility,
        get_risk_free_rate(config),
    )
    log_event(
        LOGGER,
        logging.INFO,
        "pipeline_stage_completed",
        run_id=stage_run_id,
        stage="optimize",
        config_path=str(config_path),
        optimizer_status="success",
        solver_used="SLSQP",
        portfolio_return=portfolio_return,
        portfolio_volatility=portfolio_volatility,
        portfolio_sharpe=portfolio_sharpe,
        max_weight=float(weights.max()),
        output_path=[str(workbook_path), str(frontier_path)],
    )

    return workbook_path, frontier_path


def run_backtest(
    config_path: str | Path,
    *,
    project_root: Path = Path("."),
    lookback_periods: int = DEFAULT_LOOKBACK_PERIODS,
    run_id: str | None = None,
    persist_metrics: bool = True,
) -> tuple[Path, Path]:
    """Run walk-forward backtesting and persist report and metrics artifacts."""

    config = _load_project_config(config_path, project_root)
    stage_run_id = run_id or generate_run_id("backtest")
    tracking_provenance = (
        resolve_run_provenance(config=config, project_root=project_root)
        if persist_metrics
        else None
    )
    returns = pd.read_parquet(project_root / "data/processed/returns.parquet")
    validated_prices_path = project_root / "data/processed/prices_validated.parquet"
    validated_prices = (
        pd.read_parquet(validated_prices_path) if validated_prices_path.exists() else None
    )
    metadata = load_etf_universe_metadata(_metadata_path(project_root)).set_index("ticker")
    asset_returns = _asset_returns(returns, config)
    benchmark_returns = _benchmark_returns(returns, config)
    common_dates = asset_returns.index.intersection(benchmark_returns.index)
    asset_returns = asset_returns.loc[common_dates]
    benchmark_returns = benchmark_returns.loc[common_dates]
    asset_returns, benchmark_returns = _apply_backtest_window(
        asset_returns,
        benchmark_returns,
        config,
    )

    rebalance_dates = _rebalance_dates(asset_returns.index, config.rebalance.frequency)
    effective_lookback = min(lookback_periods, len(asset_returns.index) - 1)
    if effective_lookback <= 0:
        raise ValueError("Not enough return history is available to run the backtest.")

    method = _select_optimization_method(config)
    risk_free_rate = get_risk_free_rate(config)
    log_event(
        LOGGER,
        logging.INFO,
        "pipeline_stage_started",
        run_id=stage_run_id,
        stage="backtest",
        config_path=str(config_path),
        tickers_requested=config.universe.tickers,
        tickers_loaded=asset_returns.columns.tolist(),
        start_date=asset_returns.index.min().date().isoformat(),
        end_date=asset_returns.index.max().date().isoformat(),
        risk_free_rate=risk_free_rate,
    )
    optimization_constraints = _build_optimization_constraints(
        config,
        metadata,
        asset_returns.columns,
    )
    asset_class_map = optimization_constraints["asset_classes"]
    expense_ratio_map = optimization_constraints["expense_ratios"]
    asset_class_bounds = optimization_constraints["asset_class_bounds"]
    ticker_bounds = optimization_constraints["ticker_bounds"]
    bond_assets = optimization_constraints["bond_assets"]
    fallback_drift_threshold = (
        config.rebalance.fallback.sell_allowed_if_absolute_drift_exceeds
        if config.rebalance.fallback is not None
        else None
    )
    backtest_result = run_walk_forward_backtest(
        asset_returns,
        rebalance_dates=rebalance_dates,
        lookback_periods=effective_lookback,
        optimization_method=method,
        max_weight=config.optimization.default_max_weight_per_etf,
        transaction_cost_rate=_transaction_cost_rate(config),
        asset_classes=asset_class_map,
        asset_class_bounds=asset_class_bounds,
        ticker_bounds=ticker_bounds,
        bond_assets=bond_assets,
        min_bond_exposure=optimization_constraints["min_bond_exposure"],
        expense_ratios=expense_ratio_map,
        risk_free_rate=risk_free_rate,
        rebalance_mode=config.rebalance.mode,
        contribution_amount=config.rebalance.contribution_amount,
        tolerance_bands=config.rebalance.tolerance_bands,
        initial_capital=config.backtest.initial_capital,
        fallback_sell_allowed=config.rebalance.fallback_sell_allowed,
        fallback_drift_threshold=fallback_drift_threshold,
        realized_constraint_policy=config.rebalance.realized_constraint_policy,
        covariance_method=config.optimization.risk_model,
        expected_return_method=config.optimization.expected_return_estimator,
    )

    aligned_benchmark = benchmark_returns.reindex(backtest_result.portfolio_returns.index).dropna()
    aligned_portfolio = backtest_result.portfolio_returns.reindex(aligned_benchmark.index)
    in_sample_returns = asset_returns.loc[asset_returns.index < backtest_result.weights.index[-1]]
    expected_returns = estimate_expected_returns(
        in_sample_returns,
        method=config.optimization.expected_return_estimator,
    )
    covariance_matrix = calculate_covariance_matrix(
        in_sample_returns,
        method=config.optimization.risk_model,
    )
    benchmark_artifacts = _build_benchmark_suite(
        returns,
        asset_returns,
        config=config,
        rebalance_dates=rebalance_dates,
        lookback_periods=effective_lookback,
        transaction_cost_rate=_transaction_cost_rate(config),
        asset_classes=asset_class_map,
        asset_class_bounds=asset_class_bounds,
        ticker_bounds=ticker_bounds,
        bond_assets=bond_assets,
        min_bond_exposure=optimization_constraints["min_bond_exposure"],
        expense_ratios=expense_ratio_map,
        risk_free_rate=risk_free_rate,
    )
    benchmark_suite = benchmark_artifacts.returns
    aligned_benchmark_suite = {
        name: series.reindex(aligned_portfolio.index).dropna()
        for name, series in benchmark_suite.items()
    }
    aligned_benchmark_suite = {
        name: series.reindex(aligned_portfolio.index)
        for name, series in aligned_benchmark_suite.items()
    }
    report_benchmarks = pd.DataFrame(
        {
            name: series.reindex(aligned_portfolio.index)
            for name, series in aligned_benchmark_suite.items()
        }
    )
    primary_benchmark = aligned_benchmark_suite.get("Selected Benchmark ETF", aligned_benchmark)

    report_path = project_root / "reports/html/latest_report.html"
    workbook_path = project_root / "reports/excel/portfolio_results.xlsx"
    figures_dir = project_root / "reports/figures"
    report_artifacts = generate_report_bundle(
        backtest_result,
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        html_output_path=report_path,
        workbook_output_path=workbook_path,
        figures_output_dir=figures_dir,
        benchmark_returns=report_benchmarks,
        primary_benchmark_returns=primary_benchmark,
        periods_per_year=252,
        risk_free_rate=risk_free_rate,
        max_weight=config.optimization.default_max_weight_per_etf,
        title=f"{config.project.name} Backtest Report",
        benchmark_suite=aligned_benchmark_suite,
        benchmark_weights=benchmark_artifacts.weights,
        metadata=metadata.reindex(backtest_result.weights.columns),
        prices=(
            validated_prices.reindex(columns=backtest_result.weights.columns)
            if validated_prices is not None
            else None
        ),
        assumptions=_report_assumptions(config, method, run_id=stage_run_id),
        limitations=_report_limitations(config),
        asset_returns=asset_returns,
        asset_classes=asset_class_map,
        asset_class_bounds=asset_class_bounds,
        ticker_bounds=ticker_bounds,
        bond_assets=bond_assets,
        min_bond_exposure=optimization_constraints["min_bond_exposure"],
        expense_ratios=expense_ratio_map,
    )
    legacy_report_path = project_root / "reports/html/backtest_report.html"
    legacy_report_path.write_text(
        report_artifacts.html_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    metrics_path = project_root / "reports/metrics/backtest_metrics.json"
    metrics_table = build_metrics_table(
        aligned_portfolio,
        weights=backtest_result.weights,
        periods_per_year=252,
        benchmark_returns=primary_benchmark,
        benchmark_suite=aligned_benchmark_suite,
        benchmark_weights=benchmark_artifacts.weights,
        risk_free_rate=risk_free_rate,
    )
    metrics_by_strategy = {
        str(row["Strategy"]): {key: value for key, value in row.items() if key != "Strategy"}
        for row in metrics_table.to_dict(orient="records")
    }
    strategy_metrics_payload = metrics_by_strategy.pop("Optimized Strategy")
    strategy_metrics = pd.Series(strategy_metrics_payload, dtype=float)
    metrics_payload: dict[str, Any] = {
        "run_id": stage_run_id,
        "optimized_strategy": strategy_metrics_payload,
        "benchmarks": metrics_by_strategy,
    }
    if tracking_provenance is not None:
        metrics_payload["provenance_status"] = tracking_provenance["provenance_status"]
    if persist_metrics:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        run_record_path = (
            project_root / config.tracking.artifact_dir / f"backtest_{stage_run_id}.json"
        )
        metrics_payload["run_record"] = relative_to_project_root(project_root, run_record_path)
        write_metrics_json(metrics_payload, metrics_path)
        run_record = build_run_record(
            stage="backtest",
            run_id=stage_run_id,
            config=config,
            project_root=project_root,
            data_version_path=project_root / "data/processed/returns.parquet",
            output_artifacts={
                "report": report_artifacts.html_path,
                "workbook": report_artifacts.workbook_path,
                "metrics": metrics_path,
                **{f"figure_{name}": path for name, path in report_artifacts.figure_paths.items()},
            },
            actual_start_date=aligned_portfolio.index.min().date().isoformat(),
            actual_end_date=aligned_portfolio.index.max().date().isoformat(),
            optimization_method=method,
            backtest_metrics=metrics_payload["optimized_strategy"],
        )
        written_run_record_path = write_run_record(
            run_record,
            artifact_dir=project_root / config.tracking.artifact_dir,
        )
        if written_run_record_path != run_record_path:
            raise RuntimeError(
                f"Unexpected run record path: {written_run_record_path} != {run_record_path}"
            )
    if float(strategy_metrics["Turnover"]) > 1.0:
        log_event(
            LOGGER,
            logging.WARNING,
            "backtest_high_turnover_detected",
            run_id=stage_run_id,
            stage="backtest",
            config_path=str(config_path),
            turnover=float(strategy_metrics["Turnover"]),
        )
    log_event(
        LOGGER,
        logging.INFO,
        "pipeline_stage_completed",
        run_id=stage_run_id,
        stage="backtest",
        config_path=str(config_path),
        optimizer_status="success",
        solver_used="SLSQP",
        portfolio_return=float(strategy_metrics["CAGR"]),
        portfolio_volatility=float(strategy_metrics["Annualized Volatility"]),
        portfolio_sharpe=float(strategy_metrics["Sharpe Ratio"]),
        max_weight=float(strategy_metrics["Largest Position"]),
        turnover=float(strategy_metrics["Turnover"]),
        output_path=[
            str(report_artifacts.html_path),
            str(report_artifacts.workbook_path),
            str(metrics_path),
            *[str(path) for path in report_artifacts.figure_paths.values()],
        ],
    )

    return report_artifacts.html_path, metrics_path


def run_report(
    config_path: str | Path,
    *,
    project_root: Path = Path("."),
    lookback_periods: int = DEFAULT_LOOKBACK_PERIODS,
    run_id: str | None = None,
) -> Path:
    """Generate the report bundle from the configured pipeline inputs."""

    stage_run_id = run_id or generate_run_id("report")
    log_event(
        LOGGER,
        logging.INFO,
        "pipeline_stage_started",
        run_id=stage_run_id,
        stage="report",
        config_path=str(config_path),
    )
    report_path, _ = run_backtest(
        config_path,
        project_root=project_root,
        lookback_periods=lookback_periods,
        run_id=stage_run_id,
        persist_metrics=False,
    )
    log_event(
        LOGGER,
        logging.INFO,
        "pipeline_stage_completed",
        run_id=stage_run_id,
        stage="report",
        config_path=str(config_path),
        output_path=str(report_path),
    )
    return report_path


def run_all(
    config_path: str | Path,
    *,
    project_root: Path = Path("."),
    lookback_periods: int = DEFAULT_LOOKBACK_PERIODS,
    run_id: str | None = None,
) -> dict[str, Path]:
    """Run the end-to-end research pipeline from ingestion through reporting."""

    pipeline_run_id = run_id or generate_run_id("run-all")
    return {
        "raw_prices": run_ingest(config_path, project_root=project_root, run_id=pipeline_run_id),
        "validated_prices": run_validate(
            config_path,
            project_root=project_root,
            run_id=pipeline_run_id,
        ),
        "returns": run_features(config_path, project_root=project_root, run_id=pipeline_run_id),
        "optimized_workbook": run_optimize(
            config_path,
            project_root=project_root,
            run_id=pipeline_run_id,
        )[0],
        "report": run_backtest(
            config_path,
            project_root=project_root,
            lookback_periods=lookback_periods,
            run_id=pipeline_run_id,
        )[0],
    }


def run_ml(
    config_path: str | Path,
    *,
    project_root: Path = Path("."),
    run_id: str | None = None,
) -> dict[str, Path]:
    """Build features, evaluate ML baselines, and persist artifacts."""

    config = _load_project_config(config_path, project_root)
    if not config.ml.enabled:
        raise ValueError("ML is disabled in the current config.")

    stage_run_id = run_id or generate_run_id("ml")
    tracking_provenance = resolve_run_provenance(config=config, project_root=project_root)
    returns = pd.read_parquet(project_root / "data/processed/returns.parquet")
    asset_returns = _asset_returns(returns, config)
    benchmark_returns = _benchmark_returns(returns, config)
    common_dates = asset_returns.index.intersection(benchmark_returns.index)
    asset_returns = asset_returns.loc[common_dates]
    benchmark_returns = benchmark_returns.loc[common_dates]
    log_event(
        LOGGER,
        logging.INFO,
        "pipeline_stage_started",
        run_id=stage_run_id,
        stage="ml",
        config_path=str(config_path),
        tickers_requested=config.universe.tickers,
        tickers_loaded=asset_returns.columns.tolist(),
        start_date=asset_returns.index.min().date().isoformat(),
        end_date=asset_returns.index.max().date().isoformat(),
    )

    dataset = build_ml_dataset(
        asset_returns,
        ml_config=config.ml,
        benchmark_returns=benchmark_returns,
    )
    train_frame, test_frame = chronological_train_test_split(
        dataset.frame,
        test_window_periods=config.ml.validation.test_window_periods,
    )
    evaluation = walk_forward_evaluate(
        dataset.frame,
        feature_columns=dataset.feature_columns,
        target_column=dataset.target_column,
        model_names=config.ml.models,
        task=config.ml.task,
        validation=config.ml.validation,
    )

    best_model_name = _select_best_ml_model(evaluation.summary, task=config.ml.task)
    trained_model = fit_model(
        train_frame,
        feature_columns=dataset.feature_columns,
        target_column=dataset.target_column,
        model_name=best_model_name,
        task=config.ml.task,
    )

    artifact_dir = project_root / config.ml.tracking.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = artifact_dir / "predictions.parquet"
    dataset_path = artifact_dir / "dataset.parquet"
    summary_path = artifact_dir / "summary.csv"
    fold_metrics_path = artifact_dir / "fold_metrics.csv"
    model_path = artifact_dir / f"model_{ML_MODEL_TRAINING_SCOPE}.pkl"
    metrics_path = artifact_dir / "metrics.json"
    governance_path = artifact_dir / "governance.json"
    model_card_path = artifact_dir / "model_card.md"

    dataset.frame.to_parquet(dataset_path)
    evaluation.predictions.to_parquet(predictions_path)
    evaluation.summary.to_csv(summary_path, index=False)
    evaluation.fold_metrics.to_csv(fold_metrics_path, index=False)
    save_model_bundle(
        trained_model,
        output_path=model_path,
        feature_columns=dataset.feature_columns,
        target_column=dataset.target_column,
        metadata={
            "run_id": stage_run_id,
            "model_name": best_model_name,
            "task": config.ml.task,
            "target": config.ml.target,
            "training_scope": ML_MODEL_TRAINING_SCOPE,
            "training_scope_description": ML_MODEL_TRAINING_SCOPE_DESCRIPTION,
            "training_observations": int(len(train_frame)),
            "training_start_date": _panel_date_min(train_frame),
            "training_end_date": _panel_date_max(train_frame),
            "holdout_observations": int(len(test_frame)),
            "holdout_start_date": _panel_date_min(test_frame),
            "holdout_end_date": _panel_date_max(test_frame),
        },
    )

    leakage_checks = evaluate_leakage_checks(dataset.frame, evaluation.fold_metrics)
    governance = evaluate_model_governance(
        config=config,
        dataset=dataset,
        evaluation_summary=evaluation.summary,
        fold_metrics=evaluation.fold_metrics,
        best_model_name=best_model_name,
        model_path=model_path,
        leakage_checks=leakage_checks,
    )
    governance["model_artifact"] = relative_to_project_root(project_root, model_path)
    governance["model_training_scope"] = ML_MODEL_TRAINING_SCOPE
    governance["model_training_scope_description"] = ML_MODEL_TRAINING_SCOPE_DESCRIPTION
    governance["training_observations"] = int(len(train_frame))
    governance["holdout_observations"] = int(len(test_frame))
    write_model_card(
        output_path=model_card_path,
        run_id=stage_run_id,
        config=config,
        governance=governance,
        summary=evaluation.summary,
    )
    write_metrics_json(governance, governance_path)

    metrics_payload = {
        "run_id": stage_run_id,
        "best_model": best_model_name,
        "task": config.ml.task,
        "target": config.ml.target,
        "horizon_periods": config.ml.horizon_periods,
        "model_artifact": relative_to_project_root(project_root, model_path),
        "model_training_scope": ML_MODEL_TRAINING_SCOPE,
        "model_training_scope_description": ML_MODEL_TRAINING_SCOPE_DESCRIPTION,
        "training_observations": int(len(train_frame)),
        "holdout_observations": int(len(test_frame)),
        "feature_columns": dataset.feature_columns,
        "summary": evaluation.summary.round(6).to_dict(orient="records"),
        "fold_count": int(evaluation.fold_metrics["fold"].nunique()),
        "governance": governance,
        "provenance_status": tracking_provenance["provenance_status"],
    }
    run_record_path = project_root / config.tracking.artifact_dir / f"ml_{stage_run_id}.json"
    metrics_payload["run_record"] = relative_to_project_root(project_root, run_record_path)
    mlflow_status = log_mlflow_run(
        config=config,
        run_id=stage_run_id,
        metrics=metrics_payload,
        params=config.model_dump(mode="json"),
        artifacts={
            "dataset": str(dataset_path),
            "predictions": str(predictions_path),
            "summary": str(summary_path),
            "fold_metrics": str(fold_metrics_path),
            "model": str(model_path),
            "governance": str(governance_path),
            "model_card": str(model_card_path),
        },
        tags={
            "approval_status": governance["approval_status"],
            "run_record": metrics_payload["run_record"],
        },
    )
    metrics_payload["mlflow"] = mlflow_status
    write_metrics_json(metrics_payload, metrics_path)
    run_record = build_run_record(
        stage="ml",
        run_id=stage_run_id,
        config=config,
        project_root=project_root,
        data_version_path=project_root / "data/processed/returns.parquet",
        output_artifacts={
            "dataset": dataset_path,
            "predictions": predictions_path,
            "summary": summary_path,
            "fold_metrics": fold_metrics_path,
            "model": model_path,
            "metrics": metrics_path,
            "governance": governance_path,
            "model_card": model_card_path,
        },
        actual_start_date=asset_returns.index.min().date().isoformat(),
        actual_end_date=asset_returns.index.max().date().isoformat(),
        optimization_method=None,
        extra={
            "benchmark": config.benchmark.primary,
            "feature_set": dataset.feature_columns,
            "target_definition": governance["target_definition"],
            "train_validation_test_windows": governance["train_validation_test_windows"],
            "leakage_checks": leakage_checks,
            "model_version": governance["model_version"],
            "approval_status": governance["approval_status"],
            "model_artifact": governance["model_artifact"],
            "model_training_scope": governance["model_training_scope"],
            "model_card": relative_to_project_root(project_root, model_card_path),
        },
    )
    written_run_record_path = write_run_record(
        run_record,
        artifact_dir=project_root / config.tracking.artifact_dir,
    )
    if written_run_record_path != run_record_path:
        raise RuntimeError(
            f"Unexpected run record path: {written_run_record_path} != {run_record_path}"
        )
    log_event(
        LOGGER,
        logging.INFO,
        "pipeline_stage_completed",
        run_id=stage_run_id,
        stage="ml",
        config_path=str(config_path),
        benchmark=config.benchmark.primary,
        output_path=[
            str(dataset_path),
            str(predictions_path),
            str(summary_path),
            str(fold_metrics_path),
            str(model_path),
            str(governance_path),
            str(model_card_path),
            str(metrics_path),
        ],
    )

    return {
        "dataset": dataset_path,
        "predictions": predictions_path,
        "summary": summary_path,
        "fold_metrics": fold_metrics_path,
        "model": model_path,
        "governance": governance_path,
        "model_card": model_card_path,
        "metrics": metrics_path,
    }


def _panel_date_min(frame: pd.DataFrame) -> str | None:
    if frame.empty:
        return None
    date_index = frame.index.get_level_values("date")
    return pd.Timestamp(date_index.min()).date().isoformat()


def _panel_date_max(frame: pd.DataFrame) -> str | None:
    if frame.empty:
        return None
    date_index = frame.index.get_level_values("date")
    return pd.Timestamp(date_index.max()).date().isoformat()


def build_price_provider(config: AppConfig) -> PriceDataProvider:
    """Construct the configured market data provider."""

    return build_named_price_provider(config.data.provider)


def build_named_price_provider(name: str) -> PriceDataProvider:
    """Construct a price provider by name.

    Centralizes provider selection so cross-check providers and the primary
    provider can be swapped via configuration without ad-hoc string handling.
    """

    provider_name = name.strip().lower()
    if provider_name == "yfinance":
        return YFinancePriceProvider()
    if provider_name == "tiingo":
        return TiingoPriceProvider()

    raise ValueError(f"Unsupported data provider: {name}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="etf-portfolio")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command_name in ("ingest", "validate", "features", "optimize", "ml"):
        subparser = subparsers.add_parser(command_name)
        subparser.add_argument("--config", required=True)

    backtest_parser = subparsers.add_parser("backtest")
    backtest_parser.add_argument("--config", required=True)
    backtest_parser.add_argument(
        "--lookback-periods",
        type=int,
        default=DEFAULT_LOOKBACK_PERIODS,
    )

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--config", required=True)
    report_parser.add_argument(
        "--lookback-periods",
        type=int,
        default=DEFAULT_LOOKBACK_PERIODS,
    )

    run_all_parser = subparsers.add_parser("run-all")
    run_all_parser.add_argument("--config", required=True)
    run_all_parser.add_argument(
        "--lookback-periods",
        type=int,
        default=DEFAULT_LOOKBACK_PERIODS,
    )

    return parser


def _load_project_config(config_path: str | Path, project_root: Path) -> AppConfig:
    return load_config(_resolve_path(project_root, config_path))


def _resolve_path(project_root: Path, candidate: str | Path) -> Path:
    path = Path(candidate)
    if path.is_absolute():
        return path
    return project_root / path


def _metadata_path(project_root: Path) -> Path:
    return project_root / "data/metadata/etf_universe.csv"


def _required_tickers(config: AppConfig) -> list[str]:
    tickers: list[str] = []
    tickers.extend(config.universe.tickers)
    tickers.append(config.benchmark.primary)
    for allocations in config.benchmark.secondary.values():
        tickers.extend(allocations.allocations.keys())
    return list(dict.fromkeys(tickers))


def _report_assumptions(
    config: AppConfig,
    optimization_method: str,
    *,
    run_id: str | None = None,
) -> dict[str, str]:
    assumptions = {
        "data_provider": f"Data provider: {config.data.provider}",
        "benchmark": f"Primary benchmark: {config.benchmark.primary}",
        "optimization_method": f"Optimization method: {optimization_method}",
        "risk_model": f"Risk model: {config.optimization.risk_model}",
        "expected_returns": (
            f"Expected return estimator: {config.optimization.expected_return_estimator}"
        ),
        "rebalance_mode": f"Rebalance mode: {config.rebalance.mode}",
        "rebalance_frequency": f"Rebalance frequency: {config.rebalance.frequency}",
        "rebalance_execution": (
            "Rebalance execution: target weights are estimated with returns strictly before "
            "each rebalance date; the return labeled with the rebalance date remains in the "
            "previous holding period; new weights and trade-cost impact start on the next "
            "return date."
        ),
        "benchmark_fairness": (
            "Optimized benchmark objectives use the same rebalance mode "
            f"({config.rebalance.mode}), contribution amount "
            f"({config.rebalance.contribution_amount:.2f}), initial capital, transaction "
            "cost/slippage rate, constraints, and rebalance schedule as the main strategy."
        ),
        "benchmark_return_series": (
            "The selected benchmark ETF and configured secondary allocation benchmarks are "
            "external/theoretical return-series baselines; they are not simulated with "
            "contribution-only drift or strategy transaction costs."
        ),
        "transaction_costs": (
            "Transaction cost assumptions: "
            f"{config.costs.transaction_cost_bps:.2f} bps fees + "
            f"{config.costs.slippage_bps:.2f} bps slippage"
        ),
        "weight_cap": (
            "Default maximum ETF weight: "
            f"{config.optimization.default_max_weight_per_etf:.2%}; "
            "ticker-specific bounds may override this default."
        ),
        "risk_free_rate": (
            f"Annualized risk-free rate ({config.risk_free.source}): {config.risk_free.value:.2%}"
        ),
    }
    if run_id is not None:
        assumptions["run_id"] = f"Run ID: {run_id}"
    return assumptions


def _report_limitations(config: AppConfig) -> list[str]:
    return [
        "This report is generated from pipeline artifacts, not manually curated notebook output.",
        (
            "Expected returns and covariance estimates are backward-looking and "
            "depend on the configured historical window."
        ),
        (
            "Transaction costs and configured slippage are modeled as simple bps assumptions; "
            "taxes, spreads beyond configured slippage, market impact, and account-specific "
            "fees are not modeled."
        ),
        (
            "FX-adjusted returns, MXN/NOK reporting layers, UCITS comparisons, "
            "and tax-aware domicile analysis are deferred follow-on work."
        ),
        (
            "Benchmark and portfolio analytics reflect the configured universe and "
            f"selected benchmark ({config.benchmark.primary}) only."
        ),
    ]


def _asset_returns(returns: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    return returns.loc[:, config.universe.tickers].dropna()


def _benchmark_returns(returns: pd.DataFrame, config: AppConfig) -> pd.Series:
    benchmark = returns.loc[:, config.benchmark.primary].dropna()
    benchmark.name = config.benchmark.primary
    return benchmark


def _apply_backtest_window(
    asset_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    config: AppConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    asset_window = asset_returns
    benchmark_window = benchmark_returns

    if config.backtest.start_date is not None:
        start = pd.Timestamp(config.backtest.start_date)
        asset_window = asset_window.loc[asset_window.index >= start]
        benchmark_window = benchmark_window.loc[benchmark_window.index >= start]

    if config.backtest.end_date is not None:
        end = pd.Timestamp(config.backtest.end_date)
        asset_window = asset_window.loc[asset_window.index <= end]
        benchmark_window = benchmark_window.loc[benchmark_window.index <= end]

    common_dates = asset_window.index.intersection(benchmark_window.index)
    asset_window = asset_window.loc[common_dates]
    benchmark_window = benchmark_window.loc[common_dates]

    if asset_window.empty or benchmark_window.empty:
        raise ValueError("No backtest return history remains after applying backtest date window.")

    return asset_window, benchmark_window


def _build_benchmark_suite(
    returns: pd.DataFrame,
    asset_returns: pd.DataFrame,
    *,
    config: AppConfig,
    rebalance_dates: pd.DatetimeIndex,
    lookback_periods: int,
    transaction_cost_rate: float,
    asset_classes: pd.Series | None = None,
    asset_class_bounds: dict[str, tuple[float | None, float | None]] | None = None,
    ticker_bounds: dict[str, tuple[float | None, float | None]] | None = None,
    bond_assets: list[str] | None = None,
    min_bond_exposure: float | None = None,
    expense_ratios: pd.Series | None = None,
    risk_free_rate: float = 0.0,
) -> BenchmarkSuiteArtifacts:
    selected_benchmark_returns = _benchmark_returns(returns, config)
    suite: dict[str, pd.Series] = {"Selected Benchmark ETF": selected_benchmark_returns}
    benchmark_weights: dict[str, pd.DataFrame] = {
        "Selected Benchmark ETF": _constant_benchmark_weights(
            selected_benchmark_returns.index,
            {config.benchmark.primary: 1.0},
        )
    }
    for benchmark_name, benchmark_config in config.benchmark.secondary.items():
        label = _secondary_benchmark_label(benchmark_name)
        allocations = benchmark_config.allocations
        suite[label] = _composite_benchmark_returns(returns, allocations)
        benchmark_weights[label] = _constant_benchmark_weights(suite[label].index, allocations)
    for objective in config.optimization.benchmark_objectives:
        benchmark_result = run_walk_forward_backtest(
            asset_returns,
            rebalance_dates=rebalance_dates,
            lookback_periods=lookback_periods,
            optimization_method=_optimization_method_for_objective(objective),
            max_weight=config.optimization.default_max_weight_per_etf,
            transaction_cost_rate=transaction_cost_rate,
            asset_classes=asset_classes,
            asset_class_bounds=asset_class_bounds,
            ticker_bounds=ticker_bounds,
            bond_assets=bond_assets,
            min_bond_exposure=min_bond_exposure,
            expense_ratios=expense_ratios,
            risk_free_rate=risk_free_rate,
            rebalance_mode=config.rebalance.mode,
            contribution_amount=config.rebalance.contribution_amount,
            tolerance_bands=config.rebalance.tolerance_bands,
            initial_capital=config.backtest.initial_capital,
            fallback_sell_allowed=config.rebalance.fallback_sell_allowed,
            fallback_drift_threshold=(
                config.rebalance.fallback.sell_allowed_if_absolute_drift_exceeds
                if config.rebalance.fallback is not None
                else None
            ),
            realized_constraint_policy=config.rebalance.realized_constraint_policy,
            covariance_method=config.optimization.risk_model,
            expected_return_method=config.optimization.expected_return_estimator,
        )
        label = _benchmark_objective_label(objective)
        suite[label] = benchmark_result.portfolio_returns
        benchmark_weights[label] = benchmark_result.weights
    previous_result = run_walk_forward_backtest(
        asset_returns,
        rebalance_dates=rebalance_dates,
        lookback_periods=lookback_periods,
        optimization_method=_select_optimization_method(config),
        max_weight=config.optimization.default_max_weight_per_etf,
        transaction_cost_rate=transaction_cost_rate,
        asset_classes=asset_classes,
        asset_class_bounds=asset_class_bounds,
        ticker_bounds=ticker_bounds,
        bond_assets=bond_assets,
        min_bond_exposure=min_bond_exposure,
        expense_ratios=expense_ratios,
        risk_free_rate=risk_free_rate,
        apply_previous_weights_lag=True,
        rebalance_mode=config.rebalance.mode,
        contribution_amount=config.rebalance.contribution_amount,
        tolerance_bands=config.rebalance.tolerance_bands,
        initial_capital=config.backtest.initial_capital,
        fallback_sell_allowed=config.rebalance.fallback_sell_allowed,
        fallback_drift_threshold=(
            config.rebalance.fallback.sell_allowed_if_absolute_drift_exceeds
            if config.rebalance.fallback is not None
            else None
        ),
        realized_constraint_policy=config.rebalance.realized_constraint_policy,
        covariance_method=config.optimization.risk_model,
        expected_return_method=config.optimization.expected_return_estimator,
    )
    suite["Previous Optimized Strategy"] = previous_result.portfolio_returns
    benchmark_weights["Previous Optimized Strategy"] = previous_result.weights
    return BenchmarkSuiteArtifacts(returns=suite, weights=benchmark_weights)


def _benchmark_objective_label(objective: OptimizationObjective) -> str:
    labels: dict[OptimizationObjective, str] = {
        "equal_weight": "Equal-Weight ETF Universe",
        "inverse_volatility": "Inverse-Volatility Portfolio",
        "min_variance": "Minimum-Variance Portfolio",
        "max_sharpe": "Max-Sharpe Portfolio",
        "risk_parity": "Risk-Parity Portfolio",
    }
    return labels[objective]


def _secondary_benchmark_label(name: str) -> str:
    labels = {
        "global_60_40": "60/40 Portfolio",
    }
    return labels.get(name, name)


def _optimization_method_for_objective(objective: OptimizationObjective) -> OptimizationMethod:
    method_map: dict[OptimizationObjective, OptimizationMethod] = {
        "equal_weight": "equal_weight",
        "inverse_volatility": "inverse_volatility",
        "min_variance": "min_volatility",
        "max_sharpe": "max_sharpe",
        "risk_parity": "risk_parity",
    }
    return method_map[objective]


def _composite_benchmark_returns(
    returns: pd.DataFrame,
    allocations: dict[str, float],
) -> pd.Series:
    available = returns.loc[:, list(allocations)]
    weights = pd.Series(allocations, dtype=float)
    return available.mul(weights, axis=1).sum(axis=1)


def _constant_benchmark_weights(
    index: pd.Index,
    allocations: dict[str, float],
) -> pd.DataFrame:
    weight_index = pd.Index([index.min(), index.max()], name="rebalance_date").unique()
    return pd.DataFrame(
        [allocations for _ in range(len(weight_index))],
        index=weight_index,
        dtype=float,
    )


def _asset_class_bounds(
    config: AppConfig,
    asset_classes: pd.Series,
) -> dict[str, tuple[float | None, float | None]] | None:
    if not config.constraints.asset_class_bounds:
        return None
    available_asset_classes = set(asset_classes.dropna().astype(str))
    configured_classes = set(config.constraints.asset_class_bounds)
    unknown_classes = sorted(configured_classes - available_asset_classes)
    if unknown_classes:
        raise ValueError(
            "Configured constraints.asset_class_bounds contains unknown asset_class "
            f"values. unknown={unknown_classes}, available={sorted(available_asset_classes)}"
        )
    return {
        asset_class: (bounds.min, bounds.max)
        for asset_class, bounds in config.constraints.asset_class_bounds.items()
    }


def _ticker_bounds(
    config: AppConfig,
    columns: pd.Index,
) -> dict[str, tuple[float | None, float | None]] | None:
    if not config.constraints.ticker_bounds:
        return None
    available_tickers = {str(ticker).upper() for ticker in columns}
    normalized_bounds = {
        ticker.upper(): bounds for ticker, bounds in config.constraints.ticker_bounds.items()
    }
    unknown_tickers = sorted(set(normalized_bounds) - available_tickers)
    if unknown_tickers:
        raise ValueError(
            "Configured constraints.ticker_bounds contains unknown tickers for the "
            f"available return columns. unknown={unknown_tickers}, "
            f"available={sorted(available_tickers)}"
        )
    return {ticker: (bounds.min, bounds.max) for ticker, bounds in normalized_bounds.items()}


def _bond_assets(metadata: pd.DataFrame, columns: pd.Index) -> list[str] | None:
    bond_mask = metadata.reindex(columns)["asset_class"].eq(FIXED_INCOME_ASSET_CLASS)
    bond_assets = bond_mask.index[bond_mask.fillna(False)].tolist()
    return bond_assets or None


def _min_bond_exposure(config: AppConfig) -> float | None:
    if FIXED_INCOME_ASSET_CLASS not in config.constraints.asset_class_bounds:
        return None
    return config.constraints.asset_class_bounds[FIXED_INCOME_ASSET_CLASS].min


def _build_optimization_constraints(
    config: AppConfig,
    metadata: pd.DataFrame,
    columns: pd.Index,
) -> dict[str, Any]:
    metadata_subset = metadata.reindex(columns)
    asset_class_map = metadata_subset["asset_class"]
    return {
        "asset_classes": asset_class_map,
        "expense_ratios": metadata_subset["expense_ratio"],
        "asset_class_bounds": _asset_class_bounds(config, asset_class_map),
        "ticker_bounds": _ticker_bounds(config, columns),
        "bond_assets": _bond_assets(metadata, columns),
        "min_bond_exposure": _min_bond_exposure(config),
    }


def _transaction_cost_rate(config: AppConfig) -> float:
    return (config.costs.transaction_cost_bps + config.costs.slippage_bps) / 10_000


def _select_optimization_method(config: AppConfig) -> OptimizationMethod:
    method_map: dict[str, OptimizationMethod] = {
        "equal_weight": "equal_weight",
        "inverse_volatility": "inverse_volatility",
        "min_variance": "min_volatility",
        "max_sharpe": "max_sharpe",
        "risk_parity": "risk_parity",
    }

    mapped = method_map.get(config.optimization.active_objective)
    if mapped is not None:
        return mapped

    raise ValueError(
        "Unsupported optimization.active_objective is configured. "
        "Supported objectives are equal_weight, inverse_volatility, "
        "min_variance, max_sharpe, and risk_parity."
    )


def _rebalance_dates(index: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    if frequency == "daily":
        return index

    period_alias = {
        "weekly": "W",
        "monthly": "M",
        "quarterly": "Q",
        "yearly": "Y",
    }
    alias = period_alias[frequency]
    grouped = pd.Series(index, index=index).groupby(index.to_period(alias)).max()
    return pd.DatetimeIndex(grouped.tolist())


def _write_validation_summary(summary: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "missing_data_fraction": summary.missing_data_fraction.round(6).to_dict(),
        "history_coverage": summary.history_coverage.round(6).to_dict(),
        "suspicious_jump_count": int(len(summary.suspicious_jumps)),
    }
    write_metrics_json(payload, output_path)


def _select_best_ml_model(summary: pd.DataFrame, *, task: str) -> str:
    if summary.empty:
        raise ValueError("ML evaluation summary is empty.")

    if task == "classification":
        ordered = summary.sort_values(
            by=["accuracy", "log_loss"],
            ascending=[False, True],
            na_position="last",
        )
    else:
        ordered = summary.sort_values(by=["rmse", "mae"], ascending=[True, True])
    return str(ordered.iloc[0]["model"])


if __name__ == "__main__":
    raise SystemExit(main())
