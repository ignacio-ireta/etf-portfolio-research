"""Walk-forward portfolio backtesting engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from etf_portfolio.backtesting.costs import transaction_costs
from etf_portfolio.backtesting.rebalancing import (
    RebalanceMode,
    apply_rebalance_mode,
    normalize_rebalance_dates,
)
from etf_portfolio.config import RebalanceToleranceBandsConfig
from etf_portfolio.features.estimators import (
    calculate_covariance_matrix,
    estimate_expected_returns,
)
from etf_portfolio.optimization.optimizer import OptimizationMethod, optimize_portfolio


@dataclass(frozen=True)
class WalkForwardBacktestResult:
    portfolio_returns: pd.Series
    target_weights: pd.DataFrame
    applied_weights: pd.DataFrame
    rebalance_summary: pd.DataFrame
    portfolio_value: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    trades_dollars: pd.DataFrame = field(default_factory=pd.DataFrame)
    realized_constraint_violations: pd.DataFrame = field(default_factory=pd.DataFrame)
    realized_constraint_policy: Literal["report_drift", "enforce_hard"] = "report_drift"

    @property
    def weights(self) -> pd.DataFrame:
        """Backward-compatible alias for realized/applied weights."""

        return self.applied_weights


def run_walk_forward_backtest(
    asset_returns: pd.DataFrame,
    *,
    rebalance_dates: list[pd.Timestamp] | pd.DatetimeIndex | None = None,
    lookback_periods: int,
    optimization_method: OptimizationMethod,
    max_weight: float = 1.0,
    weight_sum: float = 1.0,
    target_return: float | None = None,
    target_volatility: float | None = None,
    risk_free_rate: float = 0.0,
    transaction_cost_rate: float = 0.0,
    min_weight: float = 0.0,
    asset_classes: pd.Series | None = None,
    asset_class_bounds: dict[str, tuple[float | None, float | None]] | None = None,
    ticker_bounds: dict[str, tuple[float | None, float | None]] | None = None,
    bond_assets: list[str] | None = None,
    min_bond_exposure: float | None = None,
    expense_ratios: pd.Series | None = None,
    max_expense_ratio: float | None = None,
    max_turnover: float | None = None,
    covariance_method: str = "sample",
    expected_return_method: str = "historical_mean",
    apply_previous_weights_lag: bool = False,
    rebalance_mode: RebalanceMode = "full_rebalance",
    contribution_amount: float = 0.0,
    tolerance_bands: RebalanceToleranceBandsConfig | None = None,
    initial_capital: float = 1.0,
    fallback_sell_allowed: bool = False,
    fallback_drift_threshold: float | None = None,
    realized_constraint_policy: Literal["report_drift", "enforce_hard"] = "report_drift",
) -> WalkForwardBacktestResult:
    """Run a walk-forward backtest using only trailing returns at each rebalance."""

    _validate_asset_returns(asset_returns)
    if lookback_periods <= 0:
        raise ValueError("lookback_periods must be positive.")

    rebalance_index = normalize_rebalance_dates(asset_returns.index, rebalance_dates)
    portfolio_segments: list[pd.Series] = []
    target_weights_history: list[pd.Series] = []
    applied_weights_history: list[pd.Series] = []
    trades_history: list[pd.Series] = []
    realized_constraint_violation_records: list[dict[str, Any]] = []
    rebalance_records: list[dict[str, Any]] = []
    portfolio_value_records: list[tuple[pd.Timestamp, float]] = []
    previous_weights = pd.Series(0.0, index=asset_returns.columns, dtype=float)
    optimizer_previous_weights = previous_weights.copy()
    prior_optimized_weights: pd.Series | None = None
    portfolio_value = float(initial_capital)
    bootstrap_completed = False
    if portfolio_value <= 0.0:
        raise ValueError("initial_capital must be positive.")

    for position, rebalance_date in enumerate(rebalance_index):
        training_window = asset_returns.loc[asset_returns.index < rebalance_date].tail(
            lookback_periods
        )
        if len(training_window) < lookback_periods:
            continue

        expected_returns = estimate_expected_returns(
            training_window,
            method=expected_return_method,
        )
        covariance_matrix = calculate_covariance_matrix(
            training_window,
            method=covariance_method,
        )
        optimized_weights = optimize_portfolio(
            expected_returns,
            covariance_matrix,
            method=optimization_method,
            min_weight=min_weight,
            max_weight=max_weight,
            weight_sum=weight_sum,
            target_return=target_return,
            target_volatility=target_volatility,
            risk_free_rate=risk_free_rate,
            asset_classes=asset_classes,
            asset_class_bounds=asset_class_bounds,
            ticker_bounds=ticker_bounds,
            bond_assets=bond_assets,
            min_bond_exposure=min_bond_exposure,
            expense_ratios=expense_ratios,
            max_expense_ratio=max_expense_ratio,
            initial_guess=prior_optimized_weights,
            previous_weights=optimizer_previous_weights if max_turnover is not None else None,
            max_turnover=max_turnover,
        )
        target_weights = optimized_weights
        if apply_previous_weights_lag:
            target_weights = (
                optimizer_previous_weights.copy()
                if prior_optimized_weights is None
                else prior_optimized_weights.copy()
            )
            if target_weights.sum() <= 0.0:
                target_weights = pd.Series(
                    1.0 / len(asset_returns.columns),
                    index=asset_returns.columns,
                    dtype=float,
                )

        should_bootstrap = (
            rebalance_mode == "contribution_only"
            and not bootstrap_completed
            and float(previous_weights.sum()) <= 1e-12
            and portfolio_value > 0.0
        )
        pretrade_constraint_violations: list[dict[str, Any]] = []
        force_sell_rebalance = False
        if rebalance_mode == "contribution_only" and realized_constraint_policy == "enforce_hard":
            pretrade_constraint_violations = _collect_realized_constraint_violations(
                rebalance_date=rebalance_date,
                applied_weights=previous_weights,
                asset_classes=asset_classes,
                asset_class_bounds=asset_class_bounds,
                ticker_bounds=ticker_bounds,
                bond_assets=bond_assets,
                min_bond_exposure=min_bond_exposure,
            )
            force_sell_rebalance = bool(pretrade_constraint_violations)
        if should_bootstrap:
            decision = apply_rebalance_mode(
                mode="full_rebalance",
                previous_weights=previous_weights,
                target_weights=target_weights,
                portfolio_value=portfolio_value,
                contribution_amount=contribution_amount,
            )
            bootstrap_completed = True
        else:
            decision = apply_rebalance_mode(
                mode=rebalance_mode,
                previous_weights=previous_weights,
                target_weights=target_weights,
                portfolio_value=portfolio_value,
                contribution_amount=contribution_amount,
                tolerance_bands=tolerance_bands,
                asset_classes=asset_classes,
                fallback_sell_allowed=fallback_sell_allowed,
                fallback_drift_threshold=fallback_drift_threshold,
                force_sell_rebalance=force_sell_rebalance,
            )
        applied_weights = _normalize_applied_weights(
            decision.applied_weights,
            target_weights.index,
        )
        portfolio_value = decision.portfolio_value_after

        next_rebalance_date = (
            rebalance_index[position + 1] if position + 1 < len(rebalance_index) else None
        )
        realized_window = _realized_window(
            asset_returns,
            start_date=rebalance_date,
            end_date=next_rebalance_date,
        )
        if realized_window.empty:
            continue

        segment_returns = realized_window.mul(applied_weights, axis=1).sum(axis=1)
        transaction_cost_dollars = transaction_costs(
            decision.trades_dollars,
            cost_rate=transaction_cost_rate,
        )
        return_impact = 0.0
        if portfolio_value > 0.0:
            return_impact = transaction_cost_dollars / portfolio_value
        segment_returns.iloc[0] -= return_impact

        portfolio_segments.append(segment_returns)
        target_weights_history.append(target_weights.rename(rebalance_date))
        applied_weights_history.append(applied_weights.rename(rebalance_date))
        trades_history.append(decision.trades_dollars.rename(rebalance_date))
        portfolio_value_records.append((rebalance_date, portfolio_value))
        turnover = float((applied_weights - previous_weights).abs().sum())
        realized_violations: list[dict[str, Any]] = []
        if rebalance_mode == "contribution_only":
            realized_violations = _collect_realized_constraint_violations(
                rebalance_date=rebalance_date,
                applied_weights=applied_weights,
                asset_classes=asset_classes,
                asset_class_bounds=asset_class_bounds,
                ticker_bounds=ticker_bounds,
                bond_assets=bond_assets,
                min_bond_exposure=min_bond_exposure,
            )
        realized_constraint_violation_records.extend(realized_violations)
        rebalance_records.append(
            {
                "rebalance_date": rebalance_date,
                "train_start": training_window.index.min(),
                "train_end": training_window.index.max(),
                "observation_count": len(training_window),
                "optimization_method": optimization_method,
                "rebalance_mode": rebalance_mode,
                "rebalanced": decision.rebalanced,
                "contribution_amount": float(contribution_amount),
                "realized_constraint_policy": realized_constraint_policy,
                "forced_constraint_sell_rebalance": force_sell_rebalance,
                "pretrade_constraint_violation_count": len(pretrade_constraint_violations),
                "applied_weight_count": int((applied_weights.abs() > 1e-12).sum()),
                "largest_position": float(applied_weights.max()),
                "turnover": turnover,
                "realized_constraint_violation_count": len(realized_violations),
                "transaction_cost": return_impact,
                "transaction_cost_dollars": transaction_cost_dollars,
                "transaction_cost_rate": float(transaction_cost_rate),
                "portfolio_value": portfolio_value,
            }
        )
        previous_weights = _drift_weights(applied_weights, segment_returns, realized_window)
        portfolio_value *= float((1.0 + segment_returns).prod())
        prior_optimized_weights = optimized_weights
        optimizer_previous_weights = optimized_weights

    if not portfolio_segments:
        raise ValueError(
            "Backtest produced no rebalance periods. Check lookback_periods and rebalance dates."
        )

    portfolio_returns = pd.concat(portfolio_segments).sort_index()
    target_weights_frame = pd.DataFrame(target_weights_history)
    target_weights_frame.index.name = "rebalance_date"
    applied_weights_frame = pd.DataFrame(applied_weights_history)
    applied_weights_frame.index.name = "rebalance_date"
    trades_frame = pd.DataFrame(trades_history)
    trades_frame.index.name = "rebalance_date"
    rebalance_summary = pd.DataFrame(rebalance_records).set_index("rebalance_date")
    portfolio_value_series = pd.Series(
        {date: value for date, value in portfolio_value_records},
        name="portfolio_value",
        dtype=float,
    )
    portfolio_value_series.index.name = "rebalance_date"

    realized_constraint_violations = pd.DataFrame(realized_constraint_violation_records)
    if not realized_constraint_violations.empty:
        realized_constraint_violations = realized_constraint_violations.sort_values(
            ["rebalance_date", "constraint_type", "identifier"]
        ).reset_index(drop=True)

    return WalkForwardBacktestResult(
        portfolio_returns=portfolio_returns,
        target_weights=target_weights_frame,
        applied_weights=applied_weights_frame,
        rebalance_summary=rebalance_summary,
        portfolio_value=portfolio_value_series,
        trades_dollars=trades_frame,
        realized_constraint_violations=realized_constraint_violations,
        realized_constraint_policy=realized_constraint_policy,
    )


def _drift_weights(
    applied_weights: pd.Series,
    segment_returns: pd.Series,
    realized_window: pd.DataFrame,
) -> pd.Series:
    """Project applied weights forward through the realized return window."""

    if realized_window.empty:
        return applied_weights.copy()
    asset_growth = (1.0 + realized_window).prod(axis=0)
    drifted_value = applied_weights * asset_growth
    total = float(drifted_value.sum())
    if total <= 0.0:
        return applied_weights.copy()
    return drifted_value / total


def _realized_window(
    asset_returns: pd.DataFrame,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp | None,
) -> pd.DataFrame:
    """Return returns strictly after start_date up to (and including) end_date.

    If rebalancing at T, weights are applied to returns starting at T+1.
    """

    if end_date is None:
        return asset_returns.loc[asset_returns.index > start_date]
    return asset_returns.loc[(asset_returns.index > start_date) & (asset_returns.index <= end_date)]


def _validate_asset_returns(asset_returns: pd.DataFrame) -> None:
    if asset_returns.empty:
        raise ValueError("asset_returns must not be empty.")
    if not isinstance(asset_returns.index, pd.DatetimeIndex):
        raise ValueError("asset_returns index must be a DatetimeIndex.")


def _normalize_applied_weights(applied_weights: pd.Series, assets: pd.Index) -> pd.Series:
    normalized = applied_weights.reindex(assets).fillna(0.0).astype(float)
    if _has_explicit_cash_asset(normalized.index):
        return normalized
    total_weight = float(normalized.sum())
    if total_weight <= 0.0:
        raise ValueError("applied_weights must sum to a positive value.")
    return normalized / total_weight


def _has_explicit_cash_asset(assets: pd.Index) -> bool:
    return any(str(asset).upper() == "CASH" for asset in assets)


def _collect_realized_constraint_violations(
    *,
    rebalance_date: pd.Timestamp,
    applied_weights: pd.Series,
    asset_classes: pd.Series | None,
    asset_class_bounds: dict[str, tuple[float | None, float | None]] | None,
    ticker_bounds: dict[str, tuple[float | None, float | None]] | None,
    bond_assets: list[str] | None,
    min_bond_exposure: float | None,
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    realized = applied_weights.astype(float)

    if ticker_bounds:
        normalized_bounds = {
            str(ticker).upper(): bounds for ticker, bounds in ticker_bounds.items()
        }
        for ticker, weight in realized.items():
            lower_bound, upper_bound = normalized_bounds.get(
                str(ticker).upper(),
                (None, None),
            )
            violations.extend(
                _build_bound_violation_rows(
                    rebalance_date=rebalance_date,
                    constraint_type="ticker",
                    identifier=str(ticker),
                    actual=float(weight),
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                )
            )

    if asset_classes is not None and asset_class_bounds:
        aligned_classes = asset_classes.reindex(realized.index)
        class_weights = realized.groupby(aligned_classes).sum()
        for asset_class, bounds in asset_class_bounds.items():
            lower_bound, upper_bound = bounds
            actual = float(class_weights.get(asset_class, 0.0))
            violations.extend(
                _build_bound_violation_rows(
                    rebalance_date=rebalance_date,
                    constraint_type="asset_class",
                    identifier=str(asset_class),
                    actual=actual,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                )
            )

    if min_bond_exposure is not None and bond_assets is not None:
        bond_set = {str(asset).upper() for asset in bond_assets}
        realized_bond_exposure = float(
            realized[[str(asset).upper() in bond_set for asset in realized.index]].sum()
        )
        if realized_bond_exposure + 1e-8 < float(min_bond_exposure):
            violations.append(
                {
                    "rebalance_date": rebalance_date,
                    "constraint_type": "bond_floor",
                    "identifier": "bond_assets",
                    "direction": "below_min",
                    "actual": realized_bond_exposure,
                    "bound": float(min_bond_exposure),
                    "breach": float(min_bond_exposure) - realized_bond_exposure,
                }
            )

    return violations


def _build_bound_violation_rows(
    *,
    rebalance_date: pd.Timestamp,
    constraint_type: str,
    identifier: str,
    actual: float,
    lower_bound: float | None,
    upper_bound: float | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if lower_bound is not None and actual + 1e-8 < float(lower_bound):
        rows.append(
            {
                "rebalance_date": rebalance_date,
                "constraint_type": constraint_type,
                "identifier": identifier,
                "direction": "below_min",
                "actual": actual,
                "bound": float(lower_bound),
                "breach": float(lower_bound) - actual,
            }
        )
    if upper_bound is not None and actual - 1e-8 > float(upper_bound):
        rows.append(
            {
                "rebalance_date": rebalance_date,
                "constraint_type": constraint_type,
                "identifier": identifier,
                "direction": "above_max",
                "actual": actual,
                "bound": float(upper_bound),
                "breach": actual - float(upper_bound),
            }
        )
    return rows
