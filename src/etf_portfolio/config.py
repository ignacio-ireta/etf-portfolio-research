"""Typed configuration loading for portfolio research runs."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

OptimizationObjective = Literal[
    "equal_weight",
    "inverse_volatility",
    "min_variance",
    "max_sharpe",
    "risk_parity",
]
RiskModel = Literal["sample", "ledoit_wolf"]
ExpectedReturnEstimator = Literal["historical_mean"]
MLTask = Literal["regression", "classification"]
MLTarget = Literal[
    "forward_return",
    "forward_volatility",
    "forward_drawdown",
    "beat_benchmark",
]
MLModelType = Literal["historical_mean", "ridge", "random_forest"]
REPLACE_ON_OVERLAY_PATHS = {
    ("constraints", "asset_class_bounds"),
    ("constraints", "ticker_bounds"),
}


class ProjectConfig(BaseModel):
    name: str
    base_currency: str

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("project.name must not be empty.")
        return value

    @field_validator("base_currency")
    @classmethod
    def _validate_currency(cls, value: str) -> str:
        currency = value.strip().upper()
        if len(currency) != 3:
            raise ValueError("project.base_currency must be a 3-letter currency code.")
        return currency


class UniverseConfig(BaseModel):
    tickers: list[str]

    @field_validator("tickers")
    @classmethod
    def _validate_tickers(cls, value: list[str]) -> list[str]:
        tickers = [ticker.strip().upper() for ticker in value if ticker.strip()]
        if not tickers:
            raise ValueError("universe.tickers must contain at least one ticker.")
        if len(set(tickers)) != len(tickers):
            raise ValueError("universe.tickers must be unique.")
        return tickers


class BenchmarkMixConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allocations: dict[str, float]

    @classmethod
    def from_mapping(cls, allocations: dict[str, float]) -> BenchmarkMixConfig:
        return cls(allocations=allocations)

    @field_validator("allocations")
    @classmethod
    def _validate_allocations(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("Benchmark mix must contain at least one ticker.")

        normalized = {}
        for ticker, weight in value.items():
            normalized_ticker = ticker.strip().upper()
            if not normalized_ticker:
                raise ValueError("Benchmark mix tickers must not be empty.")
            if weight < 0.0:
                raise ValueError("Benchmark mix weights must be non-negative.")
            normalized[normalized_ticker] = weight

        total_weight = sum(normalized.values())
        if abs(total_weight - 1.0) > 1e-8:
            raise ValueError("Benchmark mix weights must sum to 1.0.")

        return normalized


class BenchmarkConfig(BaseModel):
    primary: str
    secondary: dict[str, BenchmarkMixConfig] = Field(default_factory=dict)

    @field_validator("primary")
    @classmethod
    def _validate_primary(cls, value: str) -> str:
        ticker = value.strip().upper()
        if not ticker:
            raise ValueError("benchmark.primary must not be empty.")
        return ticker

    @field_validator("secondary", mode="before")
    @classmethod
    def _coerce_secondary(
        cls,
        value: dict[str, dict[str, float]] | dict[str, BenchmarkMixConfig] | None,
    ) -> dict[str, BenchmarkMixConfig]:
        if value is None:
            return {}

        coerced: dict[str, BenchmarkMixConfig] = {}
        for name, allocations in value.items():
            benchmark_name = name.strip()
            if not benchmark_name:
                raise ValueError("benchmark.secondary keys must not be empty.")
            if isinstance(allocations, BenchmarkMixConfig):
                coerced[benchmark_name] = allocations
            else:
                coerced[benchmark_name] = BenchmarkMixConfig.from_mapping(allocations)

        return coerced


class CrossCheckConfig(BaseModel):
    enabled: bool = False
    provider: str | None = None
    max_relative_divergence: float = Field(default=0.005, ge=0.0)
    min_overlap_observations: int = Field(default=20, ge=1)

    @field_validator("provider")
    @classmethod
    def _normalize_provider(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("cross_check.provider must not be blank when set.")
        return cleaned

    @model_validator(mode="after")
    def _validate_consistency(self) -> CrossCheckConfig:
        if self.enabled and self.provider is None:
            raise ValueError("cross_check.provider must be set when cross_check.enabled is true.")
        return self


class DataConfig(BaseModel):
    provider: str
    start_date: date
    end_date: date | None = None
    price_field: str
    cross_check: CrossCheckConfig = Field(default_factory=CrossCheckConfig)

    @field_validator("provider", "price_field")
    @classmethod
    def _require_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Value must not be empty.")
        return value


class InvestorProfileConfig(BaseModel):
    horizon_years: int = Field(gt=0)
    objective: str
    tax_preference: str

    @field_validator("objective", "tax_preference")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Investor profile values must not be empty.")
        return value


class OptimizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    long_only: bool = True
    default_max_weight_per_etf: float = Field(gt=0.0, le=1.0)
    risk_model: RiskModel = "sample"
    expected_return_estimator: ExpectedReturnEstimator = "historical_mean"
    active_objective: OptimizationObjective
    benchmark_objectives: list[OptimizationObjective] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_max_weight(cls, value: Any) -> Any:
        if isinstance(value, dict):
            unsupported_targets = sorted({"target_return", "target_volatility"}.intersection(value))
            if unsupported_targets:
                raise ValueError(
                    "optimization.target_return and optimization.target_volatility are "
                    "not supported in run config files. Use optimizer/frontier APIs for "
                    "targeted efficient-frontier experiments."
                )
        if not isinstance(value, dict) or "max_weight_per_etf" not in value:
            return value
        if "default_max_weight_per_etf" in value:
            raise ValueError(
                "Use only optimization.default_max_weight_per_etf; "
                "optimization.max_weight_per_etf is a legacy alias.",
            )
        migrated = dict(value)
        migrated["default_max_weight_per_etf"] = migrated.pop("max_weight_per_etf")
        return migrated

    @field_validator("benchmark_objectives")
    @classmethod
    def _validate_objectives(
        cls,
        value: list[OptimizationObjective],
    ) -> list[OptimizationObjective]:
        if len(set(value)) != len(value):
            raise ValueError("optimization.benchmark_objectives must be unique.")
        return value

    @model_validator(mode="after")
    def _validate_active_objective(self) -> OptimizationConfig:
        if self.active_objective in self.benchmark_objectives:
            raise ValueError(
                "optimization.active_objective must not also appear in "
                "optimization.benchmark_objectives.",
            )
        return self

    @property
    def max_weight_per_etf(self) -> float:
        return self.default_max_weight_per_etf


class AllocationBoundConfig(BaseModel):
    min: float = Field(ge=0.0, le=1.0)
    max: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_range(self) -> AllocationBoundConfig:
        if self.min > self.max:
            raise ValueError("Allocation bounds require min <= max.")
        return self


class ConstraintsConfig(BaseModel):
    asset_class_bounds: dict[str, AllocationBoundConfig] = Field(default_factory=dict)
    ticker_bounds: dict[str, AllocationBoundConfig] = Field(default_factory=dict)

    @field_validator("asset_class_bounds", "ticker_bounds", mode="before")
    @classmethod
    def _normalize_bounds(
        cls,
        value: dict[str, dict[str, float]] | dict[str, AllocationBoundConfig] | None,
    ) -> dict[str, AllocationBoundConfig]:
        if value is None:
            return {}

        normalized: dict[str, AllocationBoundConfig] = {}
        for key, bounds in value.items():
            normalized_key = key.strip()
            if not normalized_key:
                raise ValueError("Constraint names must not be empty.")
            normalized[normalized_key] = (
                bounds
                if isinstance(bounds, AllocationBoundConfig)
                else AllocationBoundConfig(**bounds)
            )
        return normalized


class RebalanceFallbackConfig(BaseModel):
    sell_allowed_if_absolute_drift_exceeds: float = Field(gt=0.0, le=1.0)


class RebalanceToleranceBandsConfig(BaseModel):
    """Per-ticker and per-asset-class absolute drift bands.

    The portfolio rebalances only when realized drift (in absolute weight units)
    exceeds the configured band. Both fields are required; set to 0.0 to make
    the band trigger on any drift.
    """

    model_config = ConfigDict(extra="forbid")

    per_ticker_abs_drift: float = Field(default=0.05, ge=0.0, le=1.0)
    per_asset_class_abs_drift: float = Field(default=0.10, ge=0.0, le=1.0)


class RebalanceConfig(BaseModel):
    mode: Literal["contribution_only", "full_rebalance", "tolerance_band"]
    frequency: Literal["daily", "weekly", "monthly", "quarterly", "yearly"]
    realized_constraint_policy: Literal["report_drift", "enforce_hard"] = "report_drift"
    fallback_sell_allowed: bool = False
    fallback: RebalanceFallbackConfig | None = None
    contribution_amount: float = Field(default=0.0, ge=0.0)
    tolerance_bands: RebalanceToleranceBandsConfig = Field(
        default_factory=RebalanceToleranceBandsConfig
    )

    @model_validator(mode="after")
    def _validate_fallback(self) -> RebalanceConfig:
        if self.fallback_sell_allowed and self.fallback is None:
            raise ValueError(
                "rebalance.fallback is required when rebalance.fallback_sell_allowed is true.",
            )
        return self

    @model_validator(mode="after")
    def _validate_contribution_consistency(self) -> RebalanceConfig:
        if self.mode == "contribution_only" and self.contribution_amount <= 0.0:
            raise ValueError(
                "rebalance.contribution_amount must be > 0 when "
                "rebalance.mode='contribution_only'.",
            )
        return self


class BacktestConfig(BaseModel):
    """Backtest window and capital base.

    `start_date` and `end_date` carve a window out of the available return
    history; leaving them unset means "use the full available history". The
    `initial_capital` figure is denominated in the project's base currency and
    is used by contribution-only rebalancing to convert weights into trades.
    """

    model_config = ConfigDict(extra="forbid")

    start_date: date | None = None
    end_date: date | None = None
    initial_capital: float = Field(default=100_000.0, gt=0.0)

    @model_validator(mode="after")
    def _validate_date_range(self) -> BacktestConfig:
        if (
            self.start_date is not None
            and self.end_date is not None
            and self.end_date <= self.start_date
        ):
            raise ValueError("backtest.end_date must be strictly after backtest.start_date.")
        return self


class CostsConfig(BaseModel):
    transaction_cost_bps: float = Field(ge=0.0)
    slippage_bps: float = Field(ge=0.0)


class RiskFreeConfig(BaseModel):
    """Annualized risk-free rate used by Sharpe/Sortino/alpha and max-Sharpe."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["constant"] = "constant"
    value: float = Field(ge=0.0, le=1.0)


class RunTrackingConfig(BaseModel):
    artifact_dir: str = "reports/runs"

    @field_validator("artifact_dir")
    @classmethod
    def _validate_artifact_dir(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("tracking.artifact_dir must not be empty.")
        return value


class MLFeatureConfig(BaseModel):
    lag_periods: list[int] = Field(default_factory=lambda: [1, 5, 21])
    momentum_periods: list[int] = Field(default_factory=lambda: [21, 63, 126, 252])
    volatility_windows: list[int] = Field(default_factory=lambda: [21, 63])
    drawdown_windows: list[int] = Field(default_factory=lambda: [63])
    correlation_windows: list[int] = Field(default_factory=lambda: [63])
    moving_average_windows: list[int] = Field(default_factory=lambda: [21, 63])

    @field_validator(
        "lag_periods",
        "momentum_periods",
        "volatility_windows",
        "drawdown_windows",
        "correlation_windows",
        "moving_average_windows",
    )
    @classmethod
    def _validate_windows(cls, value: list[int]) -> list[int]:
        normalized = sorted({int(window) for window in value if int(window) > 0})
        if not normalized:
            raise ValueError("ML feature windows must contain at least one positive integer.")
        return normalized


class MLValidationConfig(BaseModel):
    train_window_periods: int = Field(gt=0)
    test_window_periods: int = Field(gt=0)
    step_periods: int = Field(gt=0)
    min_train_periods: int = Field(gt=0)
    embargo_periods: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_ranges(self) -> MLValidationConfig:
        if self.train_window_periods < self.min_train_periods:
            raise ValueError("ml.validation.train_window_periods must be >= min_train_periods.")
        return self


class MLTrackingConfig(BaseModel):
    enable_mlflow: bool = True
    experiment_name: str = "etf_portfolio_research"
    artifact_dir: str = "reports/ml"
    dataset_version: str = "data/processed/returns.parquet"
    feature_version: str = "v1"

    @field_validator("experiment_name", "artifact_dir", "dataset_version", "feature_version")
    @classmethod
    def _validate_non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("ML tracking values must not be empty.")
        return value


class MLGovernanceConfig(BaseModel):
    minimum_fold_win_rate: float = Field(default=0.6, ge=0.0, le=1.0)
    minimum_folds_for_stability: int = Field(default=3, ge=1)
    require_baseline_outperformance: bool = True
    require_leakage_checks: bool = True


class MLConfig(BaseModel):
    enabled: bool = True
    task: MLTask = "regression"
    target: MLTarget = "forward_return"
    horizon_periods: int = Field(default=21, gt=0)
    models: list[MLModelType] = Field(
        default_factory=lambda: ["historical_mean", "ridge", "random_forest"]
    )
    features: MLFeatureConfig = Field(default_factory=MLFeatureConfig)
    validation: MLValidationConfig = Field(
        default_factory=lambda: MLValidationConfig(
            train_window_periods=252,
            test_window_periods=21,
            step_periods=21,
            min_train_periods=252,
        )
    )
    tracking: MLTrackingConfig = Field(default_factory=MLTrackingConfig)
    governance: MLGovernanceConfig = Field(default_factory=MLGovernanceConfig)

    @field_validator("models")
    @classmethod
    def _validate_models(cls, value: list[MLModelType]) -> list[MLModelType]:
        if not value:
            raise ValueError("ml.models must not be empty.")
        if len(set(value)) != len(value):
            raise ValueError("ml.models must be unique.")
        return value

    @model_validator(mode="after")
    def _validate_target_task_compatibility(self) -> MLConfig:
        if self.target == "beat_benchmark" and self.task != "classification":
            raise ValueError("ml.target='beat_benchmark' requires ml.task='classification'.")
        if self.target != "beat_benchmark" and self.task != "regression":
            raise ValueError("Regression targets require ml.task='regression'.")
        return self


class AppConfig(BaseModel):
    """Top-level application configuration."""

    model_config = ConfigDict(extra="forbid")

    project: ProjectConfig
    universe: UniverseConfig
    benchmark: BenchmarkConfig
    data: DataConfig
    investor_profile: InvestorProfileConfig
    optimization: OptimizationConfig
    constraints: ConstraintsConfig
    rebalance: RebalanceConfig
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    costs: CostsConfig
    risk_free: RiskFreeConfig = Field(
        default_factory=lambda: RiskFreeConfig(source="constant", value=0.0)
    )
    tracking: RunTrackingConfig = Field(default_factory=RunTrackingConfig)
    ml: MLConfig = Field(default_factory=MLConfig)

    @model_validator(mode="after")
    def _validate_feasibility(self) -> AppConfig:
        default_max_weight = self.optimization.default_max_weight_per_etf
        ticker_bounds = {
            ticker.upper(): bounds for ticker, bounds in self.constraints.ticker_bounds.items()
        }
        universe_tickers = {ticker.upper() for ticker in self.universe.tickers}
        unknown_tickers = sorted(set(ticker_bounds) - universe_tickers)
        if unknown_tickers:
            raise ValueError(
                "constraints.ticker_bounds contains tickers not present in universe.tickers: "
                f"{unknown_tickers}."
            )
        max_capacity = sum(
            ticker_bounds.get(ticker.upper()).max
            if ticker.upper() in ticker_bounds
            else default_max_weight
            for ticker in self.universe.tickers
        )
        if max_capacity + 1e-8 < 1.0:
            raise ValueError(
                "optimization.default_max_weight_per_etf and constraints.ticker_bounds "
                "are infeasible for the configured universe size.",
            )
        return self


def load_config(config_path: str | Path) -> AppConfig:
    """Load and validate a single YAML config file."""

    path = Path(config_path)
    return AppConfig.model_validate(_load_yaml_file(path))


def load_config_files(*config_paths: str | Path) -> AppConfig:
    """Load and deep-merge one or more YAML config files."""

    if not config_paths:
        raise ValueError("At least one config path is required.")

    merged: dict[str, Any] = {}
    for config_path in config_paths:
        path = Path(config_path)
        merged = _deep_merge(merged, _load_yaml_file(path))

    return AppConfig.model_validate(merged)


def config_to_dict(config: AppConfig) -> dict[str, Any]:
    """Convert a validated config back to a plain dictionary."""

    return config.model_dump(mode="json")


def _load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file_handle:
        payload = yaml.safe_load(file_handle) or {}

    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a YAML mapping at the top level: {path}")

    return payload


def _deep_merge(
    base: dict[str, Any],
    overlay: dict[str, Any],
    path: tuple[str, ...] = (),
) -> dict[str, Any]:
    merged = dict(base)

    for key, value in overlay.items():
        current_path = (*path, key)
        if current_path in REPLACE_ON_OVERLAY_PATHS:
            merged[key] = value
        elif key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value, current_path)
        else:
            merged[key] = value

    return merged


__all__ = [
    "AllocationBoundConfig",
    "AppConfig",
    "BacktestConfig",
    "BenchmarkConfig",
    "BenchmarkMixConfig",
    "ConstraintsConfig",
    "CostsConfig",
    "DataConfig",
    "ExpectedReturnEstimator",
    "InvestorProfileConfig",
    "MLConfig",
    "MLFeatureConfig",
    "MLGovernanceConfig",
    "MLModelType",
    "MLTarget",
    "MLTask",
    "MLTrackingConfig",
    "MLValidationConfig",
    "OptimizationConfig",
    "OptimizationObjective",
    "ProjectConfig",
    "RebalanceConfig",
    "RebalanceFallbackConfig",
    "RebalanceToleranceBandsConfig",
    "RiskFreeConfig",
    "RiskModel",
    "RunTrackingConfig",
    "UniverseConfig",
    "config_to_dict",
    "load_config",
    "load_config_files",
]
