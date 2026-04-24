"""Risk-free rate resolution.

The risk-free rate is a single annualized decimal (e.g. 0.03 for 3%) that
flows into the Sharpe ratio, Sortino ratio, Jensen's alpha, and the max-Sharpe
optimization objective. This module provides one entry point so the value is
sourced consistently across the pipeline.

`source: constant` is the only supported source in v1; future work can add a
`treasury_bill` provider that pulls FRED data and treats the rate as
piecewise-constant per period.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from etf_portfolio.config import AppConfig, RiskFreeConfig


def get_risk_free_rate(config: AppConfig | RiskFreeConfig) -> float:
    """Return the configured annualized risk-free rate as a decimal."""

    risk_free = _resolve_risk_free_config(config)

    if risk_free.source == "constant":
        return float(risk_free.value)

    raise ValueError(
        f"Unsupported risk-free source: {risk_free.source!r}. Only 'constant' is supported in v1."
    )


def _resolve_risk_free_config(
    config: AppConfig | RiskFreeConfig,
) -> RiskFreeConfig:
    from etf_portfolio.config import AppConfig, RiskFreeConfig

    if isinstance(config, RiskFreeConfig):
        return config
    if isinstance(config, AppConfig):
        return config.risk_free
    raise TypeError(
        f"get_risk_free_rate expects an AppConfig or RiskFreeConfig, got {type(config).__name__}."
    )


__all__ = ["get_risk_free_rate"]
