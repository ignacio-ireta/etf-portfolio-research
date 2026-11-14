"""Sampling uncertainty for backtest comparisons.

A Sharpe ratio measured on roughly a decade of daily returns is a noisy estimate.
This module puts a confidence interval around the *difference* between two Sharpe
ratios with a stationary block bootstrap (Politis and Romano, 1994): both return
series are resampled with the same randomly chosen blocks of consecutive days, so
day-to-day autocorrelation and the correlation between the two series are kept.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DEFAULT_BOOTSTRAP_RESAMPLES = 1000
DEFAULT_MEAN_BLOCK_LENGTH = 21
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_BOOTSTRAP_SEED = 0
MINIMUM_BOOTSTRAP_OBSERVATIONS = 3


def bootstrap_sharpe_difference(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    mean_block_length: int = DEFAULT_MEAN_BLOCK_LENGTH,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap the annualized Sharpe ratio of `returns` minus that of `benchmark_returns`.

    Returns a JSON-friendly dictionary. When fewer than three common observations
    exist, or either series has zero volatility, the numeric results are ``None``.
    """

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive.")
    if mean_block_length <= 0:
        raise ValueError("mean_block_length must be positive.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1.")

    paired = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    observation_count = len(paired)
    summary: dict[str, Any] = {
        "sharpe_difference": None,
        "ci_lower": None,
        "ci_upper": None,
        "probability_not_positive": None,
        "confidence_level": float(confidence_level),
        "n_observations": observation_count,
        "n_resamples": int(n_resamples),
        "mean_block_length": int(mean_block_length),
        "seed": int(seed),
    }
    if observation_count < MINIMUM_BOOTSTRAP_OBSERVATIONS:
        return summary

    strategy = paired.iloc[:, 0].to_numpy(dtype=float)
    benchmark = paired.iloc[:, 1].to_numpy(dtype=float)
    daily_risk_free = risk_free_rate / periods_per_year
    annualizer = float(np.sqrt(periods_per_year))

    strategy_volatility = float(strategy.std(ddof=1))
    benchmark_volatility = float(benchmark.std(ddof=1))
    if np.isclose(strategy_volatility, 0.0) or np.isclose(benchmark_volatility, 0.0):
        return summary

    point_estimate = (
        (strategy.mean() - daily_risk_free) / strategy_volatility
        - (benchmark.mean() - daily_risk_free) / benchmark_volatility
    ) * annualizer

    # Stationary bootstrap indices: each position either continues the current block
    # (previous index + 1, wrapping around) or jumps to a fresh random start.
    rng = np.random.default_rng(seed)
    positions = np.arange(observation_count)
    jumps = rng.random((n_resamples, observation_count)) < 1.0 / mean_block_length
    jumps[:, 0] = True
    fresh_starts = rng.integers(0, observation_count, size=(n_resamples, observation_count))
    last_jump = np.maximum.accumulate(np.where(jumps, positions, 0), axis=1)
    block_starts = np.take_along_axis(fresh_starts, last_jump, axis=1)
    indices = (block_starts + positions - last_jump) % observation_count

    strategy_samples = strategy[indices]
    benchmark_samples = benchmark[indices]
    with np.errstate(divide="ignore", invalid="ignore"):
        differences = (
            (strategy_samples.mean(axis=1) - daily_risk_free) / strategy_samples.std(axis=1, ddof=1)
            - (benchmark_samples.mean(axis=1) - daily_risk_free)
            / benchmark_samples.std(axis=1, ddof=1)
        ) * annualizer
    differences = differences[np.isfinite(differences)]
    if differences.size == 0:
        return summary

    tail = (1.0 - confidence_level) / 2.0
    summary.update(
        {
            "sharpe_difference": float(point_estimate),
            "ci_lower": float(np.quantile(differences, tail)),
            "ci_upper": float(np.quantile(differences, 1.0 - tail)),
            "probability_not_positive": float(np.mean(differences <= 0.0)),
        }
    )
    return summary


__all__ = [
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_CONFIDENCE_LEVEL",
    "DEFAULT_MEAN_BLOCK_LENGTH",
    "bootstrap_sharpe_difference",
]
