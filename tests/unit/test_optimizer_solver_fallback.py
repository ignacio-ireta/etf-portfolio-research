"""Tests for the SLSQP tolerance ladder used when the strict solve cannot converge."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import OptimizeResult

from etf_portfolio.optimization import optimizer as optimizer_module
from etf_portfolio.optimization.optimizer import SOLVER_FTOL_LADDER, optimize_portfolio

ASSETS = ["VTI", "BND", "IAU", "REMX"]


def _expected_returns() -> pd.Series:
    return pd.Series({"VTI": 0.10, "BND": 0.05, "IAU": 0.07, "REMX": 0.09})


def _covariance_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [0.040, 0.004, 0.006, 0.012],
            [0.004, 0.010, 0.002, 0.003],
            [0.006, 0.002, 0.025, 0.005],
            [0.012, 0.003, 0.005, 0.060],
        ],
        index=ASSETS,
        columns=ASSETS,
    )


def _install_fake_minimize(
    monkeypatch: pytest.MonkeyPatch,
    *,
    loosest_failing_ftol: float,
) -> list[float]:
    """Make every solve fail when ftol <= loosest_failing_ftol; record each ftol tried."""

    real_minimize = optimizer_module.minimize
    seen_ftols: list[float] = []

    def fake_minimize(objective: Any, **kwargs: Any) -> Any:
        ftol = float(kwargs["options"]["ftol"])
        seen_ftols.append(ftol)
        if ftol <= loosest_failing_ftol:
            start = np.asarray(kwargs["x0"], dtype=float)
            return OptimizeResult(
                x=start,
                success=False,
                status=8,
                message="Positive directional derivative for linesearch",
                fun=float(objective(start)),
            )
        return real_minimize(objective, **kwargs)

    monkeypatch.setattr(optimizer_module, "minimize", fake_minimize)
    return seen_ftols


def _events(caplog: pytest.LogCaptureFixture, name: str) -> list[logging.LogRecord]:
    return [record for record in caplog.records if getattr(record, "event", "") == name]


def test_ladder_starts_strict_and_is_sorted_from_strict_to_loose() -> None:
    assert SOLVER_FTOL_LADDER[0] == 1e-12
    assert list(SOLVER_FTOL_LADDER) == sorted(SOLVER_FTOL_LADDER)


def test_strict_solve_success_does_not_relax_or_warn(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    seen_ftols = _install_fake_minimize(monkeypatch, loosest_failing_ftol=0.0)

    with caplog.at_level(logging.WARNING):
        weights = optimize_portfolio(
            _expected_returns(),
            _covariance_matrix(),
            method="max_sharpe",
            max_weight=0.6,
        )

    assert weights.sum() == pytest.approx(1.0)
    assert set(seen_ftols) == {SOLVER_FTOL_LADDER[0]}
    assert _events(caplog, "optimizer_tolerance_relaxed") == []


def test_relaxed_tolerance_is_used_when_strict_solve_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    seen_ftols = _install_fake_minimize(monkeypatch, loosest_failing_ftol=1e-12)

    with caplog.at_level(logging.WARNING):
        weights = optimize_portfolio(
            _expected_returns(),
            _covariance_matrix(),
            method="max_sharpe",
            max_weight=0.6,
        )

    assert weights.sum() == pytest.approx(1.0)
    assert weights.max() <= 0.6 + 1e-8
    assert seen_ftols[0] == SOLVER_FTOL_LADDER[0]
    assert seen_ftols[-1] == SOLVER_FTOL_LADDER[1]
    relaxed_events = _events(caplog, "optimizer_tolerance_relaxed")
    assert len(relaxed_events) == 1
    assert relaxed_events[0].ftol_used == SOLVER_FTOL_LADDER[1]


def test_failure_is_raised_only_after_every_tolerance_was_tried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_ftols = _install_fake_minimize(monkeypatch, loosest_failing_ftol=1.0)

    with pytest.raises(ValueError, match="Optimization failed"):
        optimize_portfolio(
            _expected_returns(),
            _covariance_matrix(),
            method="max_sharpe",
            max_weight=0.6,
        )

    assert set(seen_ftols) == set(SOLVER_FTOL_LADDER)
