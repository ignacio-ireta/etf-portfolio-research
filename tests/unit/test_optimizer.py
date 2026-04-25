from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from etf_portfolio.optimization import frontier as frontier_module
from etf_portfolio.optimization.frontier import build_efficient_frontier
from etf_portfolio.optimization.optimizer import optimize_portfolio, summarize_constraints


def make_expected_returns() -> pd.Series:
    return pd.Series(
        {"VTI": 0.10, "BND": 0.05, "IAU": 0.07, "REMX": 0.09},
        name="expected_return",
    )


def make_covariance_matrix() -> pd.DataFrame:
    assets = ["VTI", "BND", "IAU", "REMX"]
    return pd.DataFrame(
        [
            [0.040, 0.004, 0.006, 0.012],
            [0.004, 0.010, 0.002, 0.003],
            [0.006, 0.002, 0.025, 0.005],
            [0.012, 0.003, 0.005, 0.060],
        ],
        index=assets,
        columns=assets,
    )


def make_asset_classes() -> pd.Series:
    return pd.Series(
        {
            "VTI": "equity",
            "BND": "fixed_income",
            "IAU": "gold",
            "REMX": "thematic",
        }
    )


def make_expense_ratios() -> pd.Series:
    return pd.Series({"VTI": 0.0003, "BND": 0.0004, "IAU": 0.0025, "REMX": 0.0058})


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("equal_weight", {}),
        ("inverse_volatility", {}),
        ("min_variance", {}),
        ("max_sharpe", {"risk_free_rate": 0.02}),
        ("target_return", {"target_return": 0.075}),
        ("target_volatility", {"target_volatility": 0.13}),
        ("risk_parity", {}),
    ],
)
def test_optimizer_weights_sum_to_one(
    method: str,
    kwargs: dict[str, float],
) -> None:
    weights = optimize_portfolio(
        make_expected_returns(),
        make_covariance_matrix(),
        method=method,
        max_weight=0.65,
        **kwargs,
    )

    assert weights.sum() == pytest.approx(1.0, abs=1e-8)


def test_optimizer_respects_realistic_constraints() -> None:
    previous_weights = pd.Series({"VTI": 0.20, "BND": 0.55, "IAU": 0.20, "REMX": 0.05})
    weights = optimize_portfolio(
        make_expected_returns(),
        make_covariance_matrix(),
        method="max_sharpe",
        min_weight=0.05,
        max_weight=0.45,
        risk_free_rate=0.02,
        asset_classes=make_asset_classes(),
        asset_class_bounds={
            "equity": (0.20, 0.45),
            "fixed_income": (0.25, 0.55),
            "gold": (0.05, 0.20),
            "thematic": (0.0, 0.10),
        },
        bond_assets=["BND"],
        min_bond_exposure=0.25,
        expense_ratios=make_expense_ratios(),
        max_expense_ratio=0.0020,
        previous_weights=previous_weights,
        max_turnover=0.50,
    )
    summary = summarize_constraints(
        weights,
        asset_classes=make_asset_classes(),
        expense_ratios=make_expense_ratios(),
        previous_weights=previous_weights,
    )

    assert (weights >= 0.05 - 1e-8).all()
    assert (weights <= 0.45 + 1e-8).all()
    assert summary["asset_class::equity"] <= 0.45 + 1e-8
    assert summary["asset_class::fixed_income"] >= 0.25 - 1e-8
    assert summary["asset_class::thematic"] <= 0.10 + 1e-8
    assert summary["weighted_expense_ratio"] <= 0.0020 + 1e-8
    assert summary["turnover"] <= 0.50 + 1e-8


def test_optimizer_enforces_all_metadata_asset_class_bounds() -> None:
    expected_returns = pd.Series(
        {"VTI": 0.12, "BND": 0.02, "IAU": 0.03, "VNQ": 0.04},
        name="expected_return",
    )
    covariance_matrix = pd.DataFrame(
        [
            [0.030, 0.003, 0.004, 0.005],
            [0.003, 0.012, 0.001, 0.002],
            [0.004, 0.001, 0.025, 0.003],
            [0.005, 0.002, 0.003, 0.028],
        ],
        index=expected_returns.index,
        columns=expected_returns.index,
    )
    asset_classes = pd.Series(
        {
            "VTI": "equity",
            "BND": "fixed_income",
            "IAU": "commodity",
            "VNQ": "real_estate",
        }
    )
    exact_class_bounds = {
        "equity": (0.45, 0.55),
        "fixed_income": (0.20, 0.30),
        "commodity": (0.10, 0.20),
        "real_estate": (0.05, 0.15),
    }

    weights = optimize_portfolio(
        expected_returns,
        covariance_matrix,
        method="max_sharpe",
        max_weight=1.0,
        risk_free_rate=0.0,
        asset_classes=asset_classes,
        asset_class_bounds=exact_class_bounds,
    )
    summary = summarize_constraints(weights, asset_classes=asset_classes)

    assert 0.45 - 1e-8 <= summary["asset_class::equity"] <= 0.55 + 1e-8
    assert 0.20 - 1e-8 <= summary["asset_class::fixed_income"] <= 0.30 + 1e-8
    assert 0.10 - 1e-8 <= summary["asset_class::commodity"] <= 0.20 + 1e-8
    assert 0.05 - 1e-8 <= summary["asset_class::real_estate"] <= 0.15 + 1e-8


def test_optimizer_rejects_missing_target_return() -> None:
    with pytest.raises(ValueError, match="target_return"):
        optimize_portfolio(
            make_expected_returns(),
            make_covariance_matrix(),
            method="target_return",
        )


def test_optimizer_rejects_nonfinite_expected_returns_before_slsqp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("SLSQP should not run for invalid numerical inputs")

    monkeypatch.setattr("etf_portfolio.optimization.optimizer.minimize", fail_if_called)
    expected_returns = make_expected_returns()
    expected_returns["VTI"] = np.nan

    with pytest.raises(ValueError, match="expected_returns.*finite"):
        optimize_portfolio(
            expected_returns,
            make_covariance_matrix(),
            method="max_sharpe",
        )


def test_optimizer_rejects_covariance_shape_alignment_symmetry_and_finiteness() -> None:
    expected_returns = pd.Series({"AAA": 0.10, "BBB": 0.05})

    covariance_not_square = pd.DataFrame(
        [[0.04, 0.01, 0.00], [0.01, 0.09, 0.00]],
        index=["AAA", "BBB"],
        columns=["AAA", "BBB", "CCC"],
    )
    with pytest.raises(ValueError, match="square"):
        optimize_portfolio(expected_returns, covariance_not_square, method="min_variance")

    covariance_misaligned = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=["AAA", "CCC"],
        columns=["AAA", "CCC"],
    )
    with pytest.raises(ValueError, match="align"):
        optimize_portfolio(expected_returns, covariance_misaligned, method="min_variance")

    covariance_asymmetric = pd.DataFrame(
        [[0.04, 0.01], [0.02, 0.09]],
        index=expected_returns.index,
        columns=expected_returns.index,
    )
    with pytest.raises(ValueError, match="symmetric"):
        optimize_portfolio(expected_returns, covariance_asymmetric, method="min_variance")

    covariance_nonfinite = pd.DataFrame(
        [[0.04, np.inf], [np.inf, 0.09]],
        index=expected_returns.index,
        columns=expected_returns.index,
    )
    with pytest.raises(ValueError, match="finite"):
        optimize_portfolio(expected_returns, covariance_nonfinite, method="min_variance")


def test_optimizer_rejects_materially_non_psd_covariance_before_slsqp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("SLSQP should not run for invalid numerical inputs")

    monkeypatch.setattr("etf_portfolio.optimization.optimizer.minimize", fail_if_called)
    expected_returns = pd.Series({"AAA": 0.10, "BBB": 0.05})
    covariance = pd.DataFrame(
        [[1.0, 2.0], [2.0, 1.0]],
        index=expected_returns.index,
        columns=expected_returns.index,
    )

    with pytest.raises(ValueError, match="positive semidefinite"):
        optimize_portfolio(expected_returns, covariance, method="min_variance")


def test_optimizer_repairs_tiny_psd_drift_within_tolerance() -> None:
    expected_returns = pd.Series({"AAA": 0.10, "BBB": 0.05})
    covariance = pd.DataFrame(
        [[1.0, 1.0 + 5e-11], [1.0 + 5e-11, 1.0]],
        index=expected_returns.index,
        columns=expected_returns.index,
    )

    weights = optimize_portfolio(expected_returns, covariance, method="min_variance")

    assert weights.sum() == pytest.approx(1.0, abs=1e-8)


def test_frontier_returns_feasible_points() -> None:
    frontier = build_efficient_frontier(
        make_expected_returns(),
        make_covariance_matrix(),
        num_points=6,
        max_weight=0.60,
        asset_classes=make_asset_classes(),
        asset_class_bounds={
            "equity": (0.15, 0.60),
            "fixed_income": (0.15, 0.60),
            "gold": (0.0, 0.20),
            "thematic": (0.0, 0.10),
        },
        bond_assets=["BND"],
        min_bond_exposure=0.15,
        expense_ratios=make_expense_ratios(),
        max_expense_ratio=0.0030,
    )

    assert not frontier.empty
    assert set(frontier.columns) == {
        "target_return",
        "portfolio_return",
        "portfolio_volatility",
        "sharpe_ratio",
    }
    assert frontier["portfolio_volatility"].is_monotonic_increasing


def test_optimizer_logs_failed_optimization(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError, match="Optimization failed"):
            optimize_portfolio(
                make_expected_returns(),
                make_covariance_matrix(),
                method="target_return",
                max_weight=0.30,
                target_return=0.50,
            )

    failure_records = [
        record for record in caplog.records if getattr(record, "event", "") == "optimizer_failed"
    ]
    assert failure_records
    assert failure_records[-1].optimizer_status == "failed"
    assert failure_records[-1].solver_used == "SLSQP"


def test_optimizer_retries_alternate_initial_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[pd.Series] = []

    class FakeResult:
        def __init__(self, *, success: bool, x: list[float], message: str, status: int) -> None:
            self.success = success
            self.x = x
            self.message = message
            self.status = status

    def fake_minimize(*args, **kwargs):
        x0 = pd.Series(kwargs["x0"], index=make_expected_returns().index, dtype=float)
        attempts.append(x0)
        if len(attempts) == 1:
            return FakeResult(
                success=False,
                x=x0.tolist(),
                message="Positive directional derivative for linesearch",
                status=8,
            )
        return FakeResult(
            success=True,
            x=[0.55, 0.25, 0.15, 0.05],
            message="Optimization terminated successfully",
            status=0,
        )

    monkeypatch.setattr("etf_portfolio.optimization.optimizer.minimize", fake_minimize)

    weights = optimize_portfolio(
        make_expected_returns(),
        make_covariance_matrix(),
        method="max_sharpe",
        max_weight=0.65,
        risk_free_rate=0.02,
    )

    assert len(attempts) == 2
    assert not attempts[0].equals(attempts[1])
    assert weights.sum() == pytest.approx(1.0, abs=1e-8)


def test_optimizer_respects_ticker_max_bound_for_remx() -> None:
    weights = optimize_portfolio(
        make_expected_returns(),
        make_covariance_matrix(),
        method="max_sharpe",
        max_weight=0.80,
        ticker_bounds={"REMX": (0.0, 0.05)},
        risk_free_rate=0.02,
    )

    assert weights["REMX"] <= 0.05 + 1e-8


def test_optimizer_respects_ticker_min_bound_for_vti() -> None:
    expected_returns = pd.Series(
        {"VTI": 0.01, "BND": 0.08, "IAU": 0.09, "REMX": 0.10},
        name="expected_return",
    )
    weights = optimize_portfolio(
        expected_returns,
        make_covariance_matrix(),
        method="max_sharpe",
        max_weight=0.80,
        ticker_bounds={"VTI": (0.20, 0.80)},
        risk_free_rate=0.02,
    )

    assert weights["VTI"] >= 0.20 - 1e-8


def test_optimizer_rejects_infeasible_ticker_bounds() -> None:
    with pytest.raises(ValueError, match="sum\\(max_bounds\\) is below"):
        optimize_portfolio(
            make_expected_returns(),
            make_covariance_matrix(),
            method="min_variance",
            max_weight=0.80,
            ticker_bounds={
                "VTI": (0.0, 0.20),
                "BND": (0.0, 0.20),
                "IAU": (0.0, 0.20),
                "REMX": (0.0, 0.20),
            },
        )


def test_optimizer_rejects_combined_linear_infeasibility_before_slsqp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("SLSQP should not run for infeasible linear constraints")

    monkeypatch.setattr("etf_portfolio.optimization.optimizer.minimize", fail_if_called)

    with pytest.raises(ValueError, match="combined linear constraints"):
        optimize_portfolio(
            make_expected_returns(),
            make_covariance_matrix(),
            method="max_sharpe",
            max_weight=1.0,
            asset_classes=make_asset_classes(),
            asset_class_bounds={
                "equity": (0.80, 1.0),
                "fixed_income": (0.30, 1.0),
            },
        )


def test_optimizer_logs_default_cap_ticker_cap_and_realized_largest_weight(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        optimize_portfolio(
            make_expected_returns(),
            make_covariance_matrix(),
            method="min_variance",
            max_weight=0.80,
            ticker_bounds={"REMX": (0.0, 0.05)},
        )

    started = next(
        record for record in caplog.records if getattr(record, "event", "") == "optimizer_started"
    )
    completed = next(
        record for record in caplog.records if getattr(record, "event", "") == "optimizer_completed"
    )
    assert started.default_max_weight == pytest.approx(0.80)
    assert started.ticker_bound_count == 1
    assert started.tightest_ticker_cap == pytest.approx(0.05)
    assert completed.realized_largest_weight <= 0.80 + 1e-8


def test_frontier_logs_skipped_infeasible_points(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fake_optimize_portfolio(*args, **kwargs):
        if kwargs.get("method") == "min_variance":
            return pd.Series(
                [0.35, 0.45, 0.15, 0.05],
                index=make_expected_returns().index,
                dtype=float,
                name="weight",
            )
        if kwargs.get("method") == "target_volatility":
            return pd.Series(
                [0.70, 0.10, 0.10, 0.10],
                index=make_expected_returns().index,
                dtype=float,
                name="weight",
            )
        if kwargs.get("target_return", 0.0) > 0.085:
            raise ValueError("Optimization failed: infeasible target.")
        return pd.Series(
            [0.50, 0.30, 0.15, 0.05],
            index=make_expected_returns().index,
            dtype=float,
            name="weight",
        )

    monkeypatch.setattr(frontier_module, "optimize_portfolio", fake_optimize_portfolio)

    with caplog.at_level(logging.WARNING):
        frontier = build_efficient_frontier(
            make_expected_returns(),
            make_covariance_matrix(),
            num_points=3,
            max_weight=0.60,
        )

    assert not frontier.empty
    warning_records = [
        record
        for record in caplog.records
        if getattr(record, "event", "") == "efficient_frontier_point_failed"
    ]
    assert warning_records
    assert warning_records[0].optimizer_status == "failed"


def test_frontier_targets_start_at_constrained_min_vol_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_calls: list[float] = []
    expected_returns = make_expected_returns()
    covariance = make_covariance_matrix()

    def fake_optimize_portfolio(*args, **kwargs):
        method = kwargs.get("method")
        if method == "min_variance":
            return pd.Series(
                [0.0675, 0.0, 0.0, 0.0],
                index=expected_returns.index,
                dtype=float,
                name="weight",
            )
        if method == "target_volatility":
            return pd.Series(
                [0.10, 0.0, 0.0, 0.0],
                index=expected_returns.index,
                dtype=float,
                name="weight",
            )
        target = float(kwargs["target_return"])
        target_calls.append(target)
        return pd.Series(
            [target, 0.0, 0.0, 0.0],
            index=expected_returns.index,
            dtype=float,
            name="weight",
        )

    monkeypatch.setattr(frontier_module, "optimize_portfolio", fake_optimize_portfolio)
    monkeypatch.setattr(
        frontier_module,
        "calculate_portfolio_return",
        lambda weights, _: float(weights.iloc[0]),
    )
    monkeypatch.setattr(
        frontier_module,
        "calculate_portfolio_volatility",
        lambda weights, __: abs(float(weights.iloc[0])) + 1e-6,
    )

    frontier = build_efficient_frontier(
        expected_returns,
        covariance,
        num_points=5,
        max_weight=0.60,
    )

    constrained_min_vol_return = 0.0675
    constrained_max_return = expected_returns["VTI"]
    assert min(target_calls) == pytest.approx(constrained_min_vol_return)
    assert max(target_calls) == pytest.approx(constrained_max_return)
    assert min(target_calls) > float(expected_returns.min())
    assert frontier["portfolio_return"].nunique() == 5


def test_frontier_passes_ticker_and_asset_class_constraints_to_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_calls: list[dict[str, object]] = []
    expected_returns = make_expected_returns()

    def fake_optimize_portfolio(*args, **kwargs):
        captured_calls.append(kwargs.copy())
        return pd.Series(
            [0.25, 0.25, 0.25, 0.25],
            index=expected_returns.index,
            dtype=float,
            name="weight",
        )

    monkeypatch.setattr(frontier_module, "optimize_portfolio", fake_optimize_portfolio)

    asset_classes = pd.Series(
        {
            "VTI": "equity",
            "BND": "fixed_income",
            "IAU": "commodity",
            "REMX": "equity",
        }
    )
    asset_class_bounds = {"equity": (0.20, 0.80), "fixed_income": (0.10, 0.40)}
    ticker_bounds = {"REMX": (0.0, 0.05), "VTI": (0.20, 0.80)}

    build_efficient_frontier(
        expected_returns,
        make_covariance_matrix(),
        num_points=3,
        max_weight=0.80,
        asset_classes=asset_classes,
        asset_class_bounds=asset_class_bounds,
        ticker_bounds=ticker_bounds,
        bond_assets=["BND"],
        min_bond_exposure=0.10,
    )

    assert len(captured_calls) >= 3
    for call in captured_calls:
        assert call["asset_classes"] is asset_classes
        assert call["asset_class_bounds"] == asset_class_bounds
        assert call["ticker_bounds"] == ticker_bounds
        assert call["bond_assets"] == ["BND"]
        assert call["min_bond_exposure"] == pytest.approx(0.10)
