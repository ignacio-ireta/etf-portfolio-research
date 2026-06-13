"""Golden-file test for the backtest metrics output contract.

Snapshots the optimized-strategy metrics for a fixed synthetic input + config.
Regenerate after an intentional change with::

    UPDATE_GOLDEN=1 uv run pytest tests/golden

The configured strategy is equal-weight, so the snapshot is deterministic and
not subject to optimizer/platform numerical drift. A small tolerance guards the
last-bit float noise. See docs/testing.md.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from etf_portfolio import cli

pytestmark = [pytest.mark.golden, pytest.mark.slow]

GOLDEN_PATH = Path(__file__).parent / "backtest_metrics_optimized_strategy.json"


def _normalized_metrics(payload: dict) -> dict[str, float]:
    return {key: round(float(value), 8) for key, value in payload["optimized_strategy"].items()}


def test_backtest_optimized_strategy_metrics_match_golden(
    tiny_project: Path, synthetic_prices: pd.DataFrame
) -> None:
    processed = tiny_project / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    synthetic_prices.to_parquet(processed / "prices_validated.parquet")
    synthetic_prices.pct_change(fill_method=None).dropna().to_parquet(processed / "returns.parquet")

    cli.run_backtest("configs/base.yaml", project_root=tiny_project, lookback_periods=5)
    payload = json.loads(
        (tiny_project / "reports/metrics/backtest_metrics.json").read_text(encoding="utf-8")
    )
    actual = _normalized_metrics(payload)

    assert payload["schema_version"] == "1.0"

    if os.environ.get("UPDATE_GOLDEN") or not GOLDEN_PATH.exists():
        GOLDEN_PATH.write_text(json.dumps(actual, indent=2, sort_keys=True), encoding="utf-8")
        pytest.skip("golden snapshot regenerated")

    expected = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert set(actual) == set(expected), "metric keys drifted from the golden contract"
    for key, value in expected.items():
        assert actual[key] == pytest.approx(value, rel=1e-4, abs=1e-6), key
