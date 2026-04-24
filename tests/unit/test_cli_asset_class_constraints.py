from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from etf_portfolio import cli
from etf_portfolio.config import load_config
from etf_portfolio.data.ingest import load_etf_universe_metadata


def test_asset_class_bounds_align_with_metadata_taxonomy() -> None:
    config = load_config("configs/base.yaml")
    metadata = load_etf_universe_metadata(Path("data/metadata/etf_universe.csv")).set_index(
        "ticker"
    )
    asset_class_map = metadata.reindex(config.universe.tickers)["asset_class"]

    bounds = cli._asset_class_bounds(config, asset_class_map)

    assert bounds is not None
    assert set(bounds) == {"equity", "fixed_income", "commodity", "real_estate"}
    assert cli._min_bond_exposure(config) == bounds["fixed_income"][0]


def test_asset_class_bounds_raise_when_configured_classes_match_no_metadata() -> None:
    config = load_config("configs/base.yaml")
    fake_classes = pd.Series(
        ["digital_assets", "volatility", "cash"],
        index=["VTI", "BND", "IAU"],
        dtype="object",
    )

    with pytest.raises(
        ValueError,
        match="do not match any metadata asset_class values",
    ):
        cli._asset_class_bounds(config, fake_classes)


def test_build_optimization_constraints_returns_shared_constraint_bundle() -> None:
    config = load_config("configs/base.yaml")
    metadata = load_etf_universe_metadata(Path("data/metadata/etf_universe.csv")).set_index(
        "ticker"
    )

    constraints = cli._build_optimization_constraints(
        config,
        metadata,
        pd.Index(["VTI", "BND", "VNQ", "REMX"]),
    )

    assert constraints["asset_classes"].to_dict() == {
        "VTI": "equity",
        "BND": "fixed_income",
        "VNQ": "real_estate",
        "REMX": "equity",
    }
    assert constraints["expense_ratios"].to_dict() == {
        "VTI": pytest.approx(0.0003),
        "BND": pytest.approx(0.0003),
        "VNQ": pytest.approx(0.0013),
        "REMX": pytest.approx(0.0058),
    }
    assert constraints["ticker_bounds"] == {
        "VTI": (0.20, 0.60),
        "BND": (0.00, 0.40),
        "VNQ": (0.00, 0.10),
        "REMX": (0.00, 0.05),
    }
    assert constraints["asset_class_bounds"] is not None
    assert constraints["asset_class_bounds"]["fixed_income"] == (0.10, 0.45)
    assert constraints["asset_class_bounds"]["real_estate"] == (0.00, 0.10)
    assert constraints["bond_assets"] == ["BND"]
    assert constraints["min_bond_exposure"] == pytest.approx(0.10)
