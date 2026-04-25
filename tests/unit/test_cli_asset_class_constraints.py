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
        match="unknown asset_class values",
    ):
        cli._asset_class_bounds(config, fake_classes)


def test_ticker_bounds_raise_for_missing_available_columns() -> None:
    config = load_config("configs/base.yaml")

    with pytest.raises(ValueError, match="unknown tickers"):
        cli._ticker_bounds(config, pd.Index(["VTI", "BND", "VNQ", "REMX"]))


def test_build_optimization_constraints_returns_shared_constraint_bundle() -> None:
    config = load_config("configs/base.yaml")
    metadata = load_etf_universe_metadata(Path("data/metadata/etf_universe.csv")).set_index(
        "ticker"
    )

    constraints = cli._build_optimization_constraints(
        config,
        metadata,
        pd.Index(config.universe.tickers),
    )

    assert constraints["asset_classes"]["VTI"] == "equity"
    assert constraints["asset_classes"]["BND"] == "fixed_income"
    assert constraints["asset_classes"]["VNQ"] == "real_estate"
    assert constraints["asset_classes"]["REMX"] == "equity"
    assert constraints["expense_ratios"]["VTI"] == pytest.approx(0.0003)
    assert constraints["expense_ratios"]["BND"] == pytest.approx(0.0003)
    assert constraints["expense_ratios"]["VNQ"] == pytest.approx(0.0013)
    assert constraints["expense_ratios"]["REMX"] == pytest.approx(0.0058)
    assert set(constraints["ticker_bounds"]) == set(config.constraints.ticker_bounds)
    assert constraints["ticker_bounds"]["VTI"] == (0.20, 0.60)
    assert constraints["ticker_bounds"]["BND"] == (0.00, 0.40)
    assert constraints["ticker_bounds"]["VNQ"] == (0.00, 0.10)
    assert constraints["ticker_bounds"]["REMX"] == (0.00, 0.05)
    assert constraints["asset_class_bounds"] is not None
    assert constraints["asset_class_bounds"]["fixed_income"] == (0.10, 0.45)
    assert constraints["asset_class_bounds"]["real_estate"] == (0.00, 0.10)
    assert constraints["bond_assets"] == ["BND", "IEI", "TIP", "TLT"]
    assert constraints["min_bond_exposure"] == pytest.approx(0.10)
