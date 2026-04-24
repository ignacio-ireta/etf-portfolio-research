from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from etf_portfolio.ml.train import write_metrics_json


def test_write_metrics_json_sanitizes_nonstandard_values(tmp_path: Path) -> None:
    output_path = tmp_path / "metrics.json"
    payload = {
        "float_scalar": np.float64(1.25),
        "int_scalar": np.int64(7),
        "timestamp": pd.Timestamp("2024-01-31 12:34:56"),
        "nan_value": np.nan,
        "pos_inf": np.inf,
        "neg_inf": -np.inf,
        "array": np.array([np.float64(2.5), np.nan]),
    }

    write_metrics_json(payload, output_path)

    raw_text = output_path.read_text(encoding="utf-8")
    assert "NaN" not in raw_text
    assert "Infinity" not in raw_text

    parsed = json.loads(raw_text)
    assert parsed["float_scalar"] == 1.25
    assert parsed["int_scalar"] == 7
    assert parsed["timestamp"] == "2024-01-31T12:34:56"
    assert parsed["nan_value"] is None
    assert parsed["pos_inf"] is None
    assert parsed["neg_inf"] is None
    assert parsed["array"] == [2.5, None]
