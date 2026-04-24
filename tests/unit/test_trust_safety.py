from __future__ import annotations

from pathlib import Path

from etf_portfolio.trust_safety import (
    common_false_conclusions_markdown,
    common_false_conclusions_table,
)


def test_trust_safety_docs_are_generated_from_canonical_source() -> None:
    docs_path = Path(__file__).parents[2] / "docs" / "trust_and_safety.md"

    assert docs_path.read_text(encoding="utf-8") == common_false_conclusions_markdown()


def test_common_false_conclusions_cover_required_warnings() -> None:
    table = common_false_conclusions_table()
    text = " ".join(table.astype(str).to_numpy().ravel())

    assert list(table.columns) == [
        "Common False Conclusion",
        "Safer Reading",
        "Why It Matters",
        "Where To Check",
    ]
    assert "Sharpe ratio" in text
    assert "future returns" in text
    assert "overfit" in text
    assert "Historical mean returns" in text
    assert "Expense ratios" in text
    assert "Contribution-only" in text
    assert "drift" in text
