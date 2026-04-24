"""Canonical trust-and-safety guidance for interpreting research outputs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FalseConclusion:
    """A common misreading and the safer interpretation users should apply."""

    false_conclusion: str
    safer_reading: str
    why_it_matters: str
    where_to_check: str


COMMON_FALSE_CONCLUSIONS: tuple[FalseConclusion, ...] = (
    FalseConclusion(
        false_conclusion="A high Sharpe ratio guarantees strong future returns.",
        safer_reading=(
            "A high Sharpe ratio means the portfolio earned more historical excess return "
            "per unit of historical volatility in this sample."
        ),
        why_it_matters=(
            "Sharpe ratios can fall when returns, volatility, correlations, or rates change."
        ),
        where_to_check="Metric Dictionary, Backtest Performance, Rolling Risk Metrics",
    ),
    FalseConclusion(
        false_conclusion="A good backtest proves the strategy will work.",
        safer_reading=(
            "A backtest is a historical experiment under chosen data, costs, constraints, "
            "rebalance rules, and benchmark assumptions."
        ),
        why_it_matters=(
            "Strategies can overfit the past, especially when many universes, objectives, "
            "windows, or constraints are tried."
        ),
        where_to_check="Reader Guide, Assumptions and Limitations, Stress Periods",
    ),
    FalseConclusion(
        false_conclusion="Historical mean returns are reliable forecasts.",
        safer_reading=(
            "Historical mean returns are transparent baseline estimates, not dependable "
            "predictions of future expected returns."
        ),
        why_it_matters=(
            "Mean estimates are noisy and can be dominated by sample period, regime changes, "
            "and recent winners."
        ),
        where_to_check="Metric Dictionary, Efficient Frontier Chart, Assumptions and Limitations",
    ),
    FalseConclusion(
        false_conclusion="ETF expense ratios are the only investment cost that matters.",
        safer_reading=(
            "Expense ratios matter because they compound, but real implementation can also "
            "include spreads, slippage, taxes, FX, account fees, and liquidity costs."
        ),
        why_it_matters=(
            "A low expense ratio does not automatically make an ETF cheaper to hold or trade "
            "in every investor's situation."
        ),
        where_to_check="Weighted Expense Ratio Over Time, Assumptions and Limitations",
    ),
    FalseConclusion(
        false_conclusion="Contribution-only portfolios always match optimizer targets.",
        safer_reading=(
            "Contribution-only rebalancing uses new cash to move toward targets, but realized "
            "holdings can drift when selling is restricted."
        ),
        why_it_matters=(
            "The actual simulated portfolio may carry different risks than the optimizer "
            "target, especially after large market moves."
        ),
        where_to_check=(
            "Latest Realized Portfolio Table, Optimizer Target Portfolio Table, "
            "Realized Constraint Warnings"
        ),
    ),
)


def common_false_conclusions_table() -> pd.DataFrame:
    """Return common false conclusions as a report-ready table."""

    return pd.DataFrame(
        [
            {
                "Common False Conclusion": item.false_conclusion,
                "Safer Reading": item.safer_reading,
                "Why It Matters": item.why_it_matters,
                "Where To Check": item.where_to_check,
            }
            for item in COMMON_FALSE_CONCLUSIONS
        ]
    )


def common_false_conclusions_markdown() -> str:
    """Render canonical trust-and-safety guidance as documentation markdown."""

    lines = [
        "# Trust And Safety",
        "",
        "This page is generated from `src/etf_portfolio/trust_safety.py`, which is "
        "the canonical source for common false conclusions and safer report readings.",
        "",
        "The project is a research tool, not financial advice. Use these warnings before "
        "turning any metric, chart, or optimized allocation into a conclusion.",
        "",
        "## Common False Conclusions",
        "",
    ]
    for item in COMMON_FALSE_CONCLUSIONS:
        lines.extend(
            [
                f"### {item.false_conclusion}",
                "",
                f"**Safer reading:** {item.safer_reading}",
                "",
                f"**Why it matters:** {item.why_it_matters}",
                "",
                f"**Where to check:** {item.where_to_check}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "COMMON_FALSE_CONCLUSIONS",
    "FalseConclusion",
    "common_false_conclusions_markdown",
    "common_false_conclusions_table",
]
