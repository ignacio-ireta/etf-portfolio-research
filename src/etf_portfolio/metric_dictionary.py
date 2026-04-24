"""Canonical explanations for metrics shown by the research pipeline."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MetricDefinition:
    """Human-readable definition for one reported metric."""

    name: str
    category: str
    plain_english: str
    formula_summary: str
    good_bad_interpretation: str
    caveats: str


METRIC_DEFINITIONS: tuple[MetricDefinition, ...] = (
    MetricDefinition(
        name="Portfolio Weight",
        category="Portfolio Construction",
        plain_english="The share of the portfolio allocated to one ETF or group.",
        formula_summary="Asset weight = asset market value / total portfolio value.",
        good_bad_interpretation=(
            "Weights should match the intended exposure. Very large weights indicate concentration."
        ),
        caveats="A small weight can still create large risk if the asset is very volatile.",
    ),
    MetricDefinition(
        name="Expected Return",
        category="Optimization Inputs",
        plain_english="The return estimate the optimizer uses before choosing weights.",
        formula_summary=(
            "Currently estimated from historical mean returns in the trailing training window."
        ),
        good_bad_interpretation=(
            "Higher expected return can make an asset more attractive to the optimizer."
        ),
        caveats=("This is backward-looking and should not be read as a reliable return forecast."),
    ),
    MetricDefinition(
        name="Covariance",
        category="Optimization Inputs",
        plain_english="How asset returns moved together historically.",
        formula_summary=(
            "Covariance matrix estimated from aligned asset return histories using the "
            "configured risk model."
        ),
        good_bad_interpretation=(
            "Lower or negative covariance can improve diversification if relationships persist."
        ),
        caveats="Covariance estimates can change quickly across market regimes.",
    ),
    MetricDefinition(
        name="Portfolio Return",
        category="Performance",
        plain_english="The portfolio's percentage gain or loss over one period.",
        formula_summary="Sum of asset weight times asset return for the period.",
        good_bad_interpretation="Higher is better for a single period, all else equal.",
        caveats="One-period return says little about risk, consistency, or future returns.",
    ),
    MetricDefinition(
        name="Cumulative Return",
        category="Performance",
        plain_english="How much the portfolio grew or shrank over the full path.",
        formula_summary="Compound each period: product of (1 + return) minus 1.",
        good_bad_interpretation=(
            "Higher cumulative return is better, but only after risk is checked."
        ),
        caveats="Can hide severe losses or long weak stretches along the way.",
    ),
    MetricDefinition(
        name="CAGR",
        category="Performance",
        plain_english="The annualized growth rate implied by the compounded return path.",
        formula_summary=(
            "Compound total return over the sample, then convert it to a one-year rate."
        ),
        good_bad_interpretation="Higher CAGR is better if risk and drawdowns are acceptable.",
        caveats=(
            "Smooths the path into one number and can make volatile results look cleaner "
            "than they felt."
        ),
    ),
    MetricDefinition(
        name="Annualized Volatility",
        category="Risk",
        plain_english="How much portfolio returns fluctuated, scaled to a yearly number.",
        formula_summary="Standard deviation of periodic returns times sqrt(periods per year).",
        good_bad_interpretation="Lower volatility is usually a smoother ride.",
        caveats="Does not distinguish upside surprises from downside losses.",
    ),
    MetricDefinition(
        name="Portfolio Volatility",
        category="Risk",
        plain_english="Estimated total variability of a weighted portfolio.",
        formula_summary="Square root of weights' transpose times covariance matrix times weights.",
        good_bad_interpretation="Lower estimated volatility means less modeled return variability.",
        caveats="Only as reliable as the covariance estimate and portfolio weights.",
    ),
    MetricDefinition(
        name="Sharpe Ratio",
        category="Risk-Adjusted Return",
        plain_english="Return earned per unit of total volatility after the risk-free rate.",
        formula_summary="Annualized excess return divided by annualized volatility.",
        good_bad_interpretation=(
            "Higher is generally better; below zero means underperforming cash."
        ),
        caveats="Penalizes upside and downside volatility equally.",
    ),
    MetricDefinition(
        name="Sortino Ratio",
        category="Risk-Adjusted Return",
        plain_english="Return earned per unit of downside volatility.",
        formula_summary="Annualized excess return divided by annualized downside deviation.",
        good_bad_interpretation="Higher is generally better when downside risk matters most.",
        caveats="Can be unstable when there are few negative observations.",
    ),
    MetricDefinition(
        name="Max Drawdown",
        category="Drawdown",
        plain_english="The worst historical fall from a previous peak.",
        formula_summary=(
            "Minimum value of cumulative wealth divided by its prior running peak minus 1."
        ),
        good_bad_interpretation=(
            "Closer to zero is better. A more negative value means deeper loss."
        ),
        caveats="Only captures the worst observed historical loss, not every possible loss.",
    ),
    MetricDefinition(
        name="Drawdown",
        category="Drawdown",
        plain_english="The current decline from a previous high-water mark.",
        formula_summary="Current cumulative wealth / prior running maximum wealth minus 1.",
        good_bad_interpretation="Closer to zero is better; deeper negatives mean larger losses.",
        caveats="Does not show how long recovery took unless read with the full chart.",
    ),
    MetricDefinition(
        name="Calmar Ratio",
        category="Risk-Adjusted Return",
        plain_english="Compounded return compared with the worst drawdown.",
        formula_summary="CAGR divided by absolute value of maximum drawdown.",
        good_bad_interpretation="Higher is generally better if the drawdown estimate is credible.",
        caveats="Very sensitive to one worst drawdown observation.",
    ),
    MetricDefinition(
        name="Turnover",
        category="Implementation",
        plain_english="How much of the portfolio changed at rebalances on average.",
        formula_summary="Average sum of absolute weight changes across rebalance dates.",
        good_bad_interpretation="Lower usually means fewer trades, lower costs, and less tax drag.",
        caveats="The project models costs simply; real execution and taxes can differ.",
    ),
    MetricDefinition(
        name="Average Number of Holdings",
        category="Concentration",
        plain_english="The average count of ETFs with non-trivial portfolio weights.",
        formula_summary="Average count of weights above a small tolerance at each rebalance.",
        good_bad_interpretation="Higher can mean broader diversification.",
        caveats="More holdings do not guarantee better diversification if exposures overlap.",
    ),
    MetricDefinition(
        name="Largest Position",
        category="Concentration",
        plain_english="The biggest single ETF weight observed in the portfolio history.",
        formula_summary="Maximum asset weight across all rebalance dates.",
        good_bad_interpretation="Lower usually means less single-ETF concentration.",
        caveats=(
            "A broad ETF can be less risky than its weight suggests; a narrow ETF can be riskier."
        ),
    ),
    MetricDefinition(
        name="Herfindahl Concentration Index",
        category="Concentration",
        plain_english="A concentration score based on squared portfolio weights.",
        formula_summary="Average across rebalances of the sum of squared weights.",
        good_bad_interpretation=(
            "Lower means more evenly spread weights; higher means concentration."
        ),
        caveats="Does not know whether ETFs hold overlapping underlying securities.",
    ),
    MetricDefinition(
        name="Worst Month",
        category="Period Extremes",
        plain_english="The worst compounded calendar-month return in the sample.",
        formula_summary="Compound returns by month, then take the minimum monthly return.",
        good_bad_interpretation="Closer to zero is better.",
        caveats="A bad period just outside month boundaries may be split across months.",
    ),
    MetricDefinition(
        name="Worst Quarter",
        category="Period Extremes",
        plain_english="The worst compounded calendar-quarter return in the sample.",
        formula_summary="Compound returns by quarter, then take the minimum quarterly return.",
        good_bad_interpretation="Closer to zero is better.",
        caveats="Quarterly windows are conventional but not the only stress window that matters.",
    ),
    MetricDefinition(
        name="Best Month",
        category="Period Extremes",
        plain_english="The best compounded calendar-month return in the sample.",
        formula_summary="Compound returns by month, then take the maximum monthly return.",
        good_bad_interpretation="Higher is better, but not if it came with unacceptable risk.",
        caveats="Upside extremes can make a strategy look exciting without proving robustness.",
    ),
    MetricDefinition(
        name="Beta",
        category="Benchmark Relative",
        plain_english="How sensitive the portfolio was to benchmark moves.",
        formula_summary=(
            "Covariance of portfolio and benchmark returns divided by benchmark variance."
        ),
        good_bad_interpretation=(
            "Beta above 1 moved more than the benchmark; below 1 moved less; near 0 moved "
            "independently."
        ),
        caveats="Beta depends on the chosen benchmark and historical window.",
    ),
    MetricDefinition(
        name="Alpha",
        category="Benchmark Relative",
        plain_english="Return not explained by benchmark exposure in a simple beta model.",
        formula_summary=(
            "Portfolio CAGR minus (risk-free rate plus beta times benchmark excess return)."
        ),
        good_bad_interpretation="Higher historical alpha is better, but it is not proof of skill.",
        caveats="Alpha can vanish when the benchmark, period, or risk model changes.",
    ),
    MetricDefinition(
        name="Tracking Error",
        category="Benchmark Relative",
        plain_english="How much the portfolio's returns differed from the benchmark.",
        formula_summary=(
            "Standard deviation of active returns, annualized. Active return is portfolio "
            "return minus benchmark return."
        ),
        good_bad_interpretation=(
            "Lower means benchmark-like behavior; higher means more benchmark-relative risk."
        ),
        caveats="Low tracking error is not automatically good if the benchmark is unsuitable.",
    ),
    MetricDefinition(
        name="Information Ratio",
        category="Benchmark Relative",
        plain_english="Benchmark-relative return per unit of tracking error.",
        formula_summary="Average active return divided by active return volatility, annualized.",
        good_bad_interpretation="Higher is better for benchmark-relative strategies.",
        caveats="Can be unstable when tracking error is very small.",
    ),
    MetricDefinition(
        name="Rolling Volatility",
        category="Rolling Risk",
        plain_english="Volatility measured repeatedly over moving windows.",
        formula_summary="Rolling standard deviation of returns times sqrt(periods per year).",
        good_bad_interpretation="Stable or lower rolling volatility suggests steadier behavior.",
        caveats="Window length strongly affects the result.",
    ),
    MetricDefinition(
        name="Rolling Sharpe",
        category="Rolling Risk",
        plain_english="Sharpe ratio measured repeatedly over moving windows.",
        formula_summary="Rolling mean excess return divided by rolling volatility, annualized.",
        good_bad_interpretation=(
            "Consistently positive values are better than one isolated high value."
        ),
        caveats="Short windows are noisy and can flip quickly.",
    ),
    MetricDefinition(
        name="Rolling Correlation",
        category="Rolling Risk",
        plain_english="How closely the portfolio moved with the benchmark over moving windows.",
        formula_summary="Rolling correlation between portfolio and benchmark returns.",
        good_bad_interpretation=(
            "Lower correlation can mean diversification; higher correlation means benchmark-like "
            "movement."
        ),
        caveats="Correlation can rise during crises when diversification is most needed.",
    ),
    MetricDefinition(
        name="Stress-Period Return",
        category="Stress Testing",
        plain_english="Compounded return during a named difficult historical period.",
        formula_summary="Product of (1 + return) within the stress window minus 1.",
        good_bad_interpretation="Less negative is better during market stress.",
        caveats="Historical stress windows do not cover every future crisis shape.",
    ),
    MetricDefinition(
        name="Weighted Expense Ratio",
        category="Cost",
        plain_english="The portfolio-level annual ETF fee implied by current weights.",
        formula_summary="Sum of each ETF weight times its expense ratio.",
        good_bad_interpretation="Lower is usually better for similar exposures.",
        caveats="Does not include taxes, spreads, market impact, or broker-specific costs.",
    ),
    MetricDefinition(
        name="Return Attribution",
        category="Attribution",
        plain_english="How much each asset or group contributed to portfolio return.",
        formula_summary=(
            "Per-period contribution is asset weight times asset return, then aggregated."
        ),
        good_bad_interpretation="Positive contributors helped historical return; negative hurt it.",
        caveats="Explains past contribution and does not identify future winners.",
    ),
    MetricDefinition(
        name="Risk Attribution",
        category="Attribution",
        plain_english="How much each asset or group contributed to portfolio volatility.",
        formula_summary="Euler decomposition using weights and the covariance matrix.",
        good_bad_interpretation=(
            "Lower, more balanced risk contribution usually means less hidden risk."
        ),
        caveats="Depends heavily on the covariance estimate.",
    ),
    MetricDefinition(
        name="Efficient Frontier",
        category="Optimization Outputs",
        plain_english="A set of modeled portfolios showing risk and return tradeoffs.",
        formula_summary="Optimized portfolios plotted by expected return and estimated volatility.",
        good_bad_interpretation=(
            "Portfolios higher and left look better under the model, subject to constraints."
        ),
        caveats="The frontier is estimate-driven and not a forecast.",
    ),
    MetricDefinition(
        name="Accuracy",
        category="ML Evaluation",
        plain_english="The share of classification predictions that matched actual outcomes.",
        formula_summary="Correct predictions divided by total predictions.",
        good_bad_interpretation=(
            "Higher is better when classes are balanced and errors cost the same."
        ),
        caveats="Can mislead when one class is much more common than another.",
    ),
    MetricDefinition(
        name="Log Loss",
        category="ML Evaluation",
        plain_english="How well classification probabilities matched actual outcomes.",
        formula_summary="Negative average log likelihood of the true class probability.",
        good_bad_interpretation=(
            "Lower is better; confident wrong predictions are penalized heavily."
        ),
        caveats="Requires calibrated probabilities to be meaningful.",
    ),
    MetricDefinition(
        name="ROC AUC",
        category="ML Evaluation",
        plain_english="How well a classifier ranked positives above negatives.",
        formula_summary="Area under the receiver operating characteristic curve.",
        good_bad_interpretation="Higher is better; 0.5 is roughly random ranking.",
        caveats="Does not choose a trading threshold or account for economic payoff.",
    ),
    MetricDefinition(
        name="RMSE",
        category="ML Evaluation",
        plain_english="Typical regression prediction error with larger errors penalized more.",
        formula_summary="Square root of mean squared prediction error.",
        good_bad_interpretation="Lower is better.",
        caveats="Sensitive to outliers.",
    ),
    MetricDefinition(
        name="MAE",
        category="ML Evaluation",
        plain_english="Typical absolute regression prediction error.",
        formula_summary="Mean absolute value of actual minus predicted values.",
        good_bad_interpretation="Lower is better.",
        caveats="Does not penalize large misses as strongly as RMSE.",
    ),
    MetricDefinition(
        name="R2",
        category="ML Evaluation",
        plain_english="How much target variation the regression model explained.",
        formula_summary="One minus residual sum of squares divided by total sum of squares.",
        good_bad_interpretation=(
            "Higher is better; values below zero are worse than a mean forecast."
        ),
        caveats="Can look good in-sample while failing out of sample.",
    ),
)


def metric_dictionary_table(
    *,
    categories: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return metric definitions as a report-ready table."""

    allowed_categories = set(categories) if categories is not None else None
    rows = [
        {
            "Metric": definition.name,
            "Category": definition.category,
            "Plain-English Meaning": definition.plain_english,
            "Formula-Level Summary": definition.formula_summary,
            "Good/Bad Interpretation": definition.good_bad_interpretation,
            "Caveats": definition.caveats,
        }
        for definition in METRIC_DEFINITIONS
        if allowed_categories is None or definition.category in allowed_categories
    ]
    return pd.DataFrame(rows)


def get_metric_definition(metric_name: str) -> MetricDefinition:
    """Look up one metric definition by exact metric name."""

    for definition in METRIC_DEFINITIONS:
        if definition.name == metric_name:
            return definition
    raise KeyError(f"Unknown metric: {metric_name}")


def metric_dictionary_markdown() -> str:
    """Render the canonical metric dictionary as documentation markdown."""

    lines = [
        "# Metric Dictionary",
        "",
        "This page is generated from `src/etf_portfolio/metric_dictionary.py`, which is "
        "the canonical explanation source for metrics shown by the pipeline and report.",
        "",
        "Each metric is defined with plain-English meaning, formula-level summary, "
        "good/bad interpretation, and caveats.",
        "",
    ]
    categories = list(dict.fromkeys(definition.category for definition in METRIC_DEFINITIONS))
    for category in categories:
        lines.extend([f"## {category}", ""])
        for definition in (
            definition for definition in METRIC_DEFINITIONS if definition.category == category
        ):
            lines.extend(
                [
                    f"### {definition.name}",
                    "",
                    f"**Plain-English meaning:** {definition.plain_english}",
                    "",
                    f"**Formula-level summary:** {definition.formula_summary}",
                    "",
                    f"**Good/bad interpretation:** {definition.good_bad_interpretation}",
                    "",
                    f"**Caveats:** {definition.caveats}",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "METRIC_DEFINITIONS",
    "MetricDefinition",
    "get_metric_definition",
    "metric_dictionary_markdown",
    "metric_dictionary_table",
]
