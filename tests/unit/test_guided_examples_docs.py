from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
GUIDED_EXAMPLES = REPO_ROOT / "docs" / "guided_examples.md"


def test_guided_examples_cover_required_walkthroughs() -> None:
    text = GUIDED_EXAMPLES.read_text(encoding="utf-8")

    assert "# Guided Examples" in text
    assert "## Example 1: Default Run" in text
    assert "## Example 2: Changing ETFs" in text
    assert "## Example 3: Interpreting A Backtest" in text
    assert "research tool, not financial advice" in text


def test_guided_examples_reference_existing_report_artifacts() -> None:
    artifact_paths = [
        "reports/runs/backtest_run-all-20260424T175044Z-3777911e.json",
        "reports/html/latest_report.html",
        "reports/excel/portfolio_results.xlsx",
        "reports/metrics/backtest_metrics.json",
        "reports/figures/cumulative_returns.png",
        "reports/figures/drawdown.png",
        "reports/figures/portfolio_weights.png",
        "reports/figures/benchmark_comparison.png",
        "reports/figures/rolling_volatility.png",
        "reports/figures/rolling_sharpe.png",
        "reports/figures/stress_periods.png",
    ]

    text = GUIDED_EXAMPLES.read_text(encoding="utf-8")
    for artifact_path in artifact_paths:
        assert artifact_path in text
        assert (REPO_ROOT / artifact_path).exists()


def test_beginner_docs_link_guided_examples() -> None:
    expected_links = {
        "README.md": "docs/guided_examples.md",
        "docs/start_here.md": "guided_examples.md",
        "docs/runbook.md": "guided_examples.md",
    }

    for relative_path, expected_link in expected_links.items():
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert expected_link in text


def test_beginner_docs_link_trust_and_safety() -> None:
    expected_links = {
        "README.md": "docs/trust_and_safety.md",
        "docs/start_here.md": "trust_and_safety.md",
        "docs/how_to_read_the_report.md": "trust_and_safety.md",
        "docs/interpretation_guide.md": "trust_and_safety.md",
        "docs/assumptions_and_limitations.md": "trust_and_safety.md",
        "docs/guided_examples.md": "trust_and_safety.md",
    }

    for relative_path, expected_link in expected_links.items():
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert expected_link in text
