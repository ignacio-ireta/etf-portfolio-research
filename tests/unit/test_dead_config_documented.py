"""Tests that settings with no effect are labeled as such and unused dependencies stay out."""

from __future__ import annotations

import tomllib
from pathlib import Path

from etf_portfolio.config import InvestorProfileConfig, load_config

REPO_ROOT = Path(__file__).parents[2]

INVESTOR_PROFILE_BLOCK = """investor_profile:
  horizon_years: 35
  objective: long_term_accumulation
  tax_preference: minimize_realized_gains
"""


def test_investor_profile_is_optional(tiny_project: Path) -> None:
    config_path = tiny_project / "configs/base.yaml"
    original = config_path.read_text(encoding="utf-8")
    assert INVESTOR_PROFILE_BLOCK in original
    config_path.write_text(original.replace(INVESTOR_PROFILE_BLOCK, ""), encoding="utf-8")

    config = load_config(config_path)

    assert config.investor_profile is None


def test_investor_profile_still_loads_when_present(tiny_project: Path) -> None:
    config = load_config(tiny_project / "configs/base.yaml")

    assert isinstance(config.investor_profile, InvestorProfileConfig)
    assert config.investor_profile.horizon_years == 35


def test_investor_profile_docstring_says_it_has_no_effect() -> None:
    assert "No calculation reads" in (InvestorProfileConfig.__doc__ or "")


def test_base_config_labels_settings_that_have_no_effect() -> None:
    text = (REPO_ROOT / "configs/base.yaml").read_text(encoding="utf-8")

    assert "DESCRIPTIVE ONLY" in text
    assert "NO EFFECT in this file" in text
    load_config(REPO_ROOT / "configs/base.yaml")


def test_pyportfolioopt_is_not_a_declared_dependency() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    declared = [dependency.lower() for dependency in pyproject["project"]["dependencies"]]
    assert not any(dependency.startswith("pyportfolioopt") for dependency in declared)
