"""Tests for the domain error taxonomy and exit-code mapping."""

from __future__ import annotations

import pytest

from etf_portfolio import errors


def test_domain_errors_subclass_builtins_for_backward_compatibility() -> None:
    # Existing call sites catch ValueError/RuntimeError; domain errors must still match.
    assert issubclass(errors.ConfigError, ValueError)
    assert issubclass(errors.DataValidationError, ValueError)
    assert issubclass(errors.InfeasibleConstraintsError, ValueError)
    assert issubclass(errors.InsufficientHistoryError, ValueError)
    assert issubclass(errors.MLDisabledError, ValueError)
    assert issubclass(errors.ProvenanceError, RuntimeError)
    assert issubclass(errors.PipelineInterrupted, errors.EtfResearchError)


@pytest.mark.parametrize(
    ("exc", "expected_code"),
    [
        (errors.ConfigError("x"), errors.EXIT_CONFIG),
        (errors.MLDisabledError("x"), errors.EXIT_CONFIG),
        (errors.DataValidationError("x"), errors.EXIT_DATA),
        (errors.DataIngestionError("x"), errors.EXIT_DATA),
        (errors.InfeasibleConstraintsError("x"), errors.EXIT_INFEASIBLE),
        (errors.InsufficientHistoryError("x"), errors.EXIT_INFEASIBLE),
        (errors.ProvenanceError("x"), errors.EXIT_PROVENANCE),
        (errors.PipelineInterrupted("x"), errors.EXIT_INTERRUPTED),
        (KeyboardInterrupt(), errors.EXIT_INTERRUPTED),
        (RuntimeError("x"), errors.EXIT_UNEXPECTED),
    ],
)
def test_exit_code_for(exc: BaseException, expected_code: int) -> None:
    assert errors.exit_code_for(exc) == expected_code


def test_error_code_is_stable_and_machine_readable() -> None:
    assert errors.error_code_for(errors.DataValidationError("x")) == "data_validation_error"
    assert errors.error_code_for(errors.InfeasibleConstraintsError("x")) == "infeasible_constraints"
    assert errors.error_code_for(RuntimeError("x")) == "unexpected_error"


def test_validationerror_by_name_maps_to_config() -> None:
    # Pydantic ValidationError is matched by class name without importing pydantic.
    class ValidationError(Exception):
        pass

    assert errors.exit_code_for(ValidationError()) == errors.EXIT_CONFIG
    assert errors.error_code_for(ValidationError()) == "config_error"
