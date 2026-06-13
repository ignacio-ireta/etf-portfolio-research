"""Domain-specific exception taxonomy and CLI exit codes.

Each domain error carries a stable machine-readable ``code`` and the process
``exit_code`` the CLI should return when it propagates uncaught. Domain errors
subclass the built-in they historically replaced (``ValueError`` /
``RuntimeError``) so existing ``pytest.raises`` call sites keep working while new
code can catch the precise domain type. See ``docs/engineering_standards.md`` §H.
"""

from __future__ import annotations

# Meaningful CLI exit codes (see docs/engineering_standards.md §H).
EXIT_OK = 0
EXIT_UNEXPECTED = 1
EXIT_CONFIG = 2
EXIT_DATA = 3
EXIT_INFEASIBLE = 4
EXIT_PROVENANCE = 5
EXIT_INTERRUPTED = 130


class EtfResearchError(Exception):
    """Base class for all domain errors raised by the pipeline."""

    code: str = "etf_research_error"
    exit_code: int = EXIT_UNEXPECTED


class ConfigError(EtfResearchError, ValueError):
    """Configuration is missing, malformed, or internally inconsistent."""

    code = "config_error"
    exit_code = EXIT_CONFIG


class DataIngestionError(EtfResearchError, ValueError):
    """Market data could not be fetched or assembled from a provider."""

    code = "data_ingestion_error"
    exit_code = EXIT_DATA


class DataValidationError(EtfResearchError, ValueError):
    """Ingested data failed a validation check (missing data, jumps, etc.)."""

    code = "data_validation_error"
    exit_code = EXIT_DATA


class InfeasibleConstraintsError(EtfResearchError, ValueError):
    """The configured optimization constraints admit no feasible portfolio."""

    code = "infeasible_constraints"
    exit_code = EXIT_INFEASIBLE


class InsufficientHistoryError(EtfResearchError, ValueError):
    """Not enough return history is available for the requested computation."""

    code = "insufficient_history"
    exit_code = EXIT_INFEASIBLE


class ProvenanceError(EtfResearchError, RuntimeError):
    """Run provenance requirements (e.g. a git commit) could not be satisfied."""

    code = "provenance_error"
    exit_code = EXIT_PROVENANCE


class MLDisabledError(EtfResearchError, ValueError):
    """An ML command was requested while ML is disabled in the config."""

    code = "ml_disabled"
    exit_code = EXIT_CONFIG


class PipelineInterrupted(EtfResearchError):
    """The pipeline was interrupted (e.g. SIGINT) and stopped between stages."""

    code = "pipeline_interrupted"
    exit_code = EXIT_INTERRUPTED


def exit_code_for(exc: BaseException) -> int:
    """Map an exception to the process exit code the CLI should return."""

    if isinstance(exc, EtfResearchError):
        return exc.exit_code
    if isinstance(exc, KeyboardInterrupt):
        return EXIT_INTERRUPTED
    # Pydantic ValidationError and other config-shaped failures are matched by
    # name to avoid importing pydantic here; the CLI handles the precise type.
    if type(exc).__name__ == "ValidationError":
        return EXIT_CONFIG
    return EXIT_UNEXPECTED


def error_code_for(exc: BaseException) -> str:
    """Return a stable machine-readable code for structured error reporting."""

    if isinstance(exc, EtfResearchError):
        return exc.code
    if isinstance(exc, KeyboardInterrupt):
        return PipelineInterrupted.code
    if type(exc).__name__ == "ValidationError":
        return ConfigError.code
    return "unexpected_error"


__all__ = [
    "EXIT_CONFIG",
    "EXIT_DATA",
    "EXIT_INFEASIBLE",
    "EXIT_INTERRUPTED",
    "EXIT_OK",
    "EXIT_PROVENANCE",
    "EXIT_UNEXPECTED",
    "ConfigError",
    "DataIngestionError",
    "DataValidationError",
    "EtfResearchError",
    "InfeasibleConstraintsError",
    "InsufficientHistoryError",
    "MLDisabledError",
    "PipelineInterrupted",
    "ProvenanceError",
    "error_code_for",
    "exit_code_for",
]
