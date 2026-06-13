"""Tests for structured logging setup, handler hygiene, and stream re-targeting."""

from __future__ import annotations

import io
import logging
import sys

import pytest

from etf_portfolio.logging_config import configure_logging, get_logger, reset_logging


def _etf_handlers() -> list[logging.Handler]:
    root_logger = logging.getLogger()
    return [
        handler
        for handler in root_logger.handlers
        if getattr(handler, "_etf_structured_logging", False)
        or getattr(handler, "_etf_log_file", None) is not None
    ]


def test_reset_logging_removes_attached_handlers() -> None:
    configure_logging()
    assert _etf_handlers(), "configure_logging should attach a tagged stderr handler"

    reset_logging()
    assert _etf_handlers() == []


def test_stderr_handler_retargets_current_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """The handler must follow the live sys.stderr, never a stale captured stream."""

    configure_logging()
    logger = get_logger("test.retarget")

    first = io.StringIO()
    monkeypatch.setattr(sys, "stderr", first)
    logger.warning("first-message")
    first_value = first.getvalue()
    assert "first-message" in first_value

    # Swap sys.stderr (as pytest does per test) and close the previous stream.
    second = io.StringIO()
    monkeypatch.setattr(sys, "stderr", second)
    first.close()

    # Emitting now must go to the new stream and must not touch the closed one.
    logger.warning("second-message")
    second_value = second.getvalue()
    assert "second-message" in second_value
    assert "second-message" not in first_value  # the stale stream is untouched
