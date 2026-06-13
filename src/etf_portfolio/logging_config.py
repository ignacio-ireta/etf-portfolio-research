"""Structured logging helpers for the research pipeline."""

from __future__ import annotations

import json
import logging
import math
import numbers
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_RESERVED_LOG_RECORD_KEYS = set(logging.makeLogRecord({}).__dict__.keys()) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON payloads."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp_utc": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key in _RESERVED_LOG_RECORD_KEYS or key.startswith("_"):
                continue
            payload[key] = _normalize_log_value(value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, sort_keys=True, allow_nan=False)


def configure_logging(level: int = logging.INFO, *, log_file: Path | str | None = None) -> None:
    """Configure process-wide structured logging.

    Idempotent for the stderr stream handler (added once) and for any given
    ``log_file`` path (added once). Re-invoking updates the root level, so
    ``--log-level`` takes effect even after an earlier call.
    """

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if not any(
        getattr(handler, "_etf_structured_logging", False) for handler in root_logger.handlers
    ):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        handler._etf_structured_logging = True  # type: ignore[attr-defined]
        root_logger.addHandler(handler)

    if log_file is not None:
        log_path = Path(log_file)
        already_attached = any(
            getattr(handler, "_etf_log_file", None) == str(log_path)
            for handler in root_logger.handlers
        )
        if not already_attached:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(JsonFormatter())
            file_handler._etf_log_file = str(log_path)  # type: ignore[attr-defined]
            root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module."""

    return logging.getLogger(name)


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    /,
    **fields: Any,
) -> None:
    """Emit a structured log event with normalized extra fields."""

    normalized_fields = {key: _normalize_log_value(value) for key, value in fields.items()}
    logger.log(level, event, extra={"event": event, **normalized_fields})


def _normalize_log_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "isoformat") and callable(value.isoformat):
        try:
            return value.isoformat()
        except TypeError:
            pass
    if isinstance(value, dict):
        return {str(key): _normalize_log_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_log_value(item) for item in value]
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        scalar = float(value)
        return scalar if math.isfinite(scalar) else None
    return value


__all__ = ["JsonFormatter", "configure_logging", "get_logger", "log_event"]
