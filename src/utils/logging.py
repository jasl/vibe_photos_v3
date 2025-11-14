"""Central logging utilities used across the Phase 1 stack."""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from .config import load_settings

_logging_configured = False
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


def get_correlation_id() -> str:
    """Return the active correlation ID, generating one if absent."""
    current = _correlation_id.get()
    if current:
        return current

    new_id = uuid4().hex
    _correlation_id.set(new_id)
    return new_id


def bind_correlation_id(value: str) -> None:
    """Force a specific correlation ID for the current context."""
    _correlation_id.set(value)


def _ensure_logging_configured() -> None:
    """Initialize Python's logging only once."""
    global _logging_configured  # noqa: PLW0603

    if _logging_configured:
        return

    settings = load_settings().data
    logging_config: Dict[str, Any] = settings.get("logging", {})

    log_dir = Path(logging_config.get("directory", "log")).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / logging_config.get("file", "phase1.log")
    log_format = logging_config.get(
        "format",
        "[%(asctime)s] %(levelname)s %(name)s %(message)s correlation_id=%(correlation_id)s",
    )
    level = logging_config.get("level", "INFO")

    rotation = logging_config.get("rotation", {})
    max_bytes = int(rotation.get("max_bytes", 10_485_760))
    backup_count = int(rotation.get("backup_count", 5))

    handlers = [
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count),
    ]

    logging.basicConfig(level=level, format=log_format, handlers=handlers)
    _logging_configured = True


class CorrelationIdAdapter(logging.LoggerAdapter):
    """Logger adapter that injects correlation IDs into log records."""

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple[Any, Dict[str, Any]]:
        correlation_id = kwargs.pop("correlation_id", None) or get_correlation_id()
        extra = kwargs.get("extra") or {}
        extra.setdefault("correlation_id", correlation_id)
        kwargs["extra"] = extra
        return msg, kwargs


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured with correlation ID support."""
    _ensure_logging_configured()
    base_logger = logging.getLogger(name)
    return CorrelationIdAdapter(base_logger, {})
