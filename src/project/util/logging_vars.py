"""Allow attaching variables to log messages to give more context."""

import logging
import logging.config
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

__all__ = [
    "bind_logvars",
    "unbind_logvars",
    "bound_logvars",
]

_log_context = ContextVar("log_context", default={})


def bind_logvars(**values: Any) -> dict[str, Any]:
    """Bind one or more logging variables to be attached to all logs going forward."""
    ctx = _log_context.get().copy()
    values_backup = {k: ctx[k] for k in ctx.keys() & values.keys()}
    ctx.update(values)
    _log_context.set(ctx)
    return values_backup


def unbind_logvars(*keys: str) -> None:
    """Remove one or multiple logging variables by their names."""
    ctx = _log_context.get().copy()
    for k in keys:
        ctx.pop(k, None)
    _log_context.set(ctx)


def clear_logvars():
    """Clear all bound logging variables for testing."""
    _log_context.set({})


@contextmanager
def bound_logvars(**values: Any) -> Generator[None, None, None]:
    """Temporarily bind and unbind/restore logging variables using with/decorator."""
    backup = bind_logvars(**values)
    try:
        yield
    finally:
        unbind_logvars(*values)
        bind_logvars(**backup)


class LoggingVarsFilter(logging.Filter):
    """Filter to inject contextvar logging variables into log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.__dict__.update(_log_context.get())
        return True
