import logging

import pytest
from inline_snapshot import snapshot

from project.util import logging_vars


@pytest.fixture(scope="function", autouse=True)
def clear_logvars():
    logging_vars.clear_logvars()


def test_bind_attaches_multiple_values() -> None:
    """Binding multiple values attaches all of them."""
    inputs = {"hello": "world", "foo": "bar"}
    expected = inputs.copy()

    logging_vars.bind_logvars(**inputs)
    actual = logging_vars._log_context.get()

    assert actual == expected


def test_bind_returns_previous_values() -> None:
    """Overwritten bound values need to be restorable."""
    previous = {"hello": "world", "foo": "bar"}
    inputs = {"hello": "space", "baz": 42}
    expected = snapshot({"hello": "world"})

    logging_vars.bind_logvars(**previous)
    actual = logging_vars.bind_logvars(**inputs)

    assert actual == expected


def test_unbind_removes_multiple_values() -> None:
    """It must be possible to remove multiple values at once."""
    previous = {"hello": "world", "foo": "bar", "baz": 42}
    remove = ["hello", "foo"]
    expected = snapshot({"baz": 42})

    logging_vars.bind_logvars(**previous)
    logging_vars.unbind_logvars(*remove)
    actual = logging_vars._log_context.get()

    assert actual == expected


def test_unbind_ignores_non_existent_values() -> None:
    """There should be no error if values were already unbound."""
    inputs = {"hello": "world", "foo": "bar", "baz": 42}
    expected = inputs.copy()

    logging_vars.bind_logvars(**inputs)
    logging_vars.unbind_logvars("non_exitent")
    actual = logging_vars._log_context.get()

    assert actual == expected


def test_bound_logvars_restores_values() -> None:
    """The context needs to utilize the other functions correctly."""
    previous = {"hello": "world", "foo": "bar"}
    inputs = {"hello": "space", "baz": 42}
    expected_during = snapshot({"hello": "space", "foo": "bar", "baz": 42})
    expected_after = previous.copy()

    logging_vars.bind_logvars(**previous)
    with logging_vars.bound_logvars(**inputs):
        actual_during = logging_vars._log_context.get()
    actual_after = logging_vars._log_context.get()

    assert actual_during == expected_during
    assert actual_after == expected_after


class TestLoggingVarsFilter:
    def test_filter_resolves_all_context_values(self) -> None:
        """The log record should be reliably populated."""
        inputs = {"hello": "world", "foo": "bar", "baz": 42}
        record = logging.makeLogRecord({})
        expected = {**record.__dict__, **inputs}

        logging_vars.bind_logvars(**inputs)
        actual_return = logging_vars.LoggingVarsFilter().filter(record)
        actual_record = record.__dict__

        assert actual_return == snapshot(True)
        assert actual_record == expected
