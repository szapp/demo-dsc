import logging

import pytest
from dirty_equals import Contains
from hydra_zen.third_party.pydantic import pydantic_parser
from inline_snapshot import snapshot

from project.config import InitWrapper


def raise_error(inputs: int = 0):
    raise ValueError("error")


def raise_user_interrupt():
    raise KeyboardInterrupt()


class TestInitWrapper:
    def test_exception_logger_catches_exception_and_exits(
        self, caplog: pytest.LogCaptureFixture
    ):
        """The core goal of the exception logger is to exit and log the exception."""
        wrp = InitWrapper()
        wrapped = wrp(raise_error)

        expected = [(wrp.__module__, logging.CRITICAL, "error")]

        with pytest.raises(SystemExit) as exit_status:
            with caplog.at_level(logging.INFO):
                wrapped()

        assert caplog.record_tuples == expected
        assert exit_status.value.code == snapshot(1)

    def test_exception_logger_reacts_to_user_interrupt(
        serlf, caplog: pytest.LogCaptureFixture
    ):
        """Aborting creates no stack trace and terminates with correct exit code."""
        wrp = InitWrapper()
        wrapped = wrp(raise_user_interrupt)

        expected = [(wrp.__module__, logging.ERROR, "Interrupted by user.")]

        with pytest.raises(SystemExit) as exit_status:
            with caplog.at_level(logging.INFO):
                wrapped()

        assert caplog.record_tuples == expected
        assert exit_status.value.code == snapshot(130)

    def test_exception_logger_captures_second_wrapper_exceptions(
        self, caplog: pytest.LogCaptureFixture
    ):
        """Any secondary wrapper may raise exceptions, too."""
        wrp = InitWrapper(pydantic_parser)
        wrapped = wrp(raise_error)
        expected = [(wrp.__module__, logging.CRITICAL, Contains("pydantic"))]

        with pytest.raises(SystemExit):
            with caplog.at_level(logging.INFO):
                wrapped("not_an_integer")

        assert caplog.record_tuples == expected
