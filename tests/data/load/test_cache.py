import logging
from pathlib import Path

import pytest
from inline_snapshot import snapshot
from joblib.memory import Memory

from project.data.load._cache import CachedFunc, cache


@pytest.fixture(scope="function", name="cached_func")
def _cached_func(tmp_path: Path) -> CachedFunc:
    """Create a cached function with a mocked memory backend."""
    location = tmp_path / ".cache"
    mem = Memory(location=location, verbose=0)

    @cache(_instance=mem, ignore=["note"], cache_validation_callback=None)
    def func(a: int = 1, note: str = "") -> None:
        logging.info(note)

    return func


def test_cache_is_recomputed_on_force(
    cached_func: CachedFunc, caplog: pytest.LogCaptureFixture
):
    """The computed cache is only cleared on force and available on the next call."""

    with caplog.at_level(logging.INFO):
        cached_func(note="1")  # Start
        cached_func(note="2")
        cached_func.clear()  # Reset
        cached_func(note="3")
        cached_func.uncached(note="4")  # No reset
        cached_func(note="5")
        cached_func(2, note="6")  # Add cache
        cached_func(note="7")
        cached_func(2, note="8")
        cached_func(note="9")

    assert caplog.messages == snapshot(["1", "3", "4", "6"])
