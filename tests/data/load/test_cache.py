from collections.abc import Callable
from unittest.mock import create_autospec

import pytest
from joblib.memory import FileSystemStoreBackend, Memory

from project.data.load import _cache


@pytest.fixture(scope="module", name="cached_func")
def _cached_func() -> Callable:
    """Create a cached function with a mocked memory backend."""
    backend = create_autospec(FileSystemStoreBackend, instance=True)
    mem = Memory(location=backend, verbose=0)

    @_cache.cache(_instance=mem, cache_validation_callback=None)
    def func(a: int = 1, force: bool = False) -> int:
        return 1

    # Initialize cache object on first call
    func()

    return func


def test_cache_is_recomputed_on_force(cached_func):
    """The computed cache is only cleared on force and available on the next call."""
    backend = cached_func.__memorized_func__.store_backend
    backend.reset_mock()

    cached_func()  # From cache
    cached_func()  # From cache
    cached_func(force=True)  # Clear cache
    cached_func()  # From cache

    assert backend.clear_path.call_count == 1
