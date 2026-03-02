"""Logic for data caching with Joblib."""

import logging
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import overload

from joblib import Memory, expires_after

logger = logging.getLogger(__name__)
PATH_CACHE = Path(".cache")  # CWD: Not configurable as Memory is created at import time
memory = Memory(location=PATH_CACHE, verbose=50)


@overload
def cache[**P, R](func: Callable[P, R]) -> Callable[P, R]: ...


@overload
def cache[**P, R](**joblib_kwargs) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def cache[**P, R](
    func=None, _instance: Memory = memory, **joblib_kwargs
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Typed cache decorator version of joblib.memory to work with pydantic.

    Additionally this version safely allows to force-recompute and sets an expiry
    callback if not already set. The function to be cached requires the boolean keyword
    argument `force=False`.
    """

    def inner[**P, R](func: Callable[P, R]) -> Callable[P, R]:
        # Add callback for expiry
        joblib_kwargs.setdefault("cache_validation_callback", expires_after(hours=6))
        cached_func = _instance.cache(func, **joblib_kwargs)

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if kwargs.pop("force", False):
                # Clear cache on force
                name = f"{cached_func.func.__module__}.{cached_func.func.__qualname__}"
                logger.info(f"Purge cache of {name} and force recompute.")
                cached_func.clear(warn=False)
            return cached_func(*args, **kwargs)

        # Keep reference to MemorizedFunc
        setattr(wrapper, "__memorized_func__", cached_func)

        return wrapper

    if func is not None:  # pragma: no cover
        return inner(func)

    return inner
