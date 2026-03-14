"""Logic for data caching with Joblib."""

import logging
import os
from collections.abc import Callable
from functools import wraps
from typing import Protocol, cast, overload

from joblib import Memory, expires_after
from joblib.memory import MemorizedFunc

ENV_VAR_CACHE_DIR = "JOBLIB_CACHE_DIR"
DEFAULT_CACHE_DIR = "~" + os.sep + ".cache"  # Joblib resolves the path automatically
PATH_CACHE_DIR = os.environ.get(ENV_VAR_CACHE_DIR) or DEFAULT_CACHE_DIR
memory = Memory(location=PATH_CACHE_DIR, verbose=0)
logger = logging.getLogger(__name__)


class CachedFunc[**P, R](Protocol):
    @property
    def cache(self) -> MemorizedFunc: ...
    def uncached(self, *args: P.args, **kwargs: P.kwargs) -> R: ...
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...
    def clear(self) -> None: ...


@overload
def cache[**P, R](func: Callable[P, R]) -> CachedFunc[P, R]: ...


@overload
def cache[**P, R](**joblib_kwargs) -> Callable[[Callable[P, R]], CachedFunc[P, R]]: ...


def cache[**P, R](
    func: Callable[P, R] | None = None, _instance: Memory = memory, **joblib_kwargs
) -> CachedFunc[P, R] | Callable[[Callable[P, R]], CachedFunc[P, R]]:
    """Typed cache decorator version of joblib.memory to work with pydantic."""

    def create_cache(func: Callable[P, R]) -> CachedFunc[P, R]:
        # Add callback for expiry
        joblib_kwargs.setdefault("cache_validation_callback", expires_after(hours=6))
        cached_func = _instance.cache(func, **joblib_kwargs)

        func_module = getattr(func, "__module__", "")
        func_name = getattr(func, "__qualname__", str(func))
        func_fullname = f"{func_module}.{func_name}".lstrip(".")

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            """Pydantic does not support objects, only functions, so wrap it."""
            logger.debug(f"Attempt to retrieve cache for {func_fullname}")
            return cached_func(*args, **kwargs)

        # Keep references accessible
        setattr(wrapper, "cache", cached_func)
        setattr(wrapper, "uncached", cached_func.func)
        setattr(wrapper, "clear", cached_func.clear)

        return cast(CachedFunc[P, R], wrapper)

    if func is not None:  # pragma: no cover
        return create_cache(func)

    return create_cache
