"""Utility functions exclusively for use in the config stores."""

__all__ = [
    "builds",
    "build_columns",
    "build_steps",
    "build_transformers",
]

from functools import partial
from typing import Any, Literal

from hydra_zen import make_custom_builds_fn


def _make_steps(**steps: Any) -> list[tuple[str, Any]]:
    """Convert dict of scikit-learn pipeline steps to list of tuples"""
    return list(steps.items())


def _make_transformers(
    **columns: dict[Literal["transformer", "columns"], Any],
) -> list[tuple[str, Any, list[str | int] | str | int]]:
    """Convert dict of scikit-learn column transformers to list of tuples"""
    return [(name, v["transformer"], v["columns"]) for name, v in columns.items()]


def _make_columns(
    index: dict[int, bool] | None = None, **columns: bool
) -> list[str | int]:
    """Convert a boolean dictionary into a list for column selection"""
    if index is not None:
        # As column indices: build_columns(index={1: True, 2: False})
        selection = index
    else:
        # As column names: build_columns(col1=True, col2=False)
        selection = columns

    return [k for k, v in selection.items() if v]


builds = make_custom_builds_fn(populate_full_signature=True, hydra_convert="object")
build_steps = partial(builds, _make_steps)
build_transformers = partial(builds, _make_transformers)
build_columns = partial(builds, _make_columns)
