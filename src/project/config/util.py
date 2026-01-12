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


def _make_columns(**columns: bool) -> list[str | int]:
    """Convert a boolean dictionary into a list for column selection"""
    return [k for k, v in columns.items() if v]


builds = make_custom_builds_fn(populate_full_signature=True, hydra_convert="object")
make_steps = partial(builds, _make_steps)
make_transformers = partial(builds, _make_transformers)
make_columns = partial(builds, _make_columns)
