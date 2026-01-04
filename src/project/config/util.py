from functools import partial
from typing import Any, Literal

from hydra_zen import make_custom_builds_fn

builds = make_custom_builds_fn(populate_full_signature=True, hydra_convert="object")


def make_steps_(**steps: Any) -> list[tuple[str, Any]]:
    """Convert dict of scikit-learn pipeline steps to list of tuples"""
    return [(name, step) for name, step in steps.items()]


def make_columns_(
    **columns: dict[Literal["transformer", "columns"], Any],
) -> list[tuple[str, Any, list[str | int] | str | int]]:
    """Convert dict of scikit-learn column transformers to list of tuples"""
    return [(name, v["transformer"], v["columns"]) for name, v in columns.items()]


make_steps = partial(builds, make_steps_)
make_columns = partial(builds, make_columns_)
