"""Config store for data processors."""

from hydra_zen import store

from ... import data

__all__ = []  # No exports. Use the global store

dataprocessor = store(group="dataprocessor")

# Production data processor
dataprocessor(
    data.process_data,
    zen_partial=True,
    target_column="target",
    name="prod",
)
