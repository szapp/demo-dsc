"""Config store for data loaders (data sources)."""

from hydra_zen import store

from ... import data
from .util import builds

__all__ = []  # No exports. Use the global store

dataloader_store = store(group="dataloader")

# Production data
dataloader_store(
    data.fetch_data,
    zen_partial=True,
    db_engine=None,
    sql_queries=builds(data.load_sql_files),  # Lazy loading
    hydra_defaults=[
        "_self_",
        {"/db@db_engine": "memory"},
    ],
    name="prod",
)
