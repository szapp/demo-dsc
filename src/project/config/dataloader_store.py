from hydra_zen import store

from .. import data

dataloader_store = store(group="dataloader")

# Production data
dataloader_store(
    data.fetch_data,
    zen_partial=True,
    db_engine=None,
    hydra_defaults=[
        "_self_",
        {"/db@db_engine": "memory"},
    ],
    name="prod",
)
