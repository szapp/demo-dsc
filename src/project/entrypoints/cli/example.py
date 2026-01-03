import logging
import os
from functools import partial
from pathlib import Path
from typing import Any, Literal

import polars as pl
import sklearn
from hydra_zen import make_custom_builds_fn, store, zen
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import URL, Engine, create_engine

logger = logging.getLogger(__name__)
CONFIG_PATH = str(Path("configs").resolve())


@sklearn.config_context(transform_output="polars")
def task(db: Engine, model: Pipeline, dev: bool = True) -> None:
    print(db)
    print(model)
    print(dev)

    X = pl.DataFrame(
        {
            "col1": [1, 2],
            "col2": [2, 4],
            "col3": [3, 2],
        }
    )
    y = pl.Series([1, 2])

    model.fit(X, y)
    print(model.score(X, y))


builds = make_custom_builds_fn(populate_full_signature=True, hydra_convert="object")

db = builds(create_engine, url=builds(URL.create, query=dict()), pool_recycle=1800)


def make_steps_(**steps: Any) -> list[tuple[str, Any]]:
    return [(name, step) for name, step in steps.items()]


def make_columns_(
    **columns: dict[Literal["transformer", "columns"], Any],
) -> list[tuple[str, Any, list[str | int] | str | int]]:
    return [(name, v["transformer"], v["columns"]) for name, v in columns.items()]


make_steps = partial(builds, make_steps_)
make_columns = partial(builds, make_columns_)


preprocessing = builds(StandardScaler, with_mean=False)
features = builds(
    ColumnTransformer,
    transformers=make_columns(
        example=dict(
            transformer=builds(StandardScaler, with_std=False),
            columns=["col1", "col2"],
        ),
    ),
    verbose_feature_names_out=False,
)
regressor = builds(RandomForestRegressor, random_state=42)
model = builds(
    Pipeline,
    steps=make_steps(
        preprocessing=preprocessing,
        features=features,
        regressor=regressor,
    ),
)


def main():
    dev = os.getenv("DSC_DEV", "true").lower() in {"1", "true"}
    store(task, name="my_app", db=db, model=model, dev=dev)
    store.add_to_hydra_store()
    zen(task).hydra_main(
        config_name="my_app", config_path=CONFIG_PATH, version_base=None
    )
