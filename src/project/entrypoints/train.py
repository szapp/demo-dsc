import logging
import os
from collections.abc import Callable
from datetime import date, timedelta
from typing import cast

import mlflow
import pandas as pd
import sklearn
from mlflow.data.pandas_dataset import from_pandas
from pydantic import PastDate
from sklearn.pipeline import Pipeline

from ..config import make_cli, make_model, store
from ..types import SqlParams
from ..version import PACKAGE

logger = logging.getLogger(__name__)
YESTERDAY = (date.today() - timedelta(days=1)).isoformat()


@sklearn.config_context(transform_output="pandas")
@store(
    name="prod",
    exp_name="train_prod",
    model=None,
    hydra_defaults=[
        "_self_",
        {"dataloader": "prod"},
        {"dataprocessor": "prod"},
        {"model": "prod"},
    ],
)
@store(
    name="test",
    exp_name="train_test",
    model=None,
    hydra_defaults=[
        "_self_",
        {"dataloader": "prod"},
        {"dataprocessor": "prod"},
        {"model": "prod"},
    ],
)
@store(
    name="dev",
    exp_name="train_dev",
    model=None,
    hydra_defaults=[
        "_self_",
        {"dataloader": "prod"},
        {"dataprocessor": "prod"},
        {"model": "prod"},
    ],
    zen_meta={"hydra": {"verbose": [PACKAGE]}},
)
def train(
    dataloader: Callable[[SqlParams], pd.DataFrame],
    dataprocessor: Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.Series]],
    start_date: PastDate = cast(PastDate, "2026-01-01"),
    end_date: PastDate = cast(PastDate, YESTERDAY),
    model: Pipeline = make_model("prod"),
    exp_name: str = "dev",
) -> Pipeline:
    """Train a model.

    Args:
        dataloader: Callable to produce data given the SQL parameters.
        dataprocessor: Callable to process and split the data at the target column.
        start_date: Starting date of data.
        end_date: End date of the data.
        model: Scikit-Learn ML pipeline.
        evaluate: Perform K-fold cross-validation.
        exp_name: Name of the MLflow experiment group.

    Returns:
        The score of the evaluated fit.
    """
    mlflow.sklearn.autolog(serialization_format="skops")

    sql_params: SqlParams = {"start_date": start_date, "end_date": end_date}
    raw = dataloader(sql_params)
    X, y = dataprocessor(raw)

    # Store raw data
    raw.to_parquet("raw.parquet", index=False, compression="zstd")
    dataset = from_pandas(raw, source="raw.parquet", targets=str(y.name))

    with mlflow.start_run(tags={"env": os.environ.get("ENV")}):
        mlflow.log_input(dataset, context="raw")
        mlflow.log_artifacts(".hydra", "hydra")
        model.fit(X, y)

    if os.environ.get("ENV") in {"test", "prod"}:
        model_uri = getattr(mlflow.last_logged_model(), "model_uri")
        tags = {"env": os.environ.get("ENV"), "date": end_date}
        mlflow.register_model(model_uri, PACKAGE, tags=tags)

    return model


cli = make_cli(train)
