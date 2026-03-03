import logging
from collections.abc import Callable
from datetime import date, timedelta
from typing import cast

import mlflow
import pandas as pd
import sklearn
from pydantic import PastDate
from sklearn.pipeline import Pipeline

from ..config import make_cli, make_model, store
from ..data import process_data
from ..types import SqlParams

logger = logging.getLogger(__name__)
YESTERDAY = (date.today() - timedelta(days=1)).isoformat()


@sklearn.config_context(transform_output="pandas")
@store(
    name="prod",
    exp_name="train_prod",
    model=None,
    hydra_defaults=["_self_", {"dataloader": "prod"}, {"model": "prod"}],
)
@store(
    name="test",
    exp_name="train_test",
    model=None,
    hydra_defaults=["_self_", {"dataloader": "prod"}, {"model": "prod"}],
)
@store(
    name="dev",
    exp_name="train_dev",
    model=None,
    hydra_defaults=["_self_", {"dataloader": "prod"}, {"model": "prod"}],
    zen_meta={"log_level": "DEBUG"},
)
def train(
    dataloader: Callable[[SqlParams], pd.DataFrame],
    start_date: PastDate = cast(PastDate, "2026-01-01"),
    end_date: PastDate = cast(PastDate, YESTERDAY),
    model: Pipeline = make_model("prod"),
    exp_name: str = "dev",
) -> Pipeline:
    """Train a model.

    Args:
        dataloader: Function to produce data given the dates.
        start_date: Starting date of data.
        end_date: End date of the data.
        model: Scikit-Learn ML pipeline.
        evaluate: Perform K-fold cross-validation.
        exp_name: Name of the MLflow experiment group.

    Returns:
        The score of the evaluated fit.
    """
    logger.info(f"Arguments: {start_date=!s}, {end_date=!s}")
    mlflow.sklearn.autolog()

    sql_params = {"start_date": start_date, "end_date": end_date}
    raw = dataloader(sql_params)
    X, y = process_data(raw, "target")

    with mlflow.start_run(run_name=exp_name):  # type: ignore
        logger.info("Train on full dataset.")
        model.fit(X, y)

    return model


cli = make_cli(train)
