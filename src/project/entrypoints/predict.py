import logging
from collections.abc import Callable
from datetime import date, timedelta
from typing import cast

import mlflow
import pandas as pd
import sklearn

from ..config import make_cli, make_model, store
from ..data import process_data
from ..types import FittedPipeline, SqlParams

logger = logging.getLogger(__name__)
TODAY = date.today().isoformat()
YESTERDAY = (date.today() - timedelta(days=1)).isoformat()


@sklearn.config_context(transform_output="pandas")
@store(
    name="prod",
    exp_name="predict_prod",
    model=None,
    hydra_defaults=["_self_", {"dataloader": "prod"}, {"model": "prod"}],
)
@store(
    name="test",
    exp_name="predict_test",
    model=None,
    hydra_defaults=["_self_", {"dataloader": "prod"}, {"model": "prod"}],
)
@store(
    name="dev",
    exp_name="predict_dev",
    model=None,
    hydra_defaults=["_self_", {"dataloader": "prod"}, {"model": "fitted"}],
    zen_meta={"log_level": "DEBUG"},
)
def predict(
    dataloader: Callable[[SqlParams], pd.DataFrame],
    start_date: date = cast(date, TODAY),
    end_date: date = cast(date, YESTERDAY),
    model: FittedPipeline = make_model("prod"),
    exp_name: str = "dev",
) -> pd.Series:
    """Predict with a model.

    Args:
        dataloader: Function to produce data given the dates.
        start_date: Starting date of data.
        end_date: End date of the data.
        model: A fitted Scikit-Learn ML pipeline.
        exp_name: Name of the MLflow experiment group.

    Returns:
        The predicted target column.
    """
    sql_params = {"start_date": start_date, "end_date": end_date}
    raw = dataloader(sql_params)
    X, _ = process_data(raw, "target")

    with mlflow.start_run(run_name=exp_name):  # type: ignore
        y_pred = model.predict(X)

    return y_pred


cli = make_cli(predict)
