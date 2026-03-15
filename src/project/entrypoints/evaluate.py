import logging
from collections.abc import Callable
from datetime import date, timedelta
from typing import cast

import mlflow
import pandas as pd
import sklearn
from pydantic import PastDate
from sklearn.model_selection import BaseCrossValidator, KFold, cross_validate
from sklearn.pipeline import Pipeline

from ..config import make_cli, make_model, store
from ..types import SqlParams
from ..version import PACKAGE

logger = logging.getLogger(__name__)
YESTERDAY = (date.today() - timedelta(days=1)).isoformat()


@sklearn.config_context(transform_output="pandas")
@store(
    name="prod",
    exp_name="evaluate_prod",
    model=None,
    validator=None,
    hydra_defaults=[
        "_self_",
        {"dataloader": "prod"},
        {"dataprocessor": "prod"},
        {"model": "prod"},
        {"validator": "5fold"},
    ],
)
@store(
    name="test",
    exp_name="evaluate_test",
    model=None,
    validator=None,
    hydra_defaults=[
        "_self_",
        {"dataloader": "prod"},
        {"dataprocessor": "prod"},
        {"model": "prod"},
        {"validator": "5fold"},
    ],
)
@store(
    name="dev",
    exp_name="evaluate_dev",
    model=None,
    validator=None,
    hydra_defaults=[
        "_self_",
        {"dataloader": "prod"},
        {"dataprocessor": "prod"},
        {"model": "prod"},
        {"validator": "5fold"},
    ],
    zen_meta={"hydra": {"verbose": [PACKAGE]}},
)
def evaluate(
    dataloader: Callable[[SqlParams], pd.DataFrame],
    dataprocessor: Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.Series]],
    start_date: PastDate = cast(PastDate, "2026-01-01"),
    end_date: PastDate = cast(PastDate, YESTERDAY),
    model: Pipeline = make_model("prod"),
    exp_name: str = "dev",
    validator: BaseCrossValidator = KFold(),
) -> float:
    """Evaluate a model.

    Args:
        dataloader: Callable to produce data given the SQL parameters.
        dataprocessor: Callable to process and split the data at the target column.
        start_date: Starting date of data.
        end_date: End date of the data.
        model: Scikit-Learn ML pipeline.
        evaluate: Splitter for cross-validation.
        exp_name: Name of the MLflow experiment group.

    Returns:
        The score of the evaluated fit.
    """
    logger.debug("Start experiment")
    sql_params = {"start_date": start_date, "end_date": end_date}
    raw = dataloader(sql_params)
    X, y = dataprocessor(raw)

    with mlflow.start_run(nested=True):
        logger.info(f"Run validation with {validator!s}.")
        results = cross_validate(model, X, y, cv=validator)
        score = results["test_score"].mean()

    return score


cli = make_cli(evaluate)
