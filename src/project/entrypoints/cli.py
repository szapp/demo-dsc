import logging
from collections.abc import Callable
from datetime import date, timedelta
from typing import cast

import mlflow
import pandas as pd
import sklearn
from dotenv import load_dotenv
from hydra_zen import zen
from hydra_zen.third_party.pydantic import pydantic_parser
from pydantic import PastDate
from sklearn.model_selection import BaseCrossValidator, KFold, cross_validate
from sklearn.pipeline import Pipeline

from ..config import CONFIG_PATH, InitWrapper, make_model, store
from ..data import process_data
from ..types import SqlParams

logger = logging.getLogger(__name__)
YESTERDAY = (date.today() - timedelta(days=1)).isoformat()


@sklearn.config_context(transform_output="pandas")
@store(
    name="train_prod",
    exp_name="train_prod",
    model=None,
    validator=None,
    hydra_defaults=["_self_", {"dataloader": "prod"}, {"model": "prod"}],
)
@store(
    name="train_test",
    exp_name="train_test",
    model=None,
    validator=None,
    hydra_defaults=["_self_", {"dataloader": "prod"}, {"model": "prod"}],
)
@store(
    name="train_dev",
    exp_name="train_dev",
    model=None,
    validator=None,
    hydra_defaults=["_self_", {"dataloader": "prod"}, {"model": "prod"}],
    zen_meta={"log_level": "DEBUG"},
)
def train(
    dataloader: Callable[[SqlParams], pd.DataFrame],
    start_date: PastDate = cast(PastDate, "2026-01-01"),
    end_date: PastDate = cast(PastDate, YESTERDAY),
    model: Pipeline = make_model("prod"),
    validator: BaseCrossValidator | None = KFold(),
    exp_name: str = "dev",
) -> float:
    """Train a model

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
        if validator:
            with mlflow.start_run(nested=True):  # type: ignore
                logger.info(f"Run validation with {validator!s}.")
                results = cross_validate(model, X, y, cv=validator)
                score = results["test_score"].mean()
        else:
            logger.info("Train on full dataset.")
            model.fit(X, y)
            score = model.score(X, y)

    logger.info(f"Score: {score}.")
    return score


def main():
    import optuna.logging  # noqa: F401 - Import here to suppress rogue logging

    load_dotenv()
    store.add_to_hydra_store()
    entrypoint = zen(train, instantiation_wrapper=InitWrapper(pydantic_parser))
    entrypoint.hydra_main(CONFIG_PATH, config_name="train_dev", version_base=None)
