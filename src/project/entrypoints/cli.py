import logging
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
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

from ..config import get_model, store

logger = logging.getLogger(__name__)
YESTERDAY = (date.today() - timedelta(days=1)).isoformat()


@sklearn.config_context(transform_output="pandas")
@store(
    hydra_defaults=[
        "_self_",
        {"dataloader": "prod"},
        {"model": "prod"},
        {"validator": None},
    ],
    model=None,
    validator=None,
    zen_meta={
        "log_level": "INFO",
    },
    name="train_cli",
)
def train(
    dataloader: Callable[[date, date], pd.DataFrame],
    start_date: PastDate = cast(PastDate, "2025-11-01"),
    end_date: PastDate = cast(PastDate, YESTERDAY),
    model: Pipeline = get_model("prod"),
    validator: BaseCrossValidator | None = KFold(),
) -> float:
    """Train a model

    Args:
        dataloader: Function to produce data given the dates.
        start_date: Starting date of data.
        end_date: End date of the data.
        model: Scikit-Learn ML pipeline.
        evaluate: Perform K-fold cross-validation.

    Returns:
        The score of the evaluated fit.
    """
    logger.info(f"Arguments: {start_date=!s}, {end_date=!s}")
    mlflow.sklearn.autolog()

    X, y = dataloader(start_date, end_date)

    if validator is None:
        logger.info("Train on full dataset.")
        model.fit(X, y)
        score = model.score(X, y)
    else:
        logger.info(f"Run validation with {validator!s}.")
        with mlflow.start_run(nested=True):  # type: ignore
            results = cross_validate(model, X, y, cv=validator)
            score = results["test_score"].mean()

    logger.info(f"Score: {score}.")
    return score


def main():
    import optuna.logging  # noqa: F401 - Import here to suppress rogue logging

    load_dotenv()
    store.add_to_hydra_store()
    config_path = str(Path("configs").resolve())  # Relative to working directory
    entrypoint = zen(train, instantiation_wrapper=pydantic_parser)
    entrypoint.hydra_main(config_path, config_name="train_cli", version_base=None)
