import logging
from pathlib import Path
from typing import cast

import mlflow
import sklearn
from dotenv import load_dotenv
from hydra_zen import zen
from hydra_zen.third_party.pydantic import pydantic_parser
from pydantic import PastDate
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sqlalchemy import Engine

from ..config import store
from ..data.load import fetch_data

logger = logging.getLogger(__name__)


@sklearn.config_context(transform_output="pandas")
@store(
    hydra_defaults=[
        "_self_",
        {"model": "prod"},
        {"db": "memory"},
    ],
    zen_meta={
        "log_level": "INFO",
    },
    name="train_cli",
)
def train(
    model: Pipeline,
    db: Engine,
    start_date: PastDate = cast(PastDate, "2026-01-01"),
    evaluate: bool = True,
) -> float:
    """Train a model

    Args:
        model: Scikit-Learn ML pipeline.
        db: SQLAlchemy database engine.
        evaluate: Perform K-fold cross-validation.

    Returns:
        The score of the evaluated fit.
    """
    logger.info(f"Arguments {start_date=!s}")
    mlflow.sklearn.autolog()

    X, y = fetch_data(db, start_date=start_date)

    if evaluate:
        with mlflow.start_run(nested=True):
            kf = KFold()
            results = cross_validate(model, X, y, cv=kf)
            score = results["test_score"].mean()
    else:
        model.fit(X, y)
        score = model.score(X, y)

    logger.info(score)
    return score


def main():
    import optuna.logging  # noqa: F401 - Import here to suppress rogue logging

    load_dotenv()
    store.add_to_hydra_store()
    config_path = str(Path("configs").resolve())  # Relative to working directory
    entrypoint = zen(train, instantiation_wrapper=pydantic_parser)
    entrypoint.hydra_main(config_path, config_name="train_cli", version_base=None)
