import logging
import os
from datetime import date
from pathlib import Path

import sklearn
from sklearn.pipeline import Pipeline
from sqlalchemy import Engine

from ..data.load import fetch_data

logger = logging.getLogger(__name__)


@sklearn.config_context(transform_output="polars")
def train(model: Pipeline, db: Engine, dev: bool = True) -> float:
    """Train a model

    Args:
        model: Scikit-Learn ML pipeline.
        db: SQLAlchemy database engine.
        dev: Run in development environment.

    Returns:
        The score of the evaluated fit.
    """
    logger.info(f"Arguments: {dev=!s}")

    X, y = fetch_data(db, date.today())

    model.fit(X, y)

    score = model.score(X, y)
    logger.info(score)
    return score


def main():
    import optuna.logging  # noqa: F401 - Import early to suppress rogue logging
    from hydra_zen import store, zen

    from ..config import db, model, register_base_config

    config_path = str(Path("configs").resolve())
    dev = os.getenv("DSC_DEV", "true").lower() in {"1", "true"}

    register_base_config()
    store(train, name="train_cli", db=db, model=model, dev=dev)
    store.add_to_hydra_store()

    zen(train).hydra_main(config_path, config_name="train_cli", version_base=None)
