import logging
import os
from collections.abc import Callable
from datetime import date, timedelta
from typing import cast

import mlflow
import pandas as pd
import sklearn
from mlflow.data.pandas_dataset import from_pandas
from mlflow.models import infer_signature
from pydantic import PastDate, PositiveInt
from sklearn.pipeline import Pipeline
from sklearn.utils import estimator_html_repr
from structlog.contextvars import bind_contextvars, unbind_contextvars

from ..config import make_cli, store
from ..types import SqlParams
from ..version import PACKAGE, SERVICE

logger = logging.getLogger(__name__)
YESTERDAY = (date.today() - timedelta(days=1)).isoformat()
RUN_NAME = "${hydra:job.name}-${hydra:job.config_name}_${now:%Y%m%d}_${now:%H%M%S}"


@sklearn.config_context(transform_output="pandas")
@store(
    name="prod",
    hydra_defaults=[
        {"dataloader": "prod"},
        {"dataprocessor": "prod"},
        {"model": "prod"},
        "_self_",
    ],
    run_name=RUN_NAME,
    register_model=f"{SERVICE}-prod",
)
@store(
    name="test",
    hydra_defaults=[
        {"dataloader": "prod"},
        {"dataprocessor": "prod"},
        {"model": "prod"},
        "_self_",
    ],
    run_name=RUN_NAME,
    register_model=f"{SERVICE}-test",
)
@store(
    name="dev",
    hydra_defaults=[
        {"hydra/job_logging/root/handlers": "console"},
        {"dataloader": "prod"},
        {"dataprocessor": "prod"},
        {"model": "prod"},
        "_self_",
    ],
    model={"verbose": True},
    run_name=RUN_NAME,
    register_model=f"{SERVICE}-dev",
    zen_meta={"hydra": {"verbose": [PACKAGE]}, "experiment": None},
)
def train(
    dataloader: Callable[[SqlParams], pd.DataFrame],
    dataprocessor: Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.Series]],
    model: Pipeline,
    training_cutoff: PastDate = cast(PastDate, YESTERDAY),
    num_samples: PositiveInt = 365 * 5,
    register_model: str | None = None,
    run_name: str | None = None,
    run_description: str | None = None,
) -> float:
    """Train a model.

    Args:
        dataloader: Callable to produce data given the SQL parameters.
        dataprocessor: Callable to process and split the data at the target column.
        model: Scikit-Learn ML pipeline.
        training_cutoff: End date of training data.
        num_samples: Number of days to include in training.
        register_model: Name of the MLflow model or None to skip.
        run_name: Name of the MLflow run.
        run_description: Description of the MLflow run.

    Returns:
        The score of the evaluated fit.
    """
    ENV = os.environ.get("ENV")

    start_date = training_cutoff - timedelta(days=num_samples - 1)
    sql_params = {"start_date": start_date, "end_date": training_cutoff}
    raw = dataloader(sql_params)
    X, y = dataprocessor(raw)

    logger.debug("Write raw data")
    raw.to_parquet("raw.parquet", index=False, compression="zstd")

    with mlflow.start_run(
        run_name=run_name, description=run_description, nested=True
    ) as run:
        bind_contextvars(run_id=run.info.run_id)
        logger.debug("Fit the model")
        model.fit(X, y)
        score = model.score(X, y)

        logger.debug("Log MLflow run")
        dataset = from_pandas(X)
        html = "<head><meta charset='UTF-8'></head>" + estimator_html_repr(model)
        mlflow.set_tags({"env": ENV, "training_cutoff": str(training_cutoff)})
        mlflow.log_text(html, "estimator.html")
        mlflow.log_artifacts(".hydra", "hydra")
        mlflow.log_input(dataset, context="Train")
        mlflow.log_params(model.get_params() | {"steps": None})
        mi = mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=X.head(),
            signature=infer_signature(X, y),
            registered_model_name=register_model,
            serialization_format="skops",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.compose._column_transformer.make_column_selector",
            ],
        )
        mlflow.log_metric("train_score", score, model_id=mi.model_id, dataset=dataset)
        unbind_contextvars("run_id")

    if register_model and mi.registered_model_version:
        model_version = mi.registered_model_version
        bind_contextvars(**{"model.name": register_model})
        bind_contextvars(**{"model.version": model_version})
        logger.info("Promote model to champion")
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(register_model, "champion", model_version)
        client.set_model_version_tag(
            register_model, model_version, "training_cutoff", str(training_cutoff)
        )
        unbind_contextvars("model.name", "model.version")

    if ENV == "dev":
        logger.debug("Write transformed data for debugging")
        y_transform = model[:-1].transform(X)
        y_transform.to_parquet("transformed.parquet", index=False, compression="zstd")

    return score


cli = make_cli(train)
