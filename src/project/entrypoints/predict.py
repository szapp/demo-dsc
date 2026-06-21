import logging
from collections.abc import Callable
from datetime import date, timedelta
from typing import cast

import numpy as np
import pandas as pd
import sklearn
from pydantic import PositiveInt

from ..config import make_cli, store
from ..types import FittedPipeline, SqlParams
from ..version import PACKAGE, SERVICE

logger = logging.getLogger(__name__)
TODAY = date.today().isoformat()


@sklearn.config_context(transform_output="pandas")
@store(
    name="prod",
    hydra_defaults=[
        {"dataloader": "prod"},
        {"dataprocessor": "prod"},
        {"model": "champion"},
        "_self_",
    ],
)
@store(
    name="test",
    hydra_defaults=[
        {"dataloader": "prod"},
        {"dataprocessor": "prod"},
        {"model": "champion"},
        "_self_",
    ],
)
@store(
    name="dev",
    hydra_defaults=[
        {"hydra/job_logging/root/handlers": "console"},
        {"dataloader": "prod"},
        {"dataprocessor": "prod"},
        {"model": "champion"},
        "_self_",
    ],
    model={"model_uri": f"models:/{SERVICE}-dev@champion"},
    zen_meta={"hydra": {"verbose": [PACKAGE]}},
)
def predict(
    dataloader: Callable[[SqlParams], pd.DataFrame],
    dataprocessor: Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.Series]],
    model: FittedPipeline,
    start_date: date = cast(date, TODAY),
    num_samples: PositiveInt = 60,
) -> np.ndarray:
    """Predict with a model.

    Args:
        dataloader: Callable to produce data given the SQL parameters.
        dataprocessor: Callable to process and split the data at the target column.
        model: A fitted Scikit-Learn ML pipeline.
        start_date: Starting date of data.
        end_date: End date of the data.

    Returns:
        The predicted target column.
    """
    end_date = start_date + timedelta(days=num_samples - 1)
    sql_params = {"start_date": start_date, "end_date": end_date}
    raw = dataloader(sql_params)
    X, _ = dataprocessor(raw)

    logger.debug("Write raw data")
    raw.to_parquet("raw.parquet", index=False, compression="zstd")

    logger.debug("Perform inference")
    y_pred = model.predict(X)

    return y_pred


cli = make_cli(predict)
