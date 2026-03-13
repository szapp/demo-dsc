import logging
from collections.abc import Callable
from datetime import date, timedelta
from typing import cast

import pandas as pd
import sklearn

from ..config import make_cli, make_model, store
from ..types import FittedPipeline, SqlParams
from ..version import PACKAGE

logger = logging.getLogger(__name__)
TODAY = date.today().isoformat()
YESTERDAY = (date.today() - timedelta(days=1)).isoformat()


@sklearn.config_context(transform_output="pandas")
@store(
    name="prod",
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
    model=None,
    hydra_defaults=[
        "_self_",
        {"dataloader": "prod"},
        {"dataprocessor": "prod"},
        {"model": "prod"},
    ],
    zen_meta={"hydra": {"verbose": [PACKAGE]}},
)
def predict(
    dataloader: Callable[[SqlParams], pd.DataFrame],
    dataprocessor: Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.Series]],
    start_date: date = cast(date, TODAY),
    end_date: date = cast(date, YESTERDAY),
    model: FittedPipeline = make_model("prod"),
) -> pd.Series:
    """Predict with a model.

    Args:
        dataloader: Callable to produce data given the SQL parameters.
        dataprocessor: Callable to process and split the data at the target column.
        start_date: Starting date of data.
        end_date: End date of the data.
        model: A fitted Scikit-Learn ML pipeline.

    Returns:
        The predicted target column.
    """
    sql_params: SqlParams = {"start_date": start_date, "end_date": end_date}
    raw = dataloader(sql_params)
    X, _ = dataprocessor(raw)

    y_pred = model.predict(X)

    return y_pred


cli = make_cli(predict)
