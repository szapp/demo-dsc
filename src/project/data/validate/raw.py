from typing import Annotated

import pandas as pd
from pandera.pandas import Field as F

from .base import DataModelBase


class RawDataModel(DataModelBase):
    """Data model for raw data."""

    _cat = ["Apple", "Banana", "Cherry", "Date"]

    id: pd.Int64Dtype
    date: Annotated[pd.DatetimeTZDtype, "us", "UTC"]
    col1: pd.Float32Dtype = F(ge=0, lt=8_000)
    col2: pd.UInt64Dtype = F(ge=0, nullable=True)
    col3: pd.BooleanDtype = F(nullable=True)
    col4: Annotated[pd.CategoricalDtype, _cat, False] = F(nullable=True)
    target: pd.Float64Dtype = F(nullable=True)
