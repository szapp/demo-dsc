from typing import Annotated

import pandas as pd
from pandera.pandas import Field as F

from .base import DataModelBase, DataModelBaseML


class RawDataModel(DataModelBase):
    """Data model for raw data."""

    _CAT = ("Apple", "Banana", "Cherry", "Date")

    # Identifiers
    id: pd.Int64Dtype
    date: Annotated[pd.DatetimeTZDtype, "us", "UTC"]

    # Target
    target: pd.Float64Dtype = F(nullable=True)

    # Features
    col1: pd.Float32Dtype = F(ge=0, lt=8_000)
    col2: pd.UInt64Dtype = F(ge=0, nullable=True)
    col3: pd.BooleanDtype = F(nullable=True)
    col4: Annotated[pd.CategoricalDtype, _CAT, False] = F(nullable=True)

    class Config:
        unique = ["id", "date"]  # noqa: RUF012


class ProcessedDataModel(DataModelBaseML, RawDataModel):
    """Data model for processed and ML-conform data."""

    # Datatypes from raw data model are coerced to ML-conform types.
