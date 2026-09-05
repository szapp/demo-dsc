from typing import Annotated

import pandas as pd
import pandera.pandas as pa
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
        unique = ("id", "date")


class ProcessedDataModel(RawDataModel, DataModelBaseML):
    """Data model for processed and ML-conform data."""

    # Datatypes from raw data model are coerced to ML-conform types.

    @pa.check("__disabled__", regex=True)
    def has_at_least_one_value(cls) -> None:
        """Disabled to avoid duplicated warnings after raw validation before."""
