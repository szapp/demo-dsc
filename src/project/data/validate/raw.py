from typing import Annotated, cast

import pandas as pd
import pandera.pandas as pa
import pandera.typing.pandas as pat
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

    @pa.dataframe_check
    def identifiers_are_unique(cls, df: pd.DataFrame) -> pat.Series[bool]:
        """The identifier columns have no duplicates."""
        cls.get_logger().debug("Check for duplicate identifiers")
        truth_values = ~df.duplicated([cls.id, cls.date], False)
        return cast(pat.Series[bool], truth_values)
