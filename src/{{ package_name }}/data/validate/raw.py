import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Field as F
from pandera.typing.pandas import Series as S

from .base import DataModelBase


class RawDataModel(DataModelBase):
    """Data model for raw data."""

    id: S[pa.Int64]
    date: S[pa.Timestamp]
    col1: S[pa.Float32] = F(ge=0, lt=8_000, nullable=True)
    col2: S[pd.Int64Dtype] = F(ge=0, nullable=True)
    col3: S[pa.Float64] = F(ge=0, nullable=True)
    target: S[pa.Float64] = F(nullable=True)
