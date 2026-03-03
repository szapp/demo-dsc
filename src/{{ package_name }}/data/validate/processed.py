import pandera.pandas as pa
from pandera.pandas import Field as F
from pandera.typing.pandas import Series as S

from .base import DataModelBaseML


class ProcessedDataModel(DataModelBaseML):
    """Data model for processed and ML-conform data ."""

    id: S[pa.Float32] = F(is_int=True)
    date: S[pa.Timestamp]
    col1: S[pa.Float32] = F(ge=0, lt=8_000)
    col2: S[pa.Float32] = F(ge=0, is_int=True)
    col3: S[pa.Float64] = F(ge=0)
    target: S[pa.Float64] = F(nullable=True)
