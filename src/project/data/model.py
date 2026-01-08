from typing import Self

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Field as F
from pandera.typing import DataFrame
from pandera.typing import Series as S


class MLDataModelBase(pa.DataFrameModel):
    """Base data model coercing to ML compatible data types"""

    _pre_rename: dict[str, str] = dict()

    class Config:
        coerce = True
        strict = True

    @classmethod
    def validate(
        cls: type[Self], check_obj: pd.DataFrame, *args, **kwargs
    ) -> DataFrame[Self]:
        check_obj = check_obj.rename(columns=cls._pre_rename)
        check_obj[check_obj.isna()] = None  # Coerce NaN to False for booleans
        df = super().validate(check_obj, *args, **kwargs)

        # Make data types ML conform‚
        df = df.astype(pd.Series(int, df.select_dtypes("datetime").columns))
        df = df.astype(pd.Series(float, df.select_dtypes("integer").columns))

        return df


@pa.extensions.register_check_method
def is_int(p: pd.DataFrame | pd.Series) -> bool:
    """Register check for integer float"""
    return (p % 1 == 0) | p.isna()


class RawDataModel(MLDataModelBase):
    """Data model of the raw data"""

    date: S[pa.Timestamp]
    col1: S[pa.Float32] = F(ge=0, lt=8_000)
    col2: S[pa.Float32] = F(ge=0, is_int=True)
    col3: S[pa.Float64] = F(ge=0)
    target: S[pa.Float64] = F(nullable=True)
