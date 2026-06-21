import logging
from typing import Any, Self

import pandas as pd
import pandera.pandas as pa
import pandera.typing.pandas as pat
from pandera.pandas import Field as F


class DataModelBase(pa.DataFrameModel):
    """Data model base with standard config and column renaming prior to validation."""

    _pre_rename: dict[str, str] = dict()  # Rename selected columns
    index_: pat.Index[int] = F(unique=True, ge=0)  # DataFrame index

    class Config:
        strict = "filter"  # Exclude extra columns
        coerce = True  # Auto-convert data types where possible

    @pa.check("^.*[^_]$", regex=True, ignore_na=False)
    def has_at_least_one_value(cls, col: pat.Series[Any]) -> bool:
        """Columns with all NaNs suggest faulty data."""
        return col.notna().any() or col.empty

    @pa.check(index_, ignore_na=False)
    def index_is_monotonically_increasing(cls, idx: pat.Index[int]) -> bool:
        """Index must be monotonically increasing to reduce risk of mistakes."""
        cls.get_logger().debug("Check for proper index")
        return idx.tolist() == pd.RangeIndex(len(idx)).tolist()

    @pa.dataframe_check
    def dataframe_is_non_empty(cls, df: pd.DataFrame) -> bool:
        """Training or inference will fail with zero samples."""
        cls.get_logger().debug("Check for existing samples")
        return not df.empty

    @classmethod
    def validate(
        cls: type[Self], check_obj: pd.DataFrame, *args, **kwargs
    ) -> pat.DataFrame[pa.DataFrameModel]:
        cls.get_logger().debug("Adjust column names")
        check_obj = check_obj.rename(columns=cls._pre_rename)
        return super().validate(check_obj, *args, **kwargs)

    @classmethod
    def get_logger(cls) -> logging.Logger:
        return logging.getLogger(f"{cls.__module__}.{cls.__qualname__}")


class DataModelBaseML(DataModelBase):
    """Data model base enforcing ML conform data types after validation."""

    @classmethod
    def validate(
        cls: type[Self], check_obj: pd.DataFrame, *args, **kwargs
    ) -> pat.DataFrame[pa.DataFrameModel]:
        df = super().validate(check_obj, *args, **kwargs)

        # Allowed data types to provide a deterministic and reproducible ML context
        # - Float64
        # - Boolean
        # - Categorical
        # - Datetime

        # Normalize datetime columns to microsecond resolution and drop time zone info
        cls.get_logger().debug("Coerce datetime types")
        dt = df.select_dtypes(["datetime", "datetimetz"]).columns
        df[dt] = df[dt].apply(lambda x: x.dt.tz_localize(None).dt.as_unit("us"))

        # Missing values in booleans are not fully supported by Scikit-Learn
        cls.get_logger().debug("Impute boolean types")
        bt = df.select_dtypes("boolean").columns
        df[bt] = df[bt].fillna(False)  # NaN are set to False(!)

        # There should be no string columns, but all categorical
        cls.get_logger().debug("Coerce categorical types")
        st = df.select_dtypes(include="string").columns
        df = df.astype(pd.Series(pd.CategoricalDtype(), st))

        # All numerics are promoted to float64 to prevent downstream type conversions
        cls.get_logger().debug("Coerce numeric types")
        df = df.astype(pd.Series(pd.Float64Dtype(), df.select_dtypes("number").columns))

        return df
