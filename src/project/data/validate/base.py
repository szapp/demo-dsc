import logging
from typing import Any

import pandas as pd
import pandera.pandas as pa
import pandera.typing.pandas as pat
from frozendict import frozendict
from pandera.pandas import Field as F


class DataModelBase(pa.DataFrameModel):
    """Data model base with standard config and column renaming prior to validation."""

    _pre_rename: frozendict[str, str] = frozendict()  # Rename selected columns
    index_: pat.Index[int] = F(unique=True, ge=0)  # DataFrame index

    class Config:
        strict = "filter"  # Drop extra columns
        coerce = True  # Auto-convert data types where possible

    @pa.dataframe_parser
    def rename_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns of input prior to validation."""
        cls.get_logger().debug("Adjust column names")
        return df.rename(columns=cls._pre_rename)

    @pa.check("^.*[^_]$", regex=True, ignore_na=False, raise_warning=True)
    def has_at_least_one_value(cls, col: pat.Series[Any]) -> bool:
        """Columns with all NaNs suggest faulty data."""
        return col.notna().any() or col.empty

    @pa.check(index_, ignore_na=False)
    def index_is_monotonically_increasing(cls, idx: pat.Index[int]) -> bool:
        """Index must be monotonically increasing to reduce risk of mistakes."""
        cls.get_logger().debug("Check for proper index")
        return idx.tolist() == list(range(len(idx)))

    @pa.dataframe_check
    def dataframe_is_non_empty(cls, df: pd.DataFrame) -> bool:
        """Training or inference will fail with zero samples."""
        cls.get_logger().debug("Check for existing samples")
        return not df.empty

    @classmethod
    def get_logger(cls) -> logging.Logger:
        return logging.getLogger(f"{cls.__module__}.{cls.__qualname__}")


class DataModelBaseML(DataModelBase):
    """Data model base enforcing ML conform data types after validation.

    Allowed data types to provide a deterministic and reproducible ML context are
    Float64, Boolean, Categorical, Datetime.
    """

    @pa.dataframe_check
    def coerce_data_types(cls, df: pd.DataFrame) -> bool:
        """Coerce data types in-place post validation."""

        # Normalize datetime columns to microsecond resolution and drop time zone info
        dt = df.select_dtypes(["datetime", "datetimetz"]).columns.tolist()
        cls.get_logger().debug("Coerce datetime types", extra={"columns": dt})
        df[dt] = df[dt].apply(lambda x: x.dt.tz_localize(None).dt.as_unit("us"))

        # Missing values in booleans are not fully supported by Scikit-Learn
        bt = df.select_dtypes("boolean").columns.tolist()
        cls.get_logger().debug("Impute boolean types", extra={"columns": bt})
        df[bt] = df[bt].fillna(False)  # NaN are set to False(!)

        # There should be no string columns, but all categorical
        st = df.select_dtypes("string").columns.tolist()
        cls.get_logger().debug("Coerce categorical types", extra={"columns": st})
        df[st] = df[st].astype(pd.CategoricalDtype())

        # All numerics are promoted to float64 to prevent downstream type conversions
        nt = df.select_dtypes("number").columns.tolist()
        cls.get_logger().debug("Coerce numeric types", extra={"columns": nt})
        df[nt] = df[nt].astype(pd.Float64Dtype())

        # Check remaining data types
        ALLOWED_DTYPES = {"boolean", "category", "datetime64[us]", "Float64"}
        invalid_dtypes = set(df.dtypes.astype(str)).difference(ALLOWED_DTYPES)
        if invalid_dtypes:
            columns = df.select_dtypes(invalid_dtypes).columns.tolist()
            extra = {"dtypes": list(invalid_dtypes), "columns": columns}
            cls.get_logger().error("Invalid dtypes", extra=extra)
            return False

        return True
