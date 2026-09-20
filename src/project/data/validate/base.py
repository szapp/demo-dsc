import logging
import warnings
from typing import Any

import pandas as pd
import pandera.pandas as pa
import pandera.typing.pandas as pat
from pandera.pandas import Field as F

# Only show each distinct violation once
warnings.filterwarnings("once", r".*at_least_one_value", pa.errors.SchemaWarning)
warnings.filterwarnings("once", r".*non_zero_variance", pa.errors.SchemaWarning)


class DataModelBase(pa.DataFrameModel):
    """Data model base with standard config and basic validation."""

    index_: pat.Index[int] = F(unique=True, ge=0)  # Force generic DataFrame index

    class Config:
        strict = "filter"  # Drop extra columns
        coerce = True  # Auto-convert data types where possible

    @pa.check("^.*[^_]$", regex=True, ignore_na=False, raise_warning=True)
    def has_at_least_one_value(cls, col: pat.Series[Any]) -> bool:
        """Columns with all NaNs suggest faulty data."""
        return col.notna().any() or col.empty

    @pa.check("^.*[^_]$", regex=True, ignore_na=False, raise_warning=True)
    def has_non_zero_variance(cls, col: pat.Series[Any]) -> bool:
        """Columns with no variance suggest faulty data."""
        return col.nunique(dropna=False) > 1 or col.isna().all() or col.empty

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
    """Data model base enforcing ML compliant data types after validation.

    Allowed data types to provide a deterministic and reproducible ML context are
    float64, bool, category, datetime64[us].
    """

    @pa.dataframe_parser
    def sort_rows(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Sort rows by unique columns if present and/or reset index."""
        id_cols = getattr(cls.__config__, "unique", None) or []
        cls.get_logger().debug("Sort rows and reset index", extra={"columns": id_cols})
        return df.sort_values(id_cols).reset_index(drop=True)

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
        df[bt] = df[bt].fillna(False).astype("bool")  # NaN are set to False(!)

        # There should be no string columns, but all categorical
        st = df.select_dtypes("string").columns.tolist()
        cls.get_logger().debug("Coerce string types", extra={"columns": st})
        df[st] = df[st].astype("str").astype("category")

        # All numerics are promoted to float64 to prevent downstream type conversions
        nt = df.select_dtypes("number").columns.tolist()
        cls.get_logger().debug("Coerce numeric types", extra={"columns": nt})
        df[nt] = df[nt].astype("float64")

        # All categorical columns should have primitive underlying types
        ct = df.select_dtypes("category").columns.tolist()
        cls.get_logger().debug("Coerce categorical types", extra={"columns": ct})
        invalid_cat_types = {}
        for col in ct:
            cat = df[col].cat
            val = cat.categories
            # Repeat the coercion rules from above
            if pd.api.types.is_datetime64_any_dtype(val):
                df[col] = cat.rename_categories(val.tz_localize(None).as_unit("us"))
            elif pd.api.types.is_bool_dtype(val):
                df[col] = cat.rename_categories(val.astype("bool"))
            elif pd.api.types.is_string_dtype(val):
                df[col] = cat.rename_categories(val.astype("str"))  # Native string
            elif pd.api.types.is_numeric_dtype(val):
                df[col] = cat.rename_categories(val.astype("float64"))
            else:
                invalid_cat_types[col] = f"category[{val.dtype.name}]"

        # Check/report remaining data types
        ALLOWED_TYPES = {"bool", "category", "datetime64[us]", "float64"}
        dtypes = df.dtypes.astype(str)
        dtypes[invalid_cat_types.keys()] = list(invalid_cat_types.values())
        invalid_dtypes = set(dtypes) - ALLOWED_TYPES
        if invalid_dtypes:
            columns = dtypes[dtypes.isin(invalid_dtypes)].index.tolist()
            extra = {"columns": columns, "invalid_dtypes": list(invalid_dtypes)}
            cls.get_logger().error("Non-ML-compliant types encountered", extra=extra)
            return False

        return True
