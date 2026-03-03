from typing import Self

import pandas as pd
import pandera.pandas as pa
import pandera.typing.pandas as pat


class DataModelBase(pa.DataFrameModel):
    """Data model base with standard config and column renaming prior to validation."""

    _pre_rename: dict[str, str] = dict()  # Rename selected of columns

    class Config:
        strict = "filter"  # Exclude extra columns
        coerce = True  # Auto-convert data types where possible

    @classmethod
    def validate(  # type: ignore
        cls: type[Self], check_obj: pd.DataFrame, *args, **kwargs
    ) -> pat.DataFrame[Self]:
        # Rename columns prior to validation
        check_obj = check_obj.rename(columns=cls._pre_rename)

        # NaN to None to facilitate coercion of NaN to False for booleans
        check_obj[check_obj.isna()] = None

        df = super().validate(check_obj, *args, **kwargs)

        return df


class DataModelBaseML(DataModelBase):
    """Data model base enforcing ML conform data types after validation."""

    @classmethod
    def validate(  # type: ignore
        cls: type[Self], check_obj: pd.DataFrame, *args, **kwargs
    ) -> pat.DataFrame[Self]:
        df = super().validate(check_obj, *args, **kwargs)

        # Make data types ML conform‚
        df = df.astype(pd.Series(int, index=df.select_dtypes("datetime").columns))
        df = df.astype(pd.Series(float, index=df.select_dtypes("integer").columns))

        return df
