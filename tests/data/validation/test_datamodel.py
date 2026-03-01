import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Field as F
from pandera.typing.pandas import Series as S

from project.data.validation.base import DataModelBaseML


class DummyModel(DataModelBaseML):
    _pre_rename = {"misspelled": "intfloat_col"}
    float_col: S[pa.Float64]
    int_col: S[pa.Int]
    intfloat_col: S[pa.Float32] = F(is_int=True)
    date_col: S[pa.Timestamp]
    bool_col: S[pa.Bool]


def test_data_model_validation_coerces_datetime():
    """The validation method needs to reliably coerce to conform ML-tools"""
    dates = pd.date_range("2025-01-01", "2025-01-03")
    inputs = pd.DataFrame(
        {
            "float_col": [1.2, 0, 3.14],
            "int_col": [2, -3, 42],
            "misspelled": [2, -3.0, 42.0],
            "date_col": dates,
            "bool_col": [True, False, None],
        }
    )

    expected = pd.DataFrame(
        {
            "float_col": [1.2, 0.0, 3.14],
            "int_col": [2.0, -3.0, 42.0],
            "intfloat_col": [2.0, -3.0, 42.0],
            "date_col": dates.astype(int),
            "bool_col": [True, False, False],
        }
    ).astype(
        {
            "float_col": "float64",
            "int_col": "float64",
            "intfloat_col": "float32",
            "date_col": "float64",
            "bool_col": "bool",
        }
    )

    actual = DummyModel.validate(inputs)

    pd.testing.assert_frame_equal(actual, expected)
