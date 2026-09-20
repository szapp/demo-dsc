import logging
import re
from typing import Annotated

import pandas as pd
import pytest
from dirty_equals import IsStr
from pandera.errors import SchemaError, SchemaWarning
from pandera.pandas import Field as F

from project.data.validate.base import DataModelBase, DataModelBaseML


class DummyModel(DataModelBase):
    float_col: pd.Float64Dtype
    float32_col: pd.Float32Dtype
    int_col: pd.Int64Dtype = F(nullable=True)
    date_col: Annotated[pd.DatetimeTZDtype, "us", "UTC"]
    bool_col: pd.BooleanDtype = F(nullable=True)


@pytest.fixture(scope="module", name="df_dummy_base")
def _df_dummy_base() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "float_col": [1.2, 0, 3.14],
            "float32_col": [2, -3.0, 42],
            "int_col": [2.0, -3.0, None],
            "date_col": pd.date_range("2025-01-01", "2025-01-03").as_unit("s"),
            "bool_col": [True, False, None],
        }
    )


class TestDataModelBase:
    def test_validate_drops_extra_columns_silently(self, df_dummy_base: pd.DataFrame):
        """Dropping columns is a great way to curate the data to a desired schema."""
        inputs = df_dummy_base.assign(extra_col=[0, 1, 2])
        expected = ["float_col", "float32_col", "int_col", "date_col", "bool_col"]
        actual = DummyModel.validate(inputs).columns.tolist()
        assert actual == expected

    def test_validate_coerces_datatypes_correctly(self, df_dummy_base: pd.DataFrame):
        """The datatypes are coerced correctly including the datetime type."""
        date_col_ts = df_dummy_base["date_col"].dt.as_unit("us").dt.tz_localize("UTC")
        expected = df_dummy_base.assign(date_col=date_col_ts).astype(
            {
                "float_col": pd.Float64Dtype(),
                "float32_col": pd.Float32Dtype(),
                "int_col": pd.Int64Dtype(),
                "bool_col": pd.BooleanDtype(),
            }
        )
        actual = DummyModel.validate(df_dummy_base)
        pd.testing.assert_frame_equal(actual, expected)

    def test_validate_adjusts_time_zone_to_utc(self, df_dummy_base: pd.DataFrame):
        """Existing time zones are converted to UTC to be unambiguous."""
        date_col_ber = df_dummy_base["date_col"].dt.tz_localize("Europe/Berlin")
        inputs = df_dummy_base.assign(date_col=date_col_ber)
        expected = date_col_ber.dt.tz_convert("UTC").dt.as_unit("us")
        actual = DummyModel.validate(inputs)
        pd.testing.assert_series_equal(actual["date_col"], expected)

    def test_validate_raises_on_empty_dataframe(self, df_dummy_base: pd.DataFrame):
        """Data with no samples (rows) should be caught at validation time."""
        inputs = pd.DataFrame(columns=df_dummy_base.columns)
        with pytest.raises(SchemaError, match="non_empty"):
            DummyModel.validate(inputs)

    @pytest.mark.filterwarnings(
        "all:int_col.*at_least_one_value:pandera.errors.SchemaWarning"
    )
    def test_validate_raises_on_nan_columns(self, df_dummy_base: pd.DataFrame):
        """Data with nan-columns suggest faulty data sources."""
        inputs = df_dummy_base.assign(int_col=pd.NA)
        with pytest.warns(SchemaWarning, match="int_col.*at_least_one_value"):
            DummyModel.validate(inputs)

    @pytest.mark.filterwarnings(
        "all:int_col.*non_zero_variance:pandera.errors.SchemaWarning"
    )
    def test_validate_raises_on_zero_variance(self, df_dummy_base: pd.DataFrame):
        """Data with no variance suggest flat data sources."""
        inputs = df_dummy_base.assign(int_col=1)
        with pytest.warns(SchemaWarning, match="int_col.*non_zero_variance"):
            DummyModel.validate(inputs)

    def test_validate_raises_unordered_index(self, df_dummy_base: pd.DataFrame):
        """A unordered DataFrame index risks mistakes during ML pipeline joins."""
        inputs = df_dummy_base.iloc[[2, 0, 1]]
        with pytest.raises(SchemaError, match="monotonically_increasing"):
            DummyModel.validate(inputs)


class TestDataModelBaseML:
    def test_validate_drops_time_zone(self):
        """MLflow does not support time zone information."""

        class DummyModelML(DataModelBaseML):
            col: Annotated[pd.DatetimeTZDtype, "us", "UTC"]

        date_before = pd.date_range("2025-04-02T03:00:00+02:00", periods=2, unit="us")
        date_after = pd.date_range("2025-04-02T01:00:00", periods=2, unit="us")
        inputs = pd.DataFrame({"col": date_before})
        expected = pd.DataFrame({"col": date_after})
        actual = DummyModelML.validate(inputs)
        pd.testing.assert_frame_equal(actual, expected)

    def test_validate_keeps_time_zone_agnostic_datetime_as_is(self):
        """MLflow does not support time zone information."""

        class DummyModelML(DataModelBaseML):
            col: pd.Timestamp

        date_before = pd.date_range("2025-04-02", periods=2, unit="ns")
        date_after = pd.date_range("2025-04-02", periods=2, unit="us")
        inputs = pd.DataFrame({"col": date_before})
        expected = pd.DataFrame({"col": date_after})
        actual = DummyModelML.validate(inputs)
        pd.testing.assert_frame_equal(actual, expected)

    def test_validate_coerces_datetime_resolution(self):
        """From pandas 3.0 the default datetime resolution is microseconds."""

        class DummyModelML(DataModelBaseML):
            col: Annotated[pd.DatetimeTZDtype, "us", "UTC"]

        date_before = pd.date_range("2025-04-02", periods=2, unit="ns")
        date_after = pd.date_range("2025-04-02", periods=2, unit="us")
        inputs = pd.DataFrame({"col": date_before})
        expected = pd.DataFrame({"col": date_after})
        actual = DummyModelML.validate(inputs)
        pd.testing.assert_frame_equal(actual, expected)

    def test_validate_imputes_boolean_nan_to_false(self):
        """Missing values in booleans are not fully supported by Scikit-Learn."""

        class DummyModelML(DataModelBaseML):
            col: pd.BooleanDtype = F(nullable=True)

        inputs = pd.DataFrame({"col": [True, False, None]})
        expected = pd.DataFrame({"col": [True, False, False]}, dtype=bool)
        actual = DummyModelML.validate(inputs)
        pd.testing.assert_frame_equal(actual, expected)

    def test_validate_coerces_string_columns(self):
        """To avoid ambiguity, any strings columns should become categorical."""

        class DummyModelML(DataModelBaseML):
            col: pd.StringDtype  # Actual string column

        inputs = pd.DataFrame({"col": ["Foo", "Bar", "Bay"]})
        expected = inputs.assign(col=inputs["col"].astype("string").astype("category"))
        actual = DummyModelML.validate(inputs)
        pd.testing.assert_frame_equal(actual, expected)

    def test_validate_raises_on_invalid_dtypes(self, caplog: pytest.LogCaptureFixture):
        """Only ML compliant data types are permitted."""

        class DummyModelML(DataModelBaseML):
            col1: object

        inputs = pd.DataFrame({"col1": ["Foo", 12, ()]})
        with (
            pytest.raises(SchemaError, match="coerce_data_types"),
            caplog.at_level(logging.WARNING),
        ):
            DummyModelML.validate(inputs)

        assert caplog.text == IsStr(
            regex=".*non.ml.compliant.*", regex_flags=re.IGNORECASE | re.DOTALL
        )

    def test_validate_coerces_numerical_columns_to_float64(self):
        """To avoid ambiguity, all numerical types should be promoted to float64."""

        class DummyModelML(DataModelBaseML):
            c1: int
            c2: pd.UInt8Dtype
            c3: pd.UInt32Dtype
            c4: pd.UInt64Dtype
            c5: pd.Int8Dtype
            c6: pd.Int32Dtype
            c7: pd.Int64Dtype
            c8: pd.Float32Dtype
            c9: pd.Float64Dtype

        inputs = pd.DataFrame(dict.fromkeys(DummyModelML.to_schema().columns, (4, 5)))
        expected = inputs.copy().astype(float)
        actual = DummyModelML.validate(inputs)
        pd.testing.assert_frame_equal(actual, expected)

    def test_validate_coerces_categorical_types(self):
        """Categorical types should be primitive, too."""

        DATES = pd.date_range("2025-01-01", periods=2, unit="ns", tz="Europe/Berlin")

        class DummyModelML(DataModelBaseML):
            c1: pd.CategoricalDtype
            c2: pd.CategoricalDtype
            c3: pd.CategoricalDtype
            c4: pd.CategoricalDtype
            c5: pd.CategoricalDtype

        inputs = pd.DataFrame(
            {
                "c1": [1, 2],
                "c2": [1.0, 2.0],
                "c3": ["hello", "world"],
                "c4": [False, True],
                "c5": DATES,
            }
        )
        expected = {
            "c1": pd.CategoricalDtype(inputs["c1"].astype("float64"), False),
            "c2": pd.CategoricalDtype(inputs["c2"].astype("float64"), False),
            "c3": pd.CategoricalDtype(inputs["c3"].astype("string"), False),
            "c4": pd.CategoricalDtype(inputs["c4"].astype("bool"), False),
            "c5": pd.CategoricalDtype(DATES.tz_localize(None).as_unit("us"), False),
        }
        actual = DummyModelML.validate(inputs).dtypes.to_dict()
        assert actual == expected

    def test_validate_raises_on_invalid_categorical_types(
        self, caplog: pytest.LogCaptureFixture
    ):
        """Only ML compliant data types are permitted."""

        class DummyModelML(DataModelBaseML):
            col1: pd.CategoricalDtype

        inputs = pd.DataFrame({"col1": ["Foo", 12]})
        with (
            pytest.raises(SchemaError, match="coerce_data_types"),
            caplog.at_level(logging.WARNING),
        ):
            DummyModelML.validate(inputs)

        assert caplog.text == IsStr(
            regex=".*non.ml.compliant.*", regex_flags=re.IGNORECASE | re.DOTALL
        )

    def test_validate_sorts_by_identifiers(self):
        """ML datasets should have deterministic row order if they have ID columns."""

        class DummyModelML(DataModelBaseML):
            c1: float
            c2: float
            c3: float

            class Config:
                unique = ["c2", "c3"]  # noqa: RUF012

        inputs = pd.DataFrame(
            {"c1": [1, 2, 3, 4], "c2": [3, 1, 2, 1], "c3": [2, 4, 1, 3]},
            index=pd.Index([67, 42, 60, 3]),
            dtype=float,
        )
        expected = pd.DataFrame(
            {"c1": [4, 2, 3, 1], "c2": [1, 1, 2, 3], "c3": [3, 4, 1, 2]},  # Sorted
            index=pd.RangeIndex(4),  # Sorted index
            dtype=float,
        )
        actual = DummyModelML.validate(inputs)
        pd.testing.assert_frame_equal(actual, expected)

    def test_validate_does_not_sort_if_no_identifiers_are_set(self):
        """ML datasets should have at least an ordered range-index."""

        class DummyModelML(DataModelBaseML):
            c1: float
            c2: float
            c3: float

        inputs = pd.DataFrame(
            {"c1": [1, 2, 3, 4], "c2": [3, 1, 2, 1], "c3": [2, 4, 1, 3]},
            index=pd.Index([67, 42, 60, 3]),
            dtype=float,
        )
        expected = inputs.reset_index(drop=True)
        actual = DummyModelML.validate(inputs)
        pd.testing.assert_frame_equal(actual, expected)
