from typing import Annotated

import pandas as pd
import pytest
from pandera.errors import SchemaError
from pandera.pandas import Field as F

from project.data.validate.base import DataModelBase, DataModelBaseML


class DummyModel(DataModelBase):
    _pre_rename = {"misspelled": "float32_col"}
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
    def test_validate_renames_columns(self, df_dummy_base: pd.DataFrame):
        """The names of DataFrame columns must be adjustable."""
        inputs = df_dummy_base.copy().rename(columns={"float32_col": "misspelled"})
        expected = ["float_col", "float32_col", "int_col", "date_col", "bool_col"]
        actual = DummyModel.validate(inputs).columns.tolist()
        assert actual == expected

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

    def test_validate_raises_on_nan_columns(self, df_dummy_base: pd.DataFrame):
        """Data with nan-columns suggest faulty data sources."""
        inputs = df_dummy_base.assign(int_col=pd.NA)
        with pytest.raises(SchemaError, match="at_least_one_value"):
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

        date_before = pd.to_datetime(["2025-04-02T03:00:00+02:00"]).as_unit("us")
        date_after = pd.to_datetime(["2025-04-02T01:00:00"]).as_unit("us")
        inputs = pd.DataFrame({"col": date_before})
        expected = pd.DataFrame({"col": date_after})
        actual = DummyModelML.validate(inputs)
        pd.testing.assert_frame_equal(actual, expected)

    def test_validate_keeps_time_zone_agnostic_datetime_as_is(self):
        """MLflow does not support time zone information."""

        class DummyModelML(DataModelBaseML):
            col: pd.Timestamp

        date_before = pd.to_datetime(["2025-04-02"]).as_unit("ns")
        date_after = pd.to_datetime(["2025-04-02"]).as_unit("us")
        inputs = pd.DataFrame({"col": date_before})
        expected = pd.DataFrame({"col": date_after})
        actual = DummyModelML.validate(inputs)
        pd.testing.assert_frame_equal(actual, expected)

    def test_validate_coerces_datetime_resolution(self):
        """MLflow does not support time zone information."""

        class DummyModelML(DataModelBaseML):
            col: Annotated[pd.DatetimeTZDtype, "us", "UTC"]

        date_before = pd.to_datetime(["2025-04-02"]).as_unit("ns")
        date_after = pd.to_datetime(["2025-04-02"]).as_unit("us")
        inputs = pd.DataFrame({"col": date_before})
        expected = pd.DataFrame({"col": date_after})
        actual = DummyModelML.validate(inputs)
        pd.testing.assert_frame_equal(actual, expected)

    def test_validate_imputes_boolean_nan_to_false(self):
        """Missing values in booleans are not fully supported by Scikit-Learn."""

        class DummyModelML(DataModelBaseML):
            col: pd.BooleanDtype = F(nullable=True)

        inputs = pd.DataFrame({"col": [True, False, None]})
        expected = pd.DataFrame({"col": [True, False, False]}, dtype=pd.BooleanDtype())
        actual = DummyModelML.validate(inputs)
        pd.testing.assert_frame_equal(actual, expected)

    def test_validate_coerces_categorical_columns(self):
        """To avoid ambiguity, any strings columns should become categorical."""

        class DummyModelML(DataModelBaseML):
            col: pd.StringDtype  # Actual string column
            col2: object  # Object column remains unchanged

        inputs = pd.DataFrame(
            {
                "col": ["Foo", "Bar", "Bay"],
                "col2": ["Foo", 12, dict()],
            }
        )
        expected = inputs.assign(col=inputs["col"].astype("string").astype("category"))
        actual = DummyModelML.validate(inputs)
        pd.testing.assert_frame_equal(actual, expected)

    def test_validate_coerces_numerical_columns_to_float64(self):
        """To avoid ambiguity, all numerical types should be promoted to float64."""

        class DummyModelML(DataModelBaseML):
            col: pd.UInt32Dtype
            col2: pd.Float32Dtype
            col3: pd.Float64Dtype

        inputs = pd.DataFrame({"col": [2], "col2": [3], "col3": [4]})
        expected = inputs.copy().astype(pd.Float64Dtype())
        actual = DummyModelML.validate(inputs)
        pd.testing.assert_frame_equal(actual, expected)
