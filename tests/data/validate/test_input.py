import pandas as pd
import pandera.pandas as pa
import pytest
from pandera.errors import SchemaError

from project.data.validate import ProcessedDataModel, RawDataModel


class ReducedRawDataModel(RawDataModel):
    """RawDataModel with minimal number of columns."""

    @classmethod
    def to_schema(cls) -> pa.DataFrameSchema:
        schema = super().to_schema()
        return schema.select_columns([cls.id, cls.date])


class ReducedProcessedDataModel(ProcessedDataModel):
    """ProcessedDataModel with minimal number of columns."""

    @classmethod
    def to_schema(cls) -> pa.DataFrameSchema:
        schema = super().to_schema()
        return schema.select_columns([cls.id, cls.date, cls.col2])


class TestRawDataModel:
    def test_validate_raises_on_duplicate_identifier_pairs(self):
        """Data is joined on the identifiers that have to be unique pairs."""
        inputs = pd.DataFrame(
            {
                "id": [0, 1, 1],
                "date": ["2024-01-01", "2024-01-01", "2024-01-01"],
            }
        )
        with pytest.raises(SchemaError, match="not unique"):
            ReducedRawDataModel.validate(inputs)


class TestProcessedDataModel:
    def test_validate_does_not_warn_on_nan_columns(self):
        """NaN columns are already detected when validating the raw data model."""
        inputs = pd.DataFrame(
            {
                "id": [0, 1],
                "date": ["2024-01-01", "2024-01-01"],
                "col2": [None, None],
            }
        )

        # No error or warning
        ReducedProcessedDataModel.validate(inputs)
