import pandas as pd
import pandera.pandas as pa
import pytest
from pandera.errors import SchemaError

from project.data.validate import RawDataModel


class ReducedRawDataModel(RawDataModel):
    """RawDataModel with minimal number of columns."""

    @classmethod
    def to_schema(cls) -> pa.DataFrameSchema:
        schema = super().to_schema()
        return schema.select_columns([cls.id, cls.date])


class TestRawDataModel:
    def test_validate_raises_on_duplicate_identifier_pairs(self):
        """Data is joined on the identifiers that have to be unique pairs."""
        inputs = pd.DataFrame(
            {
                "id": [123, 123, 456],
                "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            }
        )
        with pytest.raises(SchemaError, match="not unique"):
            ReducedRawDataModel.validate(inputs)

    def test_validate_does_not_raise_on_semi_duplicates(self):
        """Partial duplicates across the identifier columns are allowed."""
        inputs = pd.DataFrame(
            {
                "id": [123, 456, 789],
                "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            }
        )
        ReducedRawDataModel.validate(inputs)
