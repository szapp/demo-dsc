import pandas as pd
import pytest
from pandera.errors import SchemaError

from project.data.validate import RawDataModel


class TestRawDataModel:
    def test_validate_raises_on_duplicate_identifier_pairs(self):
        """Data is joined on the identifiers that have to be unique pairs."""
        inputs = pd.DataFrame(
            {
                "id": [0, 1, 1],
                "date": ["2024-01-01", "2024-01-01", "2024-01-01"],
                "col1": [0, 0, 0],
                "col2": [0, 0, 0],
                "col3": [0, 0, 0],
                "col4": ["Apple", None, None],
                "target": [0, 0, 0],
            }
        )
        with pytest.raises(SchemaError, match="identifiers_are_unique"):
            RawDataModel.validate(inputs)
