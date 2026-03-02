import pandas as pd
import pandera.pandas as pa

from ..validate import ProcessedDataModel


def process_data(
    raw: pd.DataFrame,
    target: str,
    *,
    data_model: type[pa.DataFrameModel] = ProcessedDataModel,
) -> tuple[pd.DataFrame, pd.Series]:
    """Filter data for valid rows and outliers and split it into features and target.

    Args:
        raw: Raw unprocessed data.
        target: Name of the target column.

    Returns:
        Feature DataFrame and Target Series.
    """

    # TODO Filter invalid rows and remove outliers.

    # Validation and ML-conform conversion
    X = data_model.validate(raw)
    y = X.pop(target)

    return X, y
