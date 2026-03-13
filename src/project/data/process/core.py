import logging

import pandas as pd
import pandera.pandas as pa

from ..validate import ProcessedDataModel

logger = logging.getLogger(__name__)


def process_data(
    raw: pd.DataFrame,
    target_column: str,
    *,
    data_model: type[pa.DataFrameModel] = ProcessedDataModel,
) -> tuple[pd.DataFrame, pd.Series]:
    """Filter data for valid rows and outliers and split it into features and target.

    Args:
        raw: Raw unprocessed data.
        target_column: Name of the target column.
        data_model: Data model for validation and conversion.

    Returns:
        Feature DataFrame and Target Series.
    """

    # TODO Filter invalid rows and remove outliers.
    logger.debug("Preprocess raw data")
    processed = raw.copy()

    # Validation and ML-conform conversion
    logger.debug("Validate processed data")
    X = data_model.validate(processed)
    y = X.pop(target_column)

    return X, y
