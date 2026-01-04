from datetime import date

import pandas as pd
from sqlalchemy import Engine


def fetch_data(
    db: Engine, end_date: date, start_date: date = date(2020, 1, 1)
) -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(
        {"col1": [1, 2], "col2": [2, 4], "col3": [3, 2], "target": [1, 2]}
    ).astype(float)
    y = X.pop("target")
    return X, y
