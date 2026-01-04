from datetime import date

import polars as pl
from sqlalchemy import Engine


def fetch_data(
    db: Engine, end_date: date, start_date: date = date(2020, 1, 1)
) -> tuple[pl.DataFrame, pl.Series]:
    data = pl.DataFrame(
        {"col1": [1, 2], "col2": [2, 4], "col3": [3, 2], "target": [1, 2]}
    )
    y = data.get_column("target")
    X = data.drop("target")
    return X, y
