import glob
import hashlib
from dataclasses import dataclass, field
from datetime import date
from functools import partial
from pathlib import Path
from typing import Self

import pandas as pd
from joblib import Memory
from sqlalchemy import Engine, text

PATH_SQL_PATTERN = Path(__file__).parent / "sql" / "*.sql"
PATH_CACHE = Path(__file__).parent / ".cache"  # TODO adjust the path
memory = Memory(location=PATH_CACHE, verbose=0)
cache = partial(memory.cache, cache_validation_callback=lambda x: True)


@dataclass(frozen=True)
class QueryLoader(tuple):
    """Cacheable SQL script loader"""

    items: tuple[str, ...] = ()  # Immutable collection of queries
    fingerprint: str = field(init=False)  # Uniquely identify the query content

    def __new__(cls, items: tuple[str, ...] | None = None):
        """Tuples are immutable so __new__ has to be used here"""
        items = items or ()
        obj = super().__new__(cls, items)
        h = hashlib.sha256("".join(items).encode("utf-8")).hexdigest()
        object.__setattr__(obj, "fingerprint", h)
        return obj

    @classmethod
    def from_files(cls: type[Self], pattern: Path | str = PATH_SQL_PATTERN) -> Self:
        """Create a QueryLoader from a path pattern leading to SQL files"""
        paths = map(Path, glob.glob(str(pattern), recursive=True))
        queries = tuple(p.read_text(encoding="utf-8") for p in paths)
        return cls(items=queries)


@cache(ignore=["db"])
def fetch_data(
    db: Engine,
    sql_queries: tuple[str, ...] = QueryLoader.from_files(),
    start_date: date = date(2020, 1, 1),
    end_date: date = date.today(),
    **params: str | int | float | date,
) -> tuple[pd.DataFrame, pd.Series]:
    """Fetch all data from the database based on multiple date-bound SQL queries.

    Args:
        db: Database connection engine.
        queries: Tuple of SQL queries.
        start_date: Start date of data.
        end_date: End date of data.

    Returns:
        Collected data from all sources.
    """
    params = dict(start_date=start_date, end_date=end_date, **params)
    dfs = (
        pd.read_sql(text(q), db, index_col="date", params=params, parse_dates=["date"])
        for q in sql_queries
    )
    df = pd.DataFrame(index=pd.date_range(start_date, end_date))
    X = df.join(dfs, validate="1:1").reset_index()
    y = X.pop("target")
    return X, y
