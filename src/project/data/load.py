import glob
import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from functools import wraps
from pathlib import Path
from time import time
from typing import ParamSpec, Self, TypeVar

import pandas as pd
from joblib import Memory
from sqlalchemy import Engine, text

from .model import RawDataModel

logger = logging.getLogger(__name__)
PATH_SQL_PATTERN = Path(__file__).parent / "sql" / "*.sql"
PATH_CACHE = Path(__file__).parent / ".cache"  # TODO adjust the path
CACHE_SEC = 60 * 60 * 6  # 6 hours as seconds
memory = Memory(location=PATH_CACHE, verbose=0)
cache = memory.cache(cache_validation_callback=lambda x: time() - x["time"] < CACHE_SEC)

P = ParamSpec("P")
R = TypeVar("R")


def typed_cache(func: Callable[P, R]) -> Callable[P, R]:
    """Allows cached functions to be recognized by pydantic."""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return func(*args, **kwargs)

    return wrapper


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


@typed_cache
@cache(ignore=["db_engine"])
def fetch_data(
    start_date: date,
    end_date: date,
    db_engine: Engine,
    sql_queries: tuple[str, ...] = QueryLoader.from_files(),
    **params: str | int | float | date,
) -> tuple[pd.DataFrame, pd.Series]:
    """Fetch all data from the database based on multiple date-bound SQL queries.

    Args:
        db: Database connection engine.
        queries: Tuple of SQL queries.
        start_date: Start date of data.
        end_date: End date of data.

    Returns:
        X: Collected features from all sources.
        y: Target columns.
    """
    logger.info("Fetch data.")
    params = dict(start_date=start_date, end_date=end_date, **params)
    dfs = (
        pd.read_sql(
            text(q), db_engine, index_col="date", params=params, parse_dates=["date"]
        )
        for q in sql_queries
    )
    df = pd.DataFrame(index=pd.date_range(start_date, end_date, name="date"))
    df = df.join(dfs, validate="1:1").reset_index()

    X = RawDataModel.validate(df)
    y = X.pop("target")

    return X, y
