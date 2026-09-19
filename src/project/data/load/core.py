"""Load, assemble, validate data from data base. No logic."""

import glob
import logging
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from frozendict import frozendict
from joblib import expires_after
from joblib_typed_cache import Memory
from sqlalchemy import Engine, TextClause, bindparam, text
from structlog.contextvars import bind_contextvars, unbind_contextvars
from tqdm.auto import tqdm

from ...types import SqlParam, SqlParams
from ..validate import RawDataModel

PATH_SQL_PATTERN = str(Path(__file__).parent / "sql" / "*.sql")
ENV_VAR_CACHE_DIR = "JOBLIB_CACHE_DIR"
DEFAULT_CACHE_DIR = "~" + os.sep + ".cache"  # Joblib resolves the path automatically
PATH_CACHE_DIR = os.environ.get(ENV_VAR_CACHE_DIR) or DEFAULT_CACHE_DIR
memory = Memory(location=PATH_CACHE_DIR, verbose=0)
cache = memory.cache(cache_validation_callback=expires_after(hours=6))
logger = logging.getLogger(__name__)


def load_sql_files(pattern: str = PATH_SQL_PATTERN) -> frozendict[str, str]:
    """Load sql files into name-content pairs.

    Args:
        pattern: Glob pattern for the sql files.

    Returns:
        Dictionary with file names as keys and file content as values.
    """
    paths = map(Path, sorted(glob.glob(pattern, recursive=True)))
    queries = frozendict({p.stem: p.read_text(encoding="utf-8") for p in paths})
    return queries


def bind_sql_params(query: str, **params: SqlParam) -> TextClause:
    """Prepare bound SQL parameters for multi-value substitutions.

    This is necessary to parameterize SQL statements like "WHERE col IN (1, 2, 3)".

    Args:
        query: SQL query with parameters as ':param'.
        params: Key-value substitutions for parameterized query.

    Returns:
        SQL statement with selectively bound and expanded parameters.
    """
    stmt = text(query)
    used = stmt._bindparams.keys()
    bound_params = (
        bindparam(param, value, expanding=isinstance(value, tuple))
        for param, value in params.items()
        if param in used
    )
    return stmt.bindparams(*bound_params)


@cache(ignore=["db_engine"])
def fetch_data(
    params: SqlParams,
    db_engine: Engine,
    sql_queries: frozendict[str, str],
    *,
    data_model: type[pa.DataFrameModel] = RawDataModel,
    date_col: Sequence[str] = ("date",),
) -> pd.DataFrame:
    """Fetch all data from the database based on index-bound SQL queries.

    The `sql_queries` must contain the key "index" that queries the identifying columns.

    Args:
        params: Key-value substitutions for parameterized queries.
        db_engine: Database connection engine.
        sql_queries: Name-query pairs to fetch.
        data_model: Data model for validation and conversion.
        date_col: List of any possibly appearing datetime-columns to parse.

    Returns:
        DataFrame with collected data from all sources.

    Notes:
        The parameter `sql_queries` is a frozendict (immutable) to allow caching. The
        cache can be cleared by invoking `fetch_data.clear()`.
    """
    queries = dict(sql_queries)

    # Fetch index first and construct identifiers with cross-join for later left-joins
    idx_queries = [name for name in queries if name.startswith("index")]
    if not idx_queries:
        raise KeyError("At least one index SQL query expected")
    bind_contextvars(**{k: str(v) for k, v in params.items()})
    index = pd.Series(1).to_frame(name="_empty")
    for name in idx_queries:
        bind_contextvars(query_name=name)
        logger.debug("Fetch data from database")
        stmt = bind_sql_params(queries.pop(name), **params)
        data = pd.read_sql_query(stmt, db_engine, parse_dates=list(date_col))
        index = index.merge(data, how="cross")
    unbind_contextvars("query_name")
    index = index.drop(columns="_empty")
    identifiers = index.columns.tolist()

    # Fetch and left-join the feature and target columns on the identifiers
    dfs: list[pd.DataFrame] = []
    for name, query in tqdm(queries.items(), desc="Load data"):
        bind_contextvars(query_name=name)
        logger.debug("Fetch data from database")
        stmt = bind_sql_params(query, **params)
        data = pd.read_sql_query(stmt, db_engine, parse_dates=list(date_col))
        dfs.append(index.merge(data, validate="m:1").set_index(identifiers))
    unbind_contextvars("query_name", *params)
    index = index.set_index(identifiers).sort_index()
    df = index.join(dfs, validate="1:1").reset_index()

    logger.info("Validate raw data", extra={"num_samples": len(df)})
    return data_model.validate(df)
