"""Load, assemble, validate data from data base. No logic."""

import glob
import logging
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from frozendict import frozendict
from sqlalchemy import Engine, TextClause, bindparam, text

from ..types import SqlParam, SqlParams
from ._cache import cache
from .validation import RawDataModel

logger = logging.getLogger(__name__)
PATH_SQL_PATTERN = Path(__file__).parent / "sql" / "*.sql"


def load_sql_files(pattern: Path | str = PATH_SQL_PATTERN) -> frozendict[str, str]:
    """Load sql files into name-content pairs.

    Args:
        pattern: Glob pattern for the sql files.

    Returns:
        Dictionary with file names as keys and file content as values.
    """
    paths = map(Path, sorted(glob.glob(str(pattern), recursive=True)))
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
        bindparam(param, expanding=isinstance(v, tuple))
        for param, v in params.items()
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
    force: bool = False,
) -> pd.DataFrame:
    """Fetch all data from the database based on index-bound SQL queries.

    The `sql_queries` must contain the key "index" that queries the identifying columns.

    Args:
        params: Key-value substitutions for parameterized queries.
        db_engine: Database connection engine.
        sql_queries: Name-query pairs to fetch.
        force: Purge cache (completely) and fetch freshly.

    Returns:
        DataFrame with collected data from all sources.

    Notes:
        The parameter `sql_queries` is a frozendict (immutable) to allow caching.
    """
    queries = dict(sql_queries)
    date_col = ["date"]  # Any possibly appearing date-columns

    # Fetch index with identifiers first
    logger.info("Fetch index")
    qrx = bind_sql_params(queries.pop("index"), **params)
    index = pd.read_sql(qrx, db_engine, params=params, parse_dates=date_col)
    identifiers = index.columns.to_list()
    index = index.set_index(identifiers)

    # Fetch and left join the feature and target columns on the identifiers
    dfs: list[pd.DataFrame] = []
    for name, query in queries.items():
        logger.info(f"Fetch {name}")
        dfs.append(
            pd.read_sql(
                bind_sql_params(query, **params),
                db_engine,
                params=params,
                index_col=identifiers,
                parse_dates=date_col,
            )
        )
    df = index.join(dfs, validate="1:1").reset_index()

    return data_model.validate(df)
