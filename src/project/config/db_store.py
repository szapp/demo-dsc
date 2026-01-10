import sqlalchemy

from .util import builds


def create_engine(
    url: sqlalchemy.URL | str, pool_recycle: int | None = None
) -> sqlalchemy.Engine:
    """Typesafe version of SQLalchemy's create_engine"""
    return sqlalchemy.create_engine(url, pool_recycle=pool_recycle)


db = builds(
    create_engine, url=builds(sqlalchemy.URL.create, query={}), pool_recycle=1800
)
