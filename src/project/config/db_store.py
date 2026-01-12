import sqlalchemy
from hydra_zen import store

from .util import builds


def create_engine(
    url: sqlalchemy.URL | str, pool_recycle: int = -1
) -> sqlalchemy.Engine:
    """Typesafe version of SQLalchemy's create_engine"""
    return sqlalchemy.create_engine(url, pool_recycle=pool_recycle)


db_store = store(group="db")

# Production
db_store(
    create_engine,
    url=builds(
        sqlalchemy.URL.create,
        drivername="postgresql+psycopg2",
        username="${os.env:DB_PROD_USERNAME}",
        password="${os.env:DB_PROD_PASSWORD}",
        host="localhost",
        port="5432",
        database="mydb",
        query=dict(),
    ),
    pool_recycle=1800,
    name="prod",
)


# In-memory sqlite for testing
db_store(
    create_engine,
    url=builds(
        sqlalchemy.URL.create,
        drivername="sqlite",
        database=":memory:",
        query=dict(),
    ),
    name="memory",
)
