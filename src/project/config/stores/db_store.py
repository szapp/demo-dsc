"""Config store for database connections with SQLAlchemy engines."""

__all__ = [
    "create_engine",
    "db_store",
    "make_db_engine",
]

from dotenv import load_dotenv
from hydra_zen import instantiate, store
from sqlalchemy import URL, Engine
from sqlalchemy import create_engine as sa_create_engine

from .util import builds

db_store = store(group="db")


def make_db_engine(name: str) -> Engine:
    """Instantiate an SQLAlchemy Engine from the DB store for use in notebooks."""
    load_dotenv()  # Load secrets
    return instantiate(db_store.get_entry(group="db", name=name)["node"])


def create_engine(url: URL | str, pool_recycle: int = -1) -> Engine:
    """Type-safe version of SQLalchemy's create_engine for pydantic"""
    return sa_create_engine(url, pool_recycle=pool_recycle)


# Production
db_store(
    create_engine,
    url=builds(
        URL.create,
        drivername="postgresql+psycopg2",
        username="${oc.env:DB_PROD_USERNAME}",
        password="${oc.env:DB_PROD_PASSWORD}",
        host="localhost",
        port="5432",
        database="mydb",
        query=dict(),
    ),
    pool_recycle=1800,
    name="prod",
)

# In-memory sqlite for dev
db_store(
    create_engine,
    url=builds(
        URL.create,
        drivername="sqlite",
        database=":memory:",
        query=dict(),
    ),
    name="memory",
)
