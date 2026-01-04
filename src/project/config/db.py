from sqlalchemy import URL, create_engine

from .util import builds

db = builds(create_engine, url=builds(URL.create, query=dict()), pool_recycle=1800)
