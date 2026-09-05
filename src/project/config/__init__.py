"""Configuration for entrypoints. Must only be imported by the `entrypoints`
subpackage.
"""

__all__ = [
    "make_cli",
    "make_db_engine",
    "make_model",
    "store",
]

from .make_cli import make_cli
from .stores import store
from .stores.db_store import make_db_engine
from .stores.model_store import make_model
