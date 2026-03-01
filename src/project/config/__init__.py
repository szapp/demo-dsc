"""Configuration for entrypoints. Must only be imported by the `entrypoints`
subpackage.
"""

__all__ = [
    "CONFIG_PATH",
    "InitWrapper",
    "make_db_engine",
    "make_model",
    "store",
]

from pathlib import Path

from .exception_logger import InitWrapper
from .stores import store
from .stores.db_store import make_db_engine
from .stores.model_store import make_model

CONFIG_PATH = str(Path("config").resolve())  # Absolute path from CWD
