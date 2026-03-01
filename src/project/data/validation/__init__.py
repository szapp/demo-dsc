"""Data models for validation of raw data and model inputs."""

__all__ = [
    "RawDataModel",
    "ProcessedDataModel",
]

from . import checks  # noqa: F401
from .processed import ProcessedDataModel
from .raw import RawDataModel
