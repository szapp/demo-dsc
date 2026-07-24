"""Data models for validation of raw data and model inputs."""

__all__ = [
    "ProcessedDataModel",
    "RawDataModel",
]

from .processed import ProcessedDataModel
from .raw import RawDataModel
