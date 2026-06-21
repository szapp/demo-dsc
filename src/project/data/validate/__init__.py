"""Data models for validation of raw data and model inputs."""

__all__ = [
    "RawDataModel",
    "ProcessedDataModel",
]

from .processed import ProcessedDataModel
from .raw import RawDataModel
