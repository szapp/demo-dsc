from .base import DataModelBaseML
from .raw import RawDataModel


class ProcessedDataModel(RawDataModel, DataModelBaseML):
    """Data model for processed and ML-conform data ."""

    # Datatypes from raw data model are coerced to ML-conform types.
    pass
