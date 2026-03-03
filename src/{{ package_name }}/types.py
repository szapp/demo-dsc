"""Types and type hint definitions."""

from datetime import date, datetime
from typing import Annotated

from pydantic import AfterValidator
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

SqlPrimitive = str | int | float | date | datetime
SqlParam = SqlPrimitive | tuple[SqlPrimitive, ...]
SqlParams = dict[str, SqlParam]

FittedPipeline = Annotated[
    Pipeline,
    AfterValidator(lambda est: check_is_fitted(est) or est),
]
