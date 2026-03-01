"""Types and type hint definitions."""

from datetime import date, datetime

SqlPrimitive = str | int | float | date | datetime
SqlParam = SqlPrimitive | tuple[SqlPrimitive, ...]
SqlParams = dict[str, SqlParam]
