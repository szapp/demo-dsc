from collections.abc import Generator

import pandas as pd
import pandera.pandas as pa
import pytest
from dirty_equals import IsList
from frozendict import frozendict
from inline_snapshot import snapshot
from pandas.errors import MergeError
from pandera.errors import SchemaError
from pandera.pandas import Field as F
from sqlalchemy import Engine, create_engine

from project.data import fetch_data

fetch_data_uncached = getattr(fetch_data, "uncached", None) or fetch_data


class RawDataModel(pa.DataFrameModel):
    id: pd.Int64Dtype
    date: pd.Timestamp
    feature: pd.Float64Dtype = F(nullable=True)
    target: pd.Int64Dtype = F(nullable=True, coerce=True)


@pytest.fixture(scope="module", name="engine")
def _engine() -> Generator[Engine]:
    """Run a data base and fill it with dummy data for the tests."""
    engine = create_engine("sqlite:///:memory:")

    data = pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3],
            "date": pd.date_range("2026-01-01", "2026-01-02").tolist() * 3,
            "feature": [42.0, 43.0, 44.0, 45.0, 46.0, 47.0],
            "target": [0, 1, 2, 3, 4, 5],
        }
    )

    # Add the data with missing values for features and target
    data[["id", "date"]].to_sql("identifier", engine, index=False)
    data.loc[::2, ["id", "date", "feature"]].to_sql("feature", engine, index=False)
    data.loc[[0, 2, 3], ["id", "date", "target"]].to_sql("target", engine, index=False)

    yield engine

    engine.dispose()


def test_fetch_data_parametrizes_queries_correctly(engine):
    """Expanded and missing parameters must be handled correctly in SQL queries."""
    sql_queries = frozendict(
        {
            "index": "SELECT * FROM identifier WHERE id in :valid_ids",
            "features": "SELECT * FROM feature WHERE feature >= :min_feature",
            "target": "SELECT * FROM target",
        }
    )
    params = {
        "valid_ids": (1, 2),  # Test expanded parameter binding
        "min_feature": 44,
        "extra_param": "unused",  # Test that unused parameters are no problem
    }
    expected = snapshot(
        {
            "id": [1, 1, 2, 2],
            "date": IsList(length=4),
            "feature": [-42, -42, 44, -42],
            "target": [0, -42, 2, 3],
        }
    )

    actual = fetch_data_uncached(
        params=params,
        db_engine=engine,
        sql_queries=sql_queries,
        data_model=RawDataModel,
    )

    actual = actual.fillna(-42)  # Issues with NaN in inline_snapshot
    assert actual.to_dict("list") == expected


def test_fetch_data_left_joins_data_correctly(engine):
    """The data is left joined on the index columns allowing NaN values in columns."""
    sql_queries = frozendict(
        {
            "index": "SELECT * FROM identifier",
            "features": "SELECT * FROM feature",
            "target": "SELECT * FROM target",
        }
    )
    params = {}
    expected = snapshot(
        {
            "id": [1, 1, 2, 2, 3, 3],
            "date": IsList(length=6),
            "feature": [42, -42, 44, -42, 46, -42],
            "target": [0, -42, 2, 3, -42, -42],
        }
    )

    actual = fetch_data_uncached(
        params=params,
        db_engine=engine,
        sql_queries=sql_queries,
        data_model=RawDataModel,
    )

    actual = actual.fillna(-42)  # Issues with NaN in inline_snapshot
    assert actual.to_dict("list") == expected


def test_fetch_data_cross_joins_multiple_indices_correctly(engine):
    """Multiple identifiers are cross joined allowing to construct a complex index."""
    sql_queries = frozendict(
        {
            "index_a": "SELECT distinct id FROM identifier",
            "index_b": "SELECT distinct date FROM identifier",
            "features": "SELECT * FROM feature",
            "target": "SELECT * FROM target",
        }
    )
    params = {}
    expected = snapshot(
        {
            "id": [1, 1, 2, 2, 3, 3],
            "date": IsList(length=6),
            "feature": [42, -42, 44, -42, 46, -42],
            "target": [0, -42, 2, 3, -42, -42],
        }
    )

    actual = fetch_data_uncached(
        params=params,
        db_engine=engine,
        sql_queries=sql_queries,
        data_model=RawDataModel,
    )

    actual = actual.fillna(-42)  # Issues with NaN in inline_snapshot
    assert actual.to_dict("list") == expected


def test_fetch_data_raises_if_index_not_supplied(engine):
    """The data fetching requires specific identifiers to left join on."""
    sql_queries = frozendict(
        {
            "features": "SELECT * FROM feature",
            "target": "SELECT * FROM target",
        }
    )
    params = {}

    with pytest.raises(KeyError, match="index"):
        fetch_data_uncached(
            params=params,
            db_engine=engine,
            sql_queries=sql_queries,
            data_model=RawDataModel,
        )


def test_fetch_data_raises_on_merge_validation(engine):
    """The left join is a 1-to-1 row mapping. There must not be more rows per ID."""
    sql_queries = frozendict(
        {
            "index": "SELECT * FROM identifier",
            "features": "SELECT * FROM feature UNION ALL SELECT * FROM feature;",
            "target": "SELECT * FROM target",
        }
    )
    params = {}

    with pytest.raises(MergeError, match="not unique"):
        fetch_data_uncached(
            params=params,
            db_engine=engine,
            sql_queries=sql_queries,
            data_model=RawDataModel,
        )


def test_fetch_data_raises_on_missing_column_during_data_model_validation(engine):
    """The resulting data must match the expected data model."""
    sql_queries = frozendict(
        {
            "index": "SELECT * FROM identifier",
            "target": "SELECT * FROM target",
        }
    )
    params = {}

    with pytest.raises(SchemaError, match="not in dataframe"):
        fetch_data_uncached(
            params=params,
            db_engine=engine,
            sql_queries=sql_queries,
            data_model=RawDataModel,
        )


def test_fetch_data_raises_on_incorrect_data_type_during_data_model_validation(engine):
    """The resulting data must match the expected data types."""
    sql_queries = frozendict(
        {
            "index": "SELECT * FROM identifier",
            "features": (
                "SELECT date, id, CAST(feature AS VARCHAR) as feature FROM feature"
            ),
            "target": "SELECT * FROM target",
        }
    )
    params = {}

    with pytest.raises(SchemaError, match="expected.*feature.*to have type [fF]loat"):
        fetch_data_uncached(
            params=params,
            db_engine=engine,
            sql_queries=sql_queries,
            data_model=RawDataModel,
        )
