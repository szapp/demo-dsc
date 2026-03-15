import pandas as pd
import pandera.pandas as pa
import pytest
from dirty_equals import IsList
from frozendict import frozendict
from inline_snapshot import snapshot
from pandas.errors import MergeError
from pandera.errors import SchemaError
from pandera.pandas import Field as F
from pandera.typing.pandas import Series as S
from sqlalchemy import create_engine

from project.data import fetch_data

fetch_data_uncached = getattr(fetch_data, "uncached", None) or fetch_data


class RawDataModel(pa.DataFrameModel):
    id: S[pa.Int64]
    date: S[pa.Timestamp]
    feature: S[pd.Int64Dtype] = F(nullable=True, coerce=True)  # Nullable integer
    target: S[pd.Int64Dtype] = F(nullable=True, coerce=True)


@pytest.fixture(scope="module", name="engine")
def _engine():
    """Run a data base and fill it with dummy data for the tests."""
    engine = create_engine("sqlite:///:memory:")

    data = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "date": pd.date_range("2026-01-01", "2026-01-04"),
            "feature": [42, 43, 44, 45],
            "target": [0, 1, 2, 3],
        }
    )

    # Add the data with missing values for features and target
    data[["id", "date"]].to_sql("identifier", engine, index=False)
    data.loc[[0, 1], ["id", "date", "feature"]].to_sql("feature", engine, index=False)
    data.loc[[0, 1, 3], ["id", "date", "target"]].to_sql("target", engine, index=False)

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
        "min_feature": 40,
        "extra_param": "unused",  # Test that unused parameters are no problem
    }
    expected = snapshot(
        {
            "id": [1, 2],
            "date": IsList(length=2),
            "feature": [42, 43],
            "target": [0, 1],
        }
    )

    actual = fetch_data_uncached(
        params=params,
        db_engine=engine,
        sql_queries=sql_queries,
        data_model=RawDataModel,
    )

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
            "id": [1, 2, 3, 4],
            "date": IsList(length=4),
            "feature": [42, 43, None, None],
            "target": [0, 1, None, 3],
        }
    )

    actual = fetch_data_uncached(
        params=params,
        db_engine=engine,
        sql_queries=sql_queries,
        data_model=RawDataModel,
    )

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
            "index": "SELECT CAST(id AS VARCHAR) as id, date FROM identifier",
            "features": "SELECT * FROM feature",
            "target": "SELECT * FROM target",
        }
    )
    params = {}

    with pytest.raises(SchemaError, match="to have type int"):
        fetch_data_uncached(
            params=params,
            db_engine=engine,
            sql_queries=sql_queries,
            data_model=RawDataModel,
        )
