import pandas as pd
from inline_snapshot import snapshot

from project.data.validation import checks


def test_is_int_checks_correctly_for_one_column_in_df():
    """The is_int works reliably for DataFrames"""
    inputs = pd.DataFrame({"col1": [1.0, 2.0, -3.2, 0]})
    expected = snapshot({"col1": [True, True, False, True]})
    actual = checks.is_int(inputs)

    assert actual.to_dict("list") == expected


def test_is_int_checks_correctly_for_more_columns_in_df():
    """The is_int works independently for several columns"""
    inputs = pd.DataFrame(
        {
            "col1": [1, 2, -3, 0],
            "col2": [0, 0, 0.2, 0],
        }
    )
    expected = snapshot(
        {
            "col1": [True, True, True, True],
            "col2": [True, True, False, True],
        }
    )
    actual = checks.is_int(inputs)

    assert actual.to_dict("list") == expected


def test_is_int_checks_correctly_for_series():
    """The is_int works reliably for Series"""
    inputs = pd.Series([1, 2.0, -3.1, 0.0])
    expected = snapshot([True, True, False, True])
    actual = checks.is_int(inputs)

    assert actual.to_list() == expected
