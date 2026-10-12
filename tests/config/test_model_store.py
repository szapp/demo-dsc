from dataclasses import dataclass
from importlib.metadata import version

import pytest
from dirty_equals import IsInstance
from packaging.requirements import Requirement
from sklearn.pipeline import Pipeline

from project.config import MODEL_REQUIREMENTS, make_model
from project.config.stores.model_store import make_column_selector_repr


def test_model_store_supplies_model():
    """The make_model function returns an actual scikit-learn pipeline object."""
    actual = make_model("baseline")
    assert actual == IsInstance(Pipeline)


def test_make_column_selector_repr():
    """The make_column_selector should have a generic representation of attributes."""

    @dataclass
    class obj:
        str_attr: str = "hello"
        int_attr: int = 42
        _hidden: str = "hidden"
        param_: str = "hidden"

    actual = make_column_selector_repr(obj())
    assert actual == "obj(str_attr='hello', int_attr=42)"


@pytest.mark.parametrize("requirement", MODEL_REQUIREMENTS)
def test_model_requirements_are_satisfied(requirement):
    """The environment for logging the model must be in sync with the environment."""
    expected = Requirement(requirement)
    actual = version(expected.name)
    assert expected.specifier.contains(actual)
