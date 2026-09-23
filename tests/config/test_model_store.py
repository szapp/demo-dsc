from importlib.metadata import version

import pytest
from dirty_equals import IsInstance
from packaging.requirements import Requirement
from sklearn.pipeline import Pipeline

from project.config import MODEL_REQUIREMENTS, make_model


def test_model_store_supplies_model():
    """The make_model function returns an actual scikit-learn pipeline object."""
    actual = make_model("baseline")
    assert actual == IsInstance(Pipeline)


@pytest.mark.parametrize("requirement", MODEL_REQUIREMENTS)
def test_model_requirements_are_satisfied(requirement):
    """The environment for logging the model must be in sync with the environment."""
    expected = Requirement(requirement)
    actual = version(expected.name)
    assert expected.specifier.contains(actual)
