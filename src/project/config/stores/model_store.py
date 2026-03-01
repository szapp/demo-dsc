"""Config store for ML models."""

__all__ = [
    "make_model",
]

from hydra_zen import instantiate, store
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .util import build_columns, build_steps, build_transformers, builds

model_store = store(group="model")


def make_model(name: str = "prod") -> Pipeline:
    """Instantiate an ML model from the model config store for use in notebooks."""
    return instantiate(model_store.get_entry(group="model", name=name)["node"])


# Preprocessing step
preprocessing = None

# Feature engineering
features = builds(
    ColumnTransformer,
    transformers=build_transformers(
        example=dict(
            transformer=builds(StandardScaler, with_std=False),
            columns=build_columns(
                col1=True,
                col2=True,
            ),
        ),
    ),
    verbose_feature_names_out=False,
)

# Feature selection
feature_selection = builds(
    ColumnTransformer,
    transformers=build_transformers(
        passthrough=dict(
            transformer="passthrough",
            columns=build_columns(
                col1=True,
                col2=True,
            ),
        ),
    ),
    remainder="drop",
    verbose_feature_names_out=False,
)

# Final predictor
regressor = builds(RandomForestRegressor, random_state=42)

# The complete model
model_store(
    Pipeline,
    steps=build_steps(
        preprocessing=preprocessing,
        features=features,
        feature_selection=feature_selection,
        regressor=regressor,
    ),
    name="prod",
)

# A baseline model
model_store(
    Pipeline,
    steps=build_steps(regressor=builds(LinearRegression)),
    name="baseline",
)
