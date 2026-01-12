from hydra_zen import instantiate, store
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .util import builds, make_columns, make_steps, make_transformers

model_store = store(group="model")

# Preprocessing step
preprocessing = None

# Feature engineering
features = builds(
    ColumnTransformer,
    transformers=make_transformers(
        example=dict(
            transformer=builds(StandardScaler, with_std=False),
            columns=make_columns(
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
    transformers=make_transformers(
        passthrough=dict(
            transformer="passthrough",
            columns=make_columns(
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
    steps=make_steps(
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
    steps=make_steps(regressor=builds(LinearRegression)),
    name="baseline",
)


def get_model(name: str = "prod"):
    """Return an instantiated model from the model config store by its name"""
    return instantiate(model_store.get_entry(group="model", name=name)["node"])
