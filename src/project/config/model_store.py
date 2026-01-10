from hydra_zen import instantiate
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .util import builds, make_columns, make_steps

# Preprocessing step
preprocessing = None

# Feature engineering
features = builds(
    ColumnTransformer,
    transformers=make_columns(
        example=dict(
            transformer=builds(StandardScaler, with_std=False),
            columns=["col1", "col2"],
        ),
    ),
    verbose_feature_names_out=False,
)

# Feature selection
feature_selection = builds(
    ColumnTransformer,
    transformers=make_columns(
        passthrough=dict(
            transformer="passthrough",
            columns=["col1", "col2"],
        ),
    ),
    remainder="drop",
    verbose_feature_names_out=False,
)

# Final predictor
regressor = builds(RandomForestRegressor, random_state=42)

# The complete model
model = builds(
    Pipeline,
    steps=make_steps(
        preprocessing=preprocessing,
        features=features,
        feature_selection=feature_selection,
        regressor=regressor,
    ),
)


def get_model():
    """Return an instantiated instance of the defined model"""
    return instantiate(model)
