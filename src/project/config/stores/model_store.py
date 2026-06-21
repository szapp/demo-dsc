"""Config store for ML models."""

from hydra_zen import instantiate, store
from mlflow.sklearn import load_model as load_mlflow_model
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ...version import SERVICE
from .util import build_columns, build_steps, build_transformers, builds

__all__ = [
    "make_model",
]

model_store = store(group="model")


def make_model(name: str = "prod") -> Pipeline:
    """Instantiate an ML model from the model config store for use in notebooks."""
    return instantiate(model_store.get_entry(group="model", name=name)["node"])


model_store(
    load_mlflow_model,
    model_uri=f"models:/{SERVICE}-prod@champion",
    dst_path=".",
    name="champion",
)


# Preprocessing step
preprocessing = builds(
    ColumnTransformer,
    transformers=build_transformers(
        numeric=dict(
            transformer=builds(SimpleImputer, strategy="constant", fill_value=0),
            columns=builds(make_column_selector, dtype_include="number"),
        ),
    ),
    remainder="passthrough",
    verbose_feature_names_out=False,
)

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
        ohe=dict(
            transformer=builds(OneHotEncoder, sparse_output=False),
            columns=builds(make_column_selector, dtype_include="category"),
        ),
    ),
    remainder="passthrough",
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
                col3=True,
                col4_Apple=True,
                col4_Banana=True,
                col4_Cherry=True,
                col4_Date=True,
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
