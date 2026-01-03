import logging
from functools import partial
from pathlib import Path
from typing import Any

import hydra
import polars as pl
from hydra_zen import MISSING, builds, instantiate, make_config, store
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import estimator_html_repr

logger = logging.getLogger(__name__)
CONFIG_PATH = str(Path("configs").resolve())


def make_steps_(**steps: Any) -> list[tuple[str, Any]]:
    return [(name, step) for name, step in steps.items()]


make_steps = partial(builds, make_steps_)


preprocessing = store(group="model/steps/preprocessing")
preprocessing(StandardScaler, with_mean=False, name="standard_scalar")

regressors = store(group="model/steps/regressor")
regressors(RandomForestRegressor, random_state=42, name="random_forest")

model = builds(
    Pipeline,
    steps=make_steps(
        preprocessing=None,
        regressor=MISSING,
    ),
    hydra_convert="object",
)

store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"model/steps/preprocessing": "standard_scalar"},
            {"model/steps/regressor": "random_forest"},
        ],
        dev="${oc.decode:${oc.env:DSC_HOME,true}}",  # Controlled by env-var "DSC_DEV" and true by default
        model=model,
    ),
    name="base_config",
)
store.add_to_hydra_store()

# store(
#     Pipeline,
#     steps=make_steps(
#         pre_processing=builds(StandardScaler, with_mean=False),
#         model=builds(RandomForestRegressor, random_state=42),
#     ),
#     hydra_convert="object",
# )


# dev = "${oc.env:DSC_DEV,false}"
# a = "${dev}"
# model = builds(
#     Pipeline,
#     steps=make_steps(
#         pre_processing=builds(StandardScaler, with_mean=False),
#         model=builds(RandomForestRegressor, random_state=42),
#     ),
#     hydra_convert="object",
# )

# BaseConfig = make_config(dev=builds(bool, dev), model=model, a=a)

# store = ZenStore(deferred_hydra_store=False)
# store(BaseConfig, name="base_config")

# a = store[None, "base_config"]
# b = a()
# print(type(b))
# print(isinstance(b, DictConfig))


def my_func(a: int) -> int:
    return a + 2


from hydra_zen.third_party.pydantic import pydantic_parser

# print("1")
# my_func_b = builds(my_func, "hello")
# print("3")
# print(instantiate(my_func_b, _target_wrapper_=pydantic_parser))
# print("4")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="base_config")
def main(cfg: DictConfig) -> None:
    print(type(cfg))

    resolved = instantiate(cfg, _target_wrapper_=pydantic_parser)

    print(resolved)

    return

    model = resolved.model
    model.set_output(transform="polars")

    X = pl.DataFrame(
        {
            "col1": [0, 1, 2, 3],
            "col2": [0, 1, 0, 1],
            "col3": [0, 0, 1, 0],
            "col4": [1, 1, 1, 0],
            "col5": [0, 1, 1, 1],
        }
    )
    y = [0, 1, 2, 3]
    model.fit(X, y)
    logger.info(model.score(X, y))

    html = estimator_html_repr(model)
    Path(CONFIG_PATH, "..", "model.html").write_text(html, encoding="utf8")
