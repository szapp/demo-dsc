import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from hydra_zen import zen
from hydra_zen.third_party.pydantic import pydantic_parser

from .exception_logger import InitWrapper
from .stores import store

CONFIG_PATH = str(Path("config").resolve())  # Absolute path from CWD


def make_cli(func: Callable[..., Any]) -> Callable[[], None]:  # pragma: no cover
    """Make a function run with Hydra-Zen CLI."""

    def cli() -> None:
        import optuna.logging  # noqa: F401 - Import here to suppress rogue logging

        load_dotenv()
        config_name = os.environ.get("ENV") or "dev"
        config_path = "." if config_name in {"test", "prod"} else CONFIG_PATH
        store.add_to_hydra_store()
        entrypoint = zen(func, instantiation_wrapper=InitWrapper(pydantic_parser))
        entrypoint.hydra_main(config_path, config_name, version_base=None)

    return cli
