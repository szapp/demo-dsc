import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)
CONFIG_PATH = str(Path("configs").resolve())


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="old_basic")
def main(cfg: DictConfig) -> None:
    resolved = hydra.utils.instantiate(cfg)
    print(resolved)
    resolved.model.fit([[3, 4], [3, 5], [1, 3]], [1, 2, 3])
