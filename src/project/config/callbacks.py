import os
from typing import Any

import mlflow
from hydra import TaskFunction
from hydra.experimental.callback import Callback
from omegaconf import DictConfig


class ParentRunCallback(Callback):  # pragma: no cover
    """Start a parent MLflow run to capture all jobs during multi-run.

    Notes:
        This callback is only relevant for multi-run jobs.
    """

    def __init__(self, name: str, description: str | None = None) -> None:
        self.name = name
        self.description = description
        self._parent_run = None

    def on_multirun_start(self, config: DictConfig, **kwargs: Any) -> None:
        """Start parent run in MLflow before any job."""
        exp_name = config.hydra.job.env_set.get("MLFLOW_EXPERIMENT_NAME")
        exp_name = exp_name or os.environ.get("MLFLOW_EXPERIMENT_NAME")
        mlflow.set_experiment(exp_name)
        mlflow.start_run(run_name=self.name, description=self.description)
        self._parent_run = mlflow.active_run()

    def on_job_start(
        self, config: DictConfig, *, task_function: TaskFunction, **kwargs: Any
    ) -> None:
        """Ensure that parent run is active before each job during multi-processing."""
        if self._parent_run is None:
            return

        if mlflow.active_run() != self._parent_run:
            mlflow.end_run()
            mlflow.start_run(run_id=self._parent_run.info.run_id)
