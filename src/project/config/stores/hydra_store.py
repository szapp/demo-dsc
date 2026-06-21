"""Config store for hydra base configuration, e.g. logging, directories."""

from hydra.conf import HydraConf, JobConf, RunDir, SweepDir
from hydra_zen import store

from ...version import SERVICE, VERSION

__all__ = []  # No exports. Use the global store

hydra_store = store()  # Hydra config does not need group or name

# Hydra configuration for all entrypoints
hydra_store(
    HydraConf(
        defaults=HydraConf().defaults
        + [
            {"override job_logging": "structlog"},
            {"override hydra_logging": "colorlog"},
            {"override launcher": "joblib"},
        ],
        run=RunDir(
            dir=(
                "outputs/${hydra.job.name}-${hydra.job.config_name}/"
                "${now:%Y-%m-%d}/${now:%H-%M-%S}"
            )
        ),
        sweep=SweepDir(dir="${hydra.run.dir}"),
        job=JobConf(
            chdir=True,
            env_set={
                "ENV": "${oc.env:ENV,dev}",
                "SERVICE": SERVICE,
                "VERSION": VERSION,
                "MLFLOW_CONFIGURE_LOGGING": "0",
                "MLFLOW_EXPERIMENT_NAME": SERVICE,
                "MLFLOW_TRACKING_URI": "sqlite:///${hydra.runtime.cwd}/mlflow.db",
                "_MLFLOW_SERVER_ARTIFACT_ROOT": "${hydra.runtime.cwd}/mlruns",
            },
        ),
        job_logging={
            "handlers": {
                "json": {
                    "filename": "${hydra.runtime.cwd}/logs/structured.log",
                },
            },
            "loggers": {
                "mlflow": {"handlers": [], "level": "WARNING", "propagate": True},
                "mlflow.types.type_hints": {"level": "ERROR"},
                "alembic": {"handlers": [], "level": "WARNING", "propagate": True},
                "joblib_typed_cache": {"level": "DEBUG"},
                "sqlalchemy.engine": {"handlers": [], "propagate": True},
            },
            "root": {"handlers": ["file", "json"]},
        },
        hydra_logging={
            "loggers": {
                "optuna": {"handlers": [], "level": "INFO", "propagate": True},
            },
        },
    ),
)

# Additional console logging handler for development
store(
    ["json", "file", "console"],
    group="hydra/job_logging/root/handlers",
    name="console",
)
