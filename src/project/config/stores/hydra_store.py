"""Config store for hydra base configuration, e.g. logging, directories."""

from hydra.conf import HydraConf, JobConf, RunDir
from hydra_zen import store

from ...version import SERVICE, VERSION

__all__ = []  # No exports. Use the global store

hydra_store = store()  # Hydra config does not need group or name

# Hydra configuration for all entrypoints
hydra_store(
    HydraConf(
        defaults=HydraConf().defaults
        + [
            {"override job_logging": "colorlog"},
            {"override hydra_logging": "colorlog"},
            {"override launcher": "joblib"},
        ],
        run=RunDir(
            dir=(
                "outputs/${hydra.job.name}-${hydra.job.config_name}/"
                "${now:%Y-%m-%d}/${now:%H-%M-%S}"
            )
        ),
        job=JobConf(
            chdir=True,
            env_set={
                "ENV": "${oc.env:ENV,dev}",
                "MLFLOW_EXPERIMENT_NAME": SERVICE,
                "MLFLOW_TRACKING_URI": "sqlite:///${hydra.runtime.cwd}/mlflow.db",
            },
        ),
        job_logging={
            "formatters": {
                "structured": {
                    "()": "pythonjsonlogger.json.JsonFormatter",
                    "fmt": [
                        "timestamp",
                        "levelname",
                        "message",
                        "name",
                        "funcName",
                        "lineno",
                        # "dd.trace_id",
                        # "dd.span_id",
                    ],
                    "timestamp": True,
                    "rename_fields": {"levelname": "status"},
                    "static_fields": {
                        "dd.version": VERSION,
                    },
                    "exc_info_as_array": False,
                    "stack_info_as_array": False,
                },
            },
            "filters": {
                "context": {
                    "()": "project.util.logging_vars.LoggingVarsFilter",
                }
            },
            "handlers": {
                "file": {
                    "filename": "${hydra.runtime.output_dir}/${hydra.job.name}.log",
                    "encoding": "utf-8",
                },
                "monitoring": {
                    "()": "logging.handlers.RotatingFileHandler",
                    "level": "INFO",
                    "filters": ["context"],
                    "formatter": "structured",
                    "filename": "${hydra.runtime.cwd}/logs/structured.log",
                    "encoding": "utf-8",
                    "maxBytes": 1024**2,  # 1 MB
                    "backupCount": 1,  # Important for safe rotation (structured.log.1)
                },
            },
            "loggers": {
                "mlflow": {"handlers": [], "propagate": True},
                "mlflow.types.type_hints": {"level": "ERROR"},
                "alembic": {"handlers": [], "propagate": True},
                "sqlalchemy.engine": {"handlers": [], "propagate": True},
            },
            "root": {
                "handlers": ["file", "monitoring", "console"],  # Order for coloring
            },
        },
        hydra_logging={
            "loggers": {
                "optuna": {"handlers": [], "level": "INFO", "propagate": True},
            },
        },
    ),
)
