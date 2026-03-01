"""Config store for hydra base configuration, e.g. logging, directories."""

__all__ = []  # No exports. Use the global store

from hydra.conf import HydraConf, JobConf, RunDir
from hydra_zen import store

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
        run=RunDir(dir="outputs/${exp_name}/${now:%Y-%m-%d}/${now:%H-%M-%S}"),
        job=JobConf(name="${exp_name}", chdir=True),
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
                    "exc_info_as_array": False,
                    "stack_info_as_array": False,
                },
            },
            "handlers": {
                "file": {
                    "filename": "${hydra.runtime.output_dir}/${hydra.job.name}.log",
                    "encoding": "utf-8",
                },
                "monitoring": {
                    "()": "logging.handlers.RotatingFileHandler",
                    "level": "INFO",
                    "formatter": "structured",
                    "filename": "${hydra.runtime.cwd}/logs/structured.log",
                    "encoding": "utf-8",
                    "maxBytes": 1024**2,  # 1 MB
                    "backupCount": 1,  # Important for safe rotation (structured.log.1)
                },
            },
            "loggers": {
                "mlflow": {"handlers": [], "level": "WARNING", "propagate": True},
                "mlflow.types.type_hints": {"level": "ERROR"},
                "alembic": {"handlers": [], "level": "WARNING", "propagate": True},
                "sqlalchemy.engine": {"handlers": [], "propagate": True},
                "urllib3": {"level": "WARNING"},
                "git": {"level": "WARNING"},
            },
            "root": {
                "handlers": ["file", "monitoring", "console"],  # Order for coloring
                "level": "${oc.select:log_level,INFO}",
            },
        },
        hydra_logging={
            "loggers": {
                "optuna": {"handlers": [], "level": "INFO", "propagate": True},
            },
        },
    ),
)
