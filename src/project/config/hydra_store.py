from hydra.conf import HydraConf
from hydra_zen import store

hydra_store = store()  # Hydra config does not need group or name

hydra_store(
    HydraConf(
        defaults=HydraConf().defaults
        + [
            {"override job_logging": "colorlog"},  # Changes output file (see below)
            {"override hydra_logging": "colorlog"},
            {"override launcher": "joblib"},
            {"override sweeper": "optuna"},
        ],
        job_logging={
            "formatters": {
                "debug": {
                    "format": "[%(asctime)s][%(pathname)s:%(lineno)s][%(levelname)s] - %(message)s"
                },
            },
            "handlers": {
                "file": {
                    "filename": "${hydra.runtime.output_dir}/${hydra.job.name}.log",  # Default
                },
            },
            "loggers": {
                "mlflow": {"handlers": [], "level": "WARNING", "propagate": True},
                "mlflow.types.type_hints": {"level": "ERROR"},
                "alembic": {"handlers": [], "level": "WARNING", "propagate": True},
                "sqlalchemy.engine": {"handlers": [], "propagate": True},
            },
            "root": {
                "handlers": ["file", "console"],  # Order for coloring
                "level": "${oc.select:log_level,INFO}",
            },
        },
        hydra_logging={
            "loggers": {
                "optuna": {"handlers": [], "level": "INFO", "propagate": True},
            },
        },
        sweeper={
            "sampler": {
                "seed": 42,
            },
            "direction": "maximize",  # Scikit-learn always maximizes scores
            "storage": "sqlite:///data/optuna.db",
        },
    ),
)
