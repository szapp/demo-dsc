from hydra.conf import HydraConf
from hydra_zen import store


def register_base_config():
    store(
        HydraConf(
            defaults=HydraConf().defaults
            + [
                {"override job_logging": "colorlog"},  # Changes output file (see below)
                {"override hydra_logging": "colorlog"},
                {"override launcher": "joblib"},
                {"override sweeper": "optuna"},
            ],
            job_logging={
                "handlers": {
                    "file": {
                        "filename": "${hydra.runtime.output_dir}/${hydra.job.name}.log",  # Default
                    },
                },
                "loggers": {
                    "mlflow": {"handlers": [], "level": "WARNING", "propagate": True},
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
        group="hydra",
        name="config",
    )
