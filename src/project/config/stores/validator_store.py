"""Config store for ML validators (e.g. cross-validation)."""

from hydra_zen import store
from sklearn.model_selection import KFold

__all__ = []  # No exports. Use the global store

validator_store = store(group="validator")

# KFold with 5 shuffled folds and fixed random state
validator_store(
    KFold,
    n_splits=5,
    shuffle=True,
    random_state=42,
    name="5fold",
)
