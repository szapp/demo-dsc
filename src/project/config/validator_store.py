from hydra_zen import store
from sklearn.model_selection import KFold

validator_store = store(group="validator")

# KFold with 5 shuffled folds and fixed random state
validator_store(
    KFold,
    n_splits=5,
    shuffle=True,
    random_state=42,
    name="5fold",
)
