"""Train a quantile regression forest (QRF) to predict f_pisn with uncertainty.

Companion to train_rf.py: instead of a point prediction, this fits a
QuantileRegressionForest (src.randomForest.quantile_rf) so downstream code can
read off arbitrary quantiles / prediction intervals of f_pisn.

Uses the SAME cached dataset, imputation, and train/test split as train_rf.py
so the median prediction is directly comparable to the point RF.

Usage:
    python src/randomForest/train_quantile_rf.py

Writes data/rf_quantile_model.joblib.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.randomForest.train_rf import CONFIG, load_dataset, _impute, DATA_DIR
from src.randomForest.quantile_rf import QuantileRegressionForest

QUANTILE_MODEL_PATH = os.path.join(DATA_DIR, "rf_quantile_model.joblib")


def train(config: dict = CONFIG):
    X, y, diag = load_dataset(config["cache_path"])
    n_nan = int(np.isnan(X).sum())
    print(f"loaded X={X.shape}, y={y.shape}, NaN cells={n_nan} "
          f"({n_nan / X.size:.2%} of features)")

    X = _impute(X)

    # identical split to train_rf.py for apples-to-apples comparison
    rng = np.random.default_rng(config["seed"])
    n = len(y)
    idx = rng.permutation(n)
    n_test = int(n * config["test_size"])
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    qrf = QuantileRegressionForest(
        n_estimators=config["n_estimators"],
        min_samples_leaf=config["min_samples_leaf"],
        random_state=config["random_state"],
        n_jobs=config["n_jobs"],
    )
    qrf.fit(X_train, y_train)

    import joblib
    os.makedirs(DATA_DIR, exist_ok=True)
    joblib.dump(
        {"model": qrf, "feature_names": diag["feature_names"], "config": config},
        QUANTILE_MODEL_PATH,
    )
    print(f"\nsaved quantile model -> {QUANTILE_MODEL_PATH}")
    return qrf, (X, y, diag)


if __name__ == "__main__":
    train()
