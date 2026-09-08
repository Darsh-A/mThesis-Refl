"""Train a Random Forest regressor to predict f_pisn from abundance ratios.

Data generation (expensive ~3s/sample) is run ONCE and cached to disk as an
.npz; training then loads that cached file instead of regenerating.

Usage:
    python src/randomForest/train_rf.py --generate   # build + cache dataset
    python src/randomForest/train_rf.py --train      # train from cached dataset
    python src/randomForest/train_rf.py              # train (errors if no cache)

The dataset cache and the fitted model are written under src/randomForest/data/.
"""

import argparse
import os
import sys

import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.randomForest.dataset import generate

# --- Paths ---
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
CACHE_PATH = os.path.join(DATA_DIR, "dataset.npz")
MODEL_PATH = os.path.join(DATA_DIR, "rf_model.joblib")

CONFIG = {
    # dataset
    "pisn_source": "HW2002",
    "sn_source": "Limongi18",
    "n_samples": 2000,
    "seed": 12345,
    "cache_path": CACHE_PATH,
    # split
    "test_size": 0.2,
    # model
    "n_estimators": 300,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
}


def load_dataset(cache_path: str = CACHE_PATH):
    """Load the cached (X, y, diagnostics) triple from disk.

    Raises FileNotFoundError with a helpful hint if generation hasn't run yet.
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"No dataset cache at {cache_path!r}. "
            f"Run `python {os.path.relpath(__file__)} --generate` first."
        )
    data = np.load(cache_path, allow_pickle=True)
    return data["X"], data["y"], dict(data["diagnostics"].item())


def _impute(X: np.ndarray) -> np.ndarray:
    """Fill NaN feature cells with the column median (SN-safe, deterministic)."""
    imp = SimpleImputer(strategy="median")
    return imp.fit_transform(X)


def generate_dataset(config: dict = CONFIG):
    """Build and cache the dataset once (expensive)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"generating dataset: {config['n_samples']} samples "
          f"[pisn={config['pisn_source']}, sn={config['sn_source']}] ...")
    t0 = __import__("time").time()
    X, y, diag = generate(
        pisn_source=config["pisn_source"],
        sn_source=config["sn_source"],
        n_samples=config["n_samples"],
        seed=config["seed"],
        cache_path=config["cache_path"],
    )
    dt = __import__("time").time() - t0
    print(f"done in {dt:.1f}s -> X={X.shape}, NaN cells={int(np.isnan(X).sum())}")
    return X, y, diag


def train(config: dict = CONFIG):
    X, y, diag = load_dataset(config["cache_path"])
    n_nan = int(np.isnan(X).sum())
    print(f"loaded X={X.shape}, y={y.shape}, NaN cells={n_nan} "
          f"({n_nan / X.size:.2%} of features)")

    X = _impute(X)

    rng = np.random.default_rng(config["seed"])
    n = len(y)
    idx = rng.permutation(n)
    n_test = int(n * config["test_size"])
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    pipe = Pipeline(
        steps=[
            # imputation done eagerly above so both splits share the same medians
            ("rf", RandomForestRegressor(
                n_estimators=config["n_estimators"],
                min_samples_leaf=config["min_samples_leaf"],
                random_state=config["random_state"],
                n_jobs=config["n_jobs"],
            )),
        ]
    )
    pipe.fit(X_train, y_train)

    from sklearn.metrics import r2_score, mean_squared_error
    y_pred_train = pipe.predict(X_train)
    y_pred_test = pipe.predict(X_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))

    print(f"\ntrain R2 = {train_r2:.4f}   RMSE = {train_rmse:.4f}")
    print(f"test  R2 = {test_r2:.4f}   RMSE = {test_rmse:.4f}")
    print(f"target std (test) = {float(np.std(y_test)):.4f}")

    os.makedirs(DATA_DIR, exist_ok=True)
    import joblib
    joblib.dump(
        {"model": pipe, "feature_names": diag["feature_names"], "config": config},
        MODEL_PATH,
    )
    print(f"\nsaved model -> {MODEL_PATH}")

    return pipe, (X, y, diag), (train_r2, test_r2, train_rmse, test_rmse)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "step",
        nargs="?",
        default="train",
        choices=("generate", "train"),
        help="generate = build+cache dataset; train = fit from cache (default)",
    )
    args = parser.parse_args(argv)

    if args.step == "generate":
        generate_dataset()
    else:
        train()


if __name__ == "__main__":
    main()
