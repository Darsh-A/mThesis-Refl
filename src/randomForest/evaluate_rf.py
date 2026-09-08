"""Evaluate a trained Random Forest model from train_rf.py.

Loads the fitted pipeline saved to data/rf_model.joblib and reports:
    1. Basic model info (config, n_samples used, feature count).
    2. Feature importances (top N + least N), ranked by Gini importance.
    3. A comparison of the model against the two "null" markers f_pisn=0 and
       f_pisn=1 (i.e. pure SN vs pure PISN) to see whether the ratios at the
       extremes are learned.
    4. Optional: re-run train/validation performance via cross-validation on the
       cached dataset (if present) so evaluate gives a stable score estimate.

Run:
    python src/randomForest/evaluate_rf.py

Assumes `python src/randomForest/train_rf.py --train` has already produced
data/rf_model.joblib (and, for cross-validation, data/dataset.npz).
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
MODEL_PATH = os.path.join(DATA_DIR, "rf_model.joblib")
CACHE_PATH = os.path.join(DATA_DIR, "dataset.npz")


def load_model(model_path: str = MODEL_PATH):
    import joblib
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No model at {model_path!r}. "
            f"Run `python {os.path.relpath(os.path.join(HERE, 'train_rf.py'))} --train` first."
        )
    return joblib.load(model_path)


def feature_importances(bundle: dict, top_n: int = 20):
    pipe = bundle["model"]
    rf = pipe.named_steps["rf"]
    names = bundle["feature_names"]
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1]

    print(f"\n=== Feature importances (top {top_n}) ===")
    for rank, i in enumerate(order[:top_n], start=1):
        print(f"  {rank:>3}. {names[i]:<12} {importances[i]:.4f}")

    print(f"\n=== Least important ({top_n}) ===")
    for rank, i in enumerate(order[-top_n:][::-1], start=1):
        print(f"  {rank:>3}. {names[i]:<12} {importances[i]:.4f}")


def extreme_probe(bundle: dict):
    """Compare the model's prediction at f_pisn extremes using the data it saw."""
    pipe = bundle["model"]
    if not os.path.exists(CACHE_PATH):
        print("\n(skipping extreme-probe: no dataset.npz cache)")
        return
    data = np.load(CACHE_PATH, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    # impute identically to training (median)
    from sklearn.impute import SimpleImputer
    X = SimpleImputer(strategy="median").fit_transform(X)

    pred = pipe.predict(X)
    print(f"\n=== Prediction vs truth overview ===")
    print(f"  y (truth):   min={y.min():.3f}  max={y.max():.3f}  mean={y.mean():.3f}")
    print(f"  y_pred:      min={pred.min():.3f}  max={pred.max():.3f}  mean={pred.mean():.3f}")
    print(f"  corr(pred, y) = {np.corrcoef(pred, y)[0, 1]:.4f}")


def cross_validate(bundle: dict, cv: int = 5):
    """Stable score estimate via K-fold CV on the cached dataset."""
    if not os.path.exists(CACHE_PATH):
        print("\n(skipping CV: no dataset.npz cache)")
        return
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score

    data = np.load(CACHE_PATH, allow_pickle=True)
    X = SimpleImputer(strategy="median").fit_transform(data["X"])
    y = data["y"]

    cfg = bundle["config"]
    rf = RandomForestRegressor(
        n_estimators=cfg["n_estimators"],
        min_samples_leaf=cfg["min_samples_leaf"],
        random_state=cfg["random_state"],
        n_jobs=cfg["n_jobs"],
    )
    scores = cross_val_score(rf, X, y, cv=cv, scoring="r2", n_jobs=cfg["n_jobs"])
    print(f"\n=== {cv}-fold CV R2 ===")
    print(f"  mean = {scores.mean():.4f}  std = {scores.std():.4f}")
    print(f"  folds = {[f'{s:.3f}' for s in scores]}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=20, help="importances to show")
    parser.add_argument("--cv", type=int, default=5, help="CV folds (0 to skip)")
    parser.add_argument("--model", default=MODEL_PATH, help="path to joblib model")
    args = parser.parse_args(argv)

    bundle = load_model(args.model)
    cfg = bundle["config"]
    print("=== Model info ===")
    print(f"  config: { {k: v for k, v in cfg.items() if k != 'cache_path'} }")
    print(f"  n_features: {len(bundle['feature_names'])}")

    feature_importances(bundle, top_n=args.top_n)
    extreme_probe(bundle)
    if args.cv > 0:
        cross_validate(bundle, cv=args.cv)


if __name__ == "__main__":
    main()
