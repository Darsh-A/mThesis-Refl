"""Calibration / coverage evaluation for the quantile regression forest.

Loads data/rf_quantile_model.joblib and the cached dataset, then on the same
held-out split used at training time reports:

  1. Prediction-interval coverage vs nominal level (calibration).
  2. Mean interval width (sharpness).
  3. Pinball (quantile) loss -- a proper scoring rule.
  4. A reliability diagram (nominal vs empirical coverage).
  5. Interval width vs the nuisance parameters (f_ratio, tpop2, pisn_mass).

Run:
    python src/randomForest/evaluate_quantile_rf.py

Plots are written to src/randomForest/diagnostic/.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.randomForest.train_rf import CONFIG, load_dataset, _impute
from src.randomForest.train_quantile_rf import QUANTILE_MODEL_PATH

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG_DIR = os.path.join(HERE, "diagnostic")
os.makedirs(DIAG_DIR, exist_ok=True)

QUANTILES = [0.025, 0.05, 0.16, 0.25, 0.5, 0.75, 0.84, 0.95, 0.975]
# central intervals: (nominal level, lower quantile, upper quantile)
INTERVALS = [
    (0.50, 0.25, 0.75),
    (0.68, 0.16, 0.84),
    (0.90, 0.05, 0.95),
    (0.95, 0.025, 0.975),
]


def pinball(y, q_hat, q):
    diff = y - q_hat
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def main():
    import joblib

    if not os.path.exists(QUANTILE_MODEL_PATH):
        raise FileNotFoundError(
            f"No quantile model at {QUANTILE_MODEL_PATH!r}. "
            f"Run `python src/randomForest/train_quantile_rf.py` first."
        )

    bundle = joblib.load(QUANTILE_MODEL_PATH)
    qrf = bundle["model"]
    cfg = bundle["config"]

    X, y, diag = load_dataset(cfg["cache_path"])
    X = _impute(X)

    # identical split to training
    rng = np.random.default_rng(cfg["seed"])
    n = len(y)
    idx = rng.permutation(n)
    n_test = int(n * cfg["test_size"])
    test_idx = idx[:n_test]
    X_test, y_test = X[test_idx], y[test_idx]

    preds = qrf.predict(X_test, quantiles=QUANTILES)   # (n_test, n_q)
    q_idx = {q: i for i, q in enumerate(QUANTILES)}

    print("=== Prediction-interval coverage (held-out) ===")
    widths = {}
    for level, lo, hi in INTERVALS:
        y_lo = preds[:, q_idx[lo]]
        y_hi = preds[:, q_idx[hi]]
        in_int = (y_test >= y_lo) & (y_test <= y_hi)
        coverage = float(in_int.mean())
        width = float(np.mean(y_hi - y_lo))
        widths[level] = width
        print(f"  {level*100:>5.0f}% CI [{lo:.3f},{hi:.3f}]: "
              f"coverage = {coverage*100:6.2f}%   mean width = {width:.4f}")

    print("\n=== Pinball (quantile) loss ===")
    pinballs = [pinball(y_test, preds[:, q_idx[q]], q) for q in QUANTILES]
    for q, pb in zip(QUANTILES, pinballs):
        print(f"  q={q:5.3f}: {pb:.4f}")
    print(f"  mean pinball = {np.mean(pinballs):.4f}")

    y_med = preds[:, q_idx[0.5]]
    ss_res = np.sum((y_test - y_med) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    print(f"\n  median-prediction R2 = {1 - ss_res / ss_tot:.4f}")
    print(f"  target std (test)    = {float(np.std(y_test)):.4f}")

    # ---- Reliability diagram ----
    nominal = np.arange(0.1, 1.0, 0.05)
    empirical = []
    for p in nominal:
        lo = (1 - p) / 2
        hi = (1 + p) / 2
        y_lo = qrf.predict(X_test, quantiles=lo)
        y_hi = qrf.predict(X_test, quantiles=hi)
        empirical.append(float(np.mean((y_test >= y_lo) & (y_test <= y_hi))))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    ax.plot(nominal, empirical, "o-", color="tab:blue", label="QRF")
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Reliability diagram (held-out)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(DIAG_DIR, "quantile_reliability.png"))
    plt.close(fig)

    # ---- Intervals vs true y (sorted) ----
    y_lo90 = preds[:, q_idx[0.05]]
    y_hi90 = preds[:, q_idx[0.95]]
    s = np.argsort(y_test)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(np.arange(n_test), y_lo90[s], y_hi90[s],
                    color="tab:blue", alpha=0.3, label="90% CI")
    ax.plot(np.arange(n_test), y_med[s], color="tab:blue", lw=1, label="median")
    ax.plot(np.arange(n_test), y_test[s], "k.", ms=3, label="true f_pisn")
    ax.set_xlabel("test sample (sorted by true f_pisn)")
    ax.set_ylabel("f_pisn")
    ax.set_title("90% prediction intervals vs truth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(DIAG_DIR, "quantile_intervals.png"))
    plt.close(fig)

    # ---- Width vs nuisance parameters ----
    w90 = y_hi90 - y_lo90
    nuisance = {k: diag[k][test_idx] for k in ("f_ratio", "tpop2", "pisn_mass")}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (k, v) in zip(axes, nuisance.items()):
        ax.scatter(v, w90, s=6, alpha=0.4, color="tab:green")
        ax.set_xlabel(k)
        ax.set_ylabel("90% CI width")
        corr = np.corrcoef(v, w90)[0, 1]
        ax.set_title(f"width vs {k}  (r = {corr:.2f})")
    fig.suptitle("Where does the uncertainty come from?")
    fig.tight_layout()
    fig.savefig(os.path.join(DIAG_DIR, "quantile_width_vs_nuisance.png"))
    plt.close(fig)

    print(f"\nsaved plots to {DIAG_DIR}")


if __name__ == "__main__":
    main()
