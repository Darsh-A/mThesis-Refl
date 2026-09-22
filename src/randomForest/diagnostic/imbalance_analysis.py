"""Imbalance analysis: does the f_pisn=0 (SN-only) spike bias the model?

The cached dataset is ~50.8% f_pisn=0.  Here we retrain the RF at a range of
zero-fractions (subsampling the f_pisn=0 rows while keeping all f_pisn>0 rows)
and measure, on a FIXED held-out test set:

  * regression quality (R2 / RMSE)
  * the predicted f_pisn distribution (mean, frac <0.1, frac >0.5)

If the f_pisn=0 spike is harmless, the held-out metrics and the predicted
distribution of the *positive* test stars should be stable across zero-fractions.
A large sensitivity indicates the spike is imprinting on the learned mapping.

Run:
    python src/randomForest/imbalance_analysis.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error

from src.randomForest.train_rf import CONFIG, load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG_DIR = os.path.join(HERE, "diagnostic", "new")
os.makedirs(DIAG_DIR, exist_ok=True)


def main():
    X, y, diag = load_dataset(CONFIG["cache_path"])
    X = SimpleImputer(strategy="median").fit_transform(X)

    rng = np.random.default_rng(CONFIG["seed"])
    n = len(y)
    idx = rng.permutation(n)
    n_test = int(n * CONFIG["test_size"])
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    X_te, y_te = X[test_idx], y[test_idx]
    X_tr_all, y_tr_all = X[train_idx], y[train_idx]

    zero_mask_tr = y_tr_all == 0.0
    pos_mask_tr = ~zero_mask_tr
    X_tr_pos = X_tr_all[pos_mask_tr]
    y_tr_pos = y_tr_all[pos_mask_tr]
    X_tr_zero = X_tr_all[zero_mask_tr]
    n_pos = len(y_tr_pos)
    n_zero_avail = len(X_tr_zero)

    # target fractions of f_pisn=0 in the training set
    fractions = [0.0, 0.1, 0.25, 0.508, 0.75, 0.9]
    print(f"train positive count = {n_pos}, zero count available = {n_zero_avail}\n")

    rows = []
    for f0 in fractions:
        # number of zero rows to keep such that n0/(n0+n_pos) = f0
        if f0 <= 0.0:
            n0 = 0
        else:
            n0 = int(round(f0 / (1.0 - f0) * n_pos))
        # bootstrap (with replacement) if we need more zeros than available
        replace = n0 > n_zero_avail
        if n0 > 0:
            pick = rng.choice(n_zero_avail, size=n0, replace=replace)
            X_tr = np.vstack([X_tr_pos, X_tr_zero[pick]])
            y_tr = np.concatenate([y_tr_pos, np.zeros(n0)])
        else:
            X_tr, y_tr = X_tr_pos, y_tr_pos

        rf = RandomForestRegressor(
            n_estimators=CONFIG["n_estimators"],
            min_samples_leaf=CONFIG["min_samples_leaf"],
            random_state=CONFIG["random_state"],
            n_jobs=CONFIG["n_jobs"],
        )
        rf.fit(X_tr, y_tr)

        pred = rf.predict(X_te)
        r2 = r2_score(y_te, pred)
        rmse = float(np.sqrt(mean_squared_error(y_te, pred)))

        # predicted distribution, split by true zero vs positive
        true_zero = y_te == 0.0
        pred_zero = pred[true_zero]
        pred_pos = pred[~true_zero]

        row = dict(
            target_f0=f0,
            actual_f0=(n0 / (n0 + n_pos)),
            n_train=len(y_tr),
            r2=r2,
            rmse=rmse,
            mean_pred_all=float(pred.mean()),
            mean_pred_true_zero=float(pred_zero.mean()) if len(pred_zero) else np.nan,
            mean_pred_true_pos=float(pred_pos.mean()) if len(pred_pos) else np.nan,
            frac_all_below_0_1=float((pred < 0.1).mean()),
        )
        rows.append(row)
        print(
            f"f0_target={f0:>5.2f}  actual_f0={row['actual_f0']:.3f}  "
            f"n_train={row['n_train']:>5}  R2={r2:.4f}  RMSE={rmse:.4f}  "
            f"mean_pred={pred.mean():.3f}  "
            f"mean_pred(pos)={row['mean_pred_true_pos']:.3f}  "
            f"frac(<0.1)={row['frac_all_below_0_1']:.3f}"
        )

    # --- plots ---
    f0s = [r["target_f0"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(f0s, [r["r2"] for r in rows], "o-", color="tab:blue")
    ax.set_xlabel("fraction f_pisn=0 in training")
    ax.set_ylabel("test R2")
    ax.set_title("Held-out R2 vs zero-fraction")

    ax = axes[1]
    ax.plot(f0s, [r["rmse"] for r in rows], "s-", color="tab:red")
    ax.set_xlabel("fraction f_pisn=0 in training")
    ax.set_ylabel("test RMSE")
    ax.set_title("Held-out RMSE vs zero-fraction")

    ax = axes[2]
    ax.plot(f0s, [r["mean_pred_true_pos"] for r in rows], "^-", color="tab:green",
            label="pred mean (true f_pisn>0)")
    ax.plot(f0s, [r["mean_pred_true_zero"] for r in rows], "v--", color="tab:orange",
            label="pred mean (true f_pisn=0)")
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_xlabel("fraction f_pisn=0 in training")
    ax.set_ylabel("predicted f_pisn mean")
    ax.set_title("Predicted mean by true class")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(DIAG_DIR, "imbalance_analysis.png"))
    plt.close(fig)

    print(f"\nsaved plot to {os.path.join(DIAG_DIR, 'imbalance_analysis.png')}")


if __name__ == "__main__":
    main()
