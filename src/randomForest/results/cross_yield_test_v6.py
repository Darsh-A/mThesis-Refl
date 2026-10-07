"""Cross-yield generalization test (test_v6).

Does the learned ratio -> f_pisn mapping transfer across yield models?

Trains a standard QRF on one yield source's synthetic dataset and evaluates it
on the OTHER source's dataset (where the ground-truth f_pisn is still known,
but the abundance ratios come from a different SNII yield grid).

Cells:
    train 4_3 -> test 4_3   (within, held-out)     train 4_3 -> test 5   (cross)
    train 5   -> test 4_3   (cross)                train 5   -> test 5   (within)

Datasets (already cached, no regeneration):
  * 4_3 -> dataset_TakaLimon_realisticNoise_v4_3.npz  (Takahashi + Limongi18 + noise)
  * 5   -> dataset_Nomotov5.npz                       (Takahashi + Nomoto13, clean)

Uses the same 55 [X/Y] ratios the deployed model sees (11 Li+22-observable
elements). Standard QRF (complete synthetic ratios, no missingness) so the
yield-source effect is isolated from the mask-aware missingness handling.

Outputs (src/randomForest/results/test_v6/):
  * cross_yield_metrics.json / .txt
  * cross_yield_scatter.png    (2x2 transfer matrix of pred vs true)
  * cross_yield_confusion.png  (confusion matrices for the two cross cells)
"""

import json
import os
import sys
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    r2_score, mean_squared_error, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
)
from scipy.stats import spearmanr


def _find_root(start: str) -> str:
    d = os.path.abspath(start)
    for _ in range(12):
        if os.path.exists(os.path.join(d, "params.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    raise RuntimeError("Could not locate repo root (params.py not found above cwd)")


ROOT = _find_root(os.getcwd())
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.randomForest.quantile_rf import QuantileRegressionForest  # noqa: E402
from src.randomForest.train_rf import CONFIG, load_dataset  # noqa: E402

DATA_DIR = os.path.join(ROOT, "src", "randomForest", "data")
OUT_DIR = os.path.join(ROOT, "src", "randomForest", "results", "test_v6")
os.makedirs(OUT_DIR, exist_ok=True)

OBS_ELEMENTS = ["C", "Na", "Mg", "Si", "Ca", "Cr", "Mn", "Fe", "Co", "Ni", "Zn"]
PAIRS = list(combinations(OBS_ELEMENTS, 2))
RATIO_NAMES = [f"[{x}/{y}]" for x, y in PAIRS]

THRESH = 0.5

SOURCES = {
    "4_3": os.path.join(DATA_DIR, "dataset_TakaLimon_realisticNoise_v4_3.npz"),
    "5": os.path.join(DATA_DIR, "dataset_Nomotov5.npz"),
}


def load_55(cache):
    X_full, y, diag = load_dataset(cache)
    full = list(diag["feature_names"])
    X = X_full[:, [full.index(n) for n in RATIO_NAMES]]
    X = SimpleImputer(strategy="median").fit_transform(X)
    return X, y


def _new_qrf(cfg):
    return QuantileRegressionForest(
        n_estimators=cfg["n_estimators"], min_samples_leaf=cfg["min_samples_leaf"],
        random_state=cfg["random_state"], n_jobs=cfg["n_jobs"])


def evaluate(y_true, y_pred):
    cm = confusion_matrix((y_true > THRESH).astype(int),
                          (y_pred > THRESH).astype(int), labels=[0, 1]).tolist()
    rho, pv = spearmanr(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "spearman_rho": float(rho),
        "bias": float(np.mean(y_pred - y_true)),
        "confusion_matrix": cm,
        "accuracy": float(accuracy_score((y_true > THRESH).astype(int),
                                         (y_pred > THRESH).astype(int))),
        "precision": float(precision_score((y_true > THRESH).astype(int),
                                           (y_pred > THRESH).astype(int), zero_division=0)),
        "recall": float(recall_score((y_true > THRESH).astype(int),
                                     (y_pred > THRESH).astype(int), zero_division=0)),
        "f1": float(f1_score((y_true > THRESH).astype(int),
                             (y_pred > THRESH).astype(int), zero_division=0)),
    }


def main():
    X43, y43 = load_55(SOURCES["4_3"])
    X5, y5 = load_55(SOURCES["5"])
    cfg = CONFIG

    rng = np.random.default_rng(cfg["seed"])
    n = len(y43)
    idx = rng.permutation(n)
    nt = int(n * cfg["test_size"])
    te43, tr43 = idx[:nt], idx[nt:]
    idx5 = np.random.default_rng(cfg["seed"]).permutation(len(y5))
    te5, tr5 = idx5[:nt], idx5[nt:]

    results = {}
    preds = {}

    # within 4_3 (held-out)
    q = _new_qrf(cfg); q.fit(X43[tr43], y43[tr43])
    preds["43_43"] = q.predict(X43[te43], 0.5)
    results["train4_3_test4_3"] = evaluate(y43[te43], preds["43_43"])

    # cross: train 4_3 (full) -> test 5 (full)
    q = _new_qrf(cfg); q.fit(X43, y43)
    preds["43_5"] = q.predict(X5, 0.5)
    results["train4_3_test5"] = evaluate(y5, preds["43_5"])

    # within 5 (held-out)
    q = _new_qrf(cfg); q.fit(X5[tr5], y5[tr5])
    preds["5_5"] = q.predict(X5[te5], 0.5)
    results["train5_test5"] = evaluate(y5[te5], preds["5_5"])

    # cross: train 5 (full) -> test 4_3 (full)
    q = _new_qrf(cfg); q.fit(X5, y5)
    preds["5_43"] = q.predict(X43, 0.5)
    results["train5_test4_3"] = evaluate(y43, preds["5_43"])

    print("\n=== Cross-yield generalization (test_v6) ===")
    print(f"{'cell':<22}{'R2':>9}{'RMSE':>9}{'rho':>8}{'F1(>0.5)':>10}")
    for k, m in results.items():
        print(f"{k:<22}{m['r2']:>9.4f}{m['rmse']:>9.4f}{m['spearman_rho']:>+8.3f}{m['f1']:>10.3f}")

    # ---- scatter transfer matrix ----
    cells = [
        ("train 4_3 -> test 4_3\n(within, held-out)", "43_43", y43[te43]),
        ("train 4_3 -> test 5\n(cross)", "43_5", y5),
        ("train 5 -> test 4_3\n(cross)", "5_43", y43),
        ("train 5 -> test 5\n(within, held-out)", "5_5", y5[te5]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, (title, key, yt) in zip(axes.ravel(), cells):
        yp = preds[key]
        ax.scatter(yt, yp, s=8, alpha=0.35, color="tab:blue")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("true f_pisn")
        ax.set_ylabel("predicted f_pisn")
        r2 = results[key.replace("43_43", "train4_3_test4_3")
                        .replace("43_5", "train4_3_test5")
                        .replace("5_43", "train5_test4_3")
                        .replace("5_5", "train5_test5")]["r2"]
        ax.set_title(f"{title}\nR2 = {r2:.4f}", fontsize=10)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
    fig.suptitle("Cross-yield generalization: pred vs true f_pisn", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT_DIR, "cross_yield_scatter.png"), dpi=130)
    plt.close(fig)

    # ---- confusion matrices for the two cross cells ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (key, yt) in zip(axes, [("43_5", y5), ("5_43", y43)]):
        cm = np.array(results[key.replace("43_5", "train4_3_test5")
                               .replace("5_43", "train5_test4_3")]["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["SN-dom\n(f<=0.5)", "PISN-dom\n(f>0.5)"])
        ax.set_yticklabels(["SN-dom\n(f<=0.5)", "PISN-dom\n(f>0.5)"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=14, fontweight="bold")
        label = "train 4_3 -> test 5" if key == "43_5" else "train 5 -> test 4_3"
        f1 = results[key.replace("43_5", "train4_3_test5")
                       .replace("5_43", "train5_test4_3")]["f1"]
        ax.set_title(f"{label}  (F1={f1:.3f})")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Cross-yield confusion matrix (f_pisn > 0.5)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT_DIR, "cross_yield_confusion.png"), dpi=130)
    plt.close(fig)

    with open(os.path.join(OUT_DIR, "cross_yield_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    lines = ["Cross-yield generalization (test_v6)",
             "Positive class = f_pisn > 0.5.", ""]
    for k, m in results.items():
        lines.append(f"{k}: R2={m['r2']:.4f}  RMSE={m['rmse']:.4f}  "
                     f"rho={m['spearman_rho']:+.3f}  bias={m['bias']:+.3f}  "
                     f"F1={m['f1']:.3f}  acc={m['accuracy']:.3f}")
        lines.append(f"    CM [[TN,FP],[FN,TP]]={m['confusion_matrix']}")
    with open(os.path.join(OUT_DIR, "cross_yield_metrics.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\ndone. outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
