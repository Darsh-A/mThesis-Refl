"""Save held-out performance metrics + confusion matrix for test_v4_3 & test_v5_0.

Reproduces the held-out benchmark from quantile_rf_results_test_v4_3.py /
test_v5_0.py (standard QRF + single-seed mask-aware QRF), but instead of
applying to the observed catalogue it reports the model's own performance on
the synthetic held-out split, where the ground-truth f_pisn is known.

For each model (4_3, 5) and each method (standard, mask-aware) it writes:

  * regression metrics: R2, RMSE, MAE, Spearman rho, bias (mean residual)
  * classification metrics at threshold 0.5 (f_pisn > 0.5 == "PISN-dominated"):
      confusion matrix, accuracy, precision, recall, F1
  * a detection view (f_pisn > 0 == "any PISN") at a 0.05 prediction threshold

Datasets (already cached, no regeneration):
  * 4_3 -> src/randomForest/data/dataset_TakaLimon_realisticNoise_v4_3.npz
  * 5   -> src/randomForest/data/dataset_Nomotov5.npz

Outputs (src/randomForest/results/model_metrics/):
  * metrics_4_3_vs_5.json
  * metrics_4_3_vs_5.txt
  * confusion_matrix.png
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
    r2_score, mean_squared_error, mean_absolute_error,
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
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
from src.load_data import load_obs_li  # noqa: E402

DATA_DIR = os.path.join(ROOT, "src", "randomForest", "data")
OUT_DIR = os.path.join(ROOT, "src", "randomForest", "results", "model_metrics")
os.makedirs(OUT_DIR, exist_ok=True)

OBS_ELEMENTS = ["C", "Na", "Mg", "Si", "Ca", "Cr", "Mn", "Fe", "Co", "Ni", "Zn"]
ELEMENT_ION = {
    "C": "C", "Na": "NaI", "Mg": "MgI", "Si": "SiI", "Ca": "CaI", "Cr": "CrI",
    "Mn": "MnI", "Fe": "FeI", "Co": "CoI", "Ni": "NiI", "Zn": "ZnI",
}
NON_FE = [el for el in OBS_ELEMENTS if el != "Fe"]
PAIRS = list(combinations(OBS_ELEMENTS, 2))
RATIO_NAMES = [f"[{x}/{y}]" for x, y in PAIRS]
EL_TO_RATIOS = {el: [j for j, (x, y) in enumerate(PAIRS) if el in (x, y)] for el in OBS_ELEMENTS}

M_AUG = 5
STRONG_THRESH = 0.5
DETECT_THRESH = 0.05

MODELS = {
    "4_3": {
        "cache": os.path.join(DATA_DIR, "dataset_TakaLimon_realisticNoise_v4_3.npz"),
        "label": "4_3 (Takahashi + Limongi18 + noise)",
    },
    "5": {
        "cache": os.path.join(DATA_DIR, "dataset_Nomotov5.npz"),
        "label": "5 (Takahashi + Nomoto13)",
    },
}


def get_xfe(star, elem):
    if elem == "Fe":
        return 0.0
    rec = star["abundances"].get(ELEMENT_ION[elem])
    if rec is None:
        return np.nan
    if rec.get("l_XFe") == "<":
        return np.nan
    xfe = rec.get("XFe")
    return np.nan if xfe is None else float(xfe)


def subset_columns(X, full_names, subset_names):
    idx = [full_names.index(nm) for nm in subset_names]
    return X[:, idx]


def mask_features(X, rng, freq):
    Xm = X.copy()
    ind = np.ones((Xm.shape[0], len(NON_FE)))
    for i in range(Xm.shape[0]):
        for k, el in enumerate(NON_FE):
            if rng.random() > freq[el]:
                ind[i, k] = 0.0
                for j in EL_TO_RATIOS[el]:
                    Xm[i, j] = np.nan
    return Xm, ind


def augment(X_complete, y, rng, freq, imp, M=M_AUG):
    n = X_complete.shape[0]
    Xs = [imp.transform(X_complete)]
    inds = [np.ones((n, len(NON_FE)))]
    ys = [y]
    for _ in range(M):
        Xm, ind = mask_features(X_complete, rng, freq)
        Xs.append(imp.transform(Xm))
        inds.append(ind)
        ys.append(y)
    return np.hstack([np.vstack(Xs), np.vstack(inds)]), np.concatenate(ys)


def _new_qrf(cfg):
    return QuantileRegressionForest(
        n_estimators=cfg["n_estimators"], min_samples_leaf=cfg["min_samples_leaf"],
        random_state=cfg["random_state"], n_jobs=cfg["n_jobs"])


def class_metrics(y_true, y_pred, thresh, detect):
    """Binary metrics. detect=True -> positive == f_pisn > 0 (any PISN)."""
    if detect:
        yb = (y_true > 0.0).astype(int)
        pb = (y_pred > thresh).astype(int)
    else:
        yb = (y_true > thresh).astype(int)
        pb = (y_pred > thresh).astype(int)
    cm = confusion_matrix(yb, pb, labels=[0, 1]).tolist()
    return {
        "confusion_matrix": cm,  # [[TN, FP], [FN, TP]]
        "accuracy": float(accuracy_score(yb, pb)),
        "precision": float(precision_score(yb, pb, zero_division=0)),
        "recall": float(recall_score(yb, pb, zero_division=0)),
        "f1": float(f1_score(yb, pb, zero_division=0)),
        "n_positive": int(yb.sum()),
        "n_negative": int((1 - yb).sum()),
    }


def regression_metrics(y_true, y_pred):
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    rho, pv = spearmanr(y_true, y_pred)
    return {
        "r2": r2, "rmse": rmse, "mae": mae,
        "spearman_rho": float(rho), "spearman_p": float(pv),
        "bias": float(np.mean(y_pred - y_true)),
        "pred_mean": float(np.mean(y_pred)), "true_mean": float(np.mean(y_true)),
    }


def evaluate_model(model_id):
    spec = MODELS[model_id]
    X_full, y, diag = load_dataset(spec["cache"])
    full_names = list(diag["feature_names"])
    print(f"\n[{model_id}] {spec['label']}")
    print(f"  cache={os.path.basename(spec['cache'])}  X={X_full.shape}  "
          f"sn={diag.get('sn_source')}  pisn={diag.get('pisn_source')}")

    X_obs_full = subset_columns(X_full, full_names, RATIO_NAMES)
    imp = SimpleImputer(strategy="median").fit(X_obs_full)
    X_imp = imp.transform(X_obs_full)

    stars = load_obs_li()
    freq = {el: float(np.mean([not np.isnan(get_xfe(s, el)) for s in stars])) for el in NON_FE}

    cfg = CONFIG
    rng = np.random.default_rng(cfg["seed"])
    n = len(y)
    idx = rng.permutation(n)
    n_test = int(n * cfg["test_size"])
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    # standard QRF
    qrf_std = _new_qrf(cfg)
    qrf_std.fit(X_imp[train_idx], y[train_idx])
    y_std = qrf_std.predict(X_imp[test_idx], 0.5)

    # mask-aware QRF (single augmented seed, masked test set)
    rng_mask = np.random.default_rng(777)
    X_tr_aug, y_tr_aug = augment(X_obs_full[train_idx], y[train_idx], rng_mask, freq, imp)
    qrf_mask = _new_qrf(cfg)
    qrf_mask.fit(X_tr_aug, y_tr_aug)
    X_te_mask, ind_te = mask_features(X_obs_full[test_idx], rng_mask, freq)
    X_te_feat = np.hstack([imp.transform(X_te_mask), ind_te])
    y_mask = qrf_mask.predict(X_te_feat, 0.5)

    y_true = y[test_idx]
    out = {"label": spec["label"], "n_test": int(n_test)}
    for method, y_pred in [("standard", y_std), ("mask_aware", y_mask)]:
        out[method] = {
            "regression": regression_metrics(y_true, y_pred),
            "strong_pisn_thr0.5": class_metrics(y_true, y_pred, STRONG_THRESH, detect=False),
            "detection_thr0.05": class_metrics(y_true, y_pred, DETECT_THRESH, detect=True),
        }
        print(f"  {method:10s}  R2={out[method]['regression']['r2']:.4f}  "
              f"RMSE={out[method]['regression']['rmse']:.4f}  "
              f"rho={out[method]['regression']['spearman_rho']:+.3f}  "
              f"F1(>0.5)={out[method]['strong_pisn_thr0.5']['f1']:.3f}")
    return out


def plot_confusion(all_cms):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for r, model_id in enumerate(["4_3", "5"]):
        for c, method in enumerate(["standard", "mask_aware"]):
            cm = np.array(all_cms[model_id][method])
            ax = axes[r][c]
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["SN-dominated\n(f<=0.5)", "PISN-dominated\n(f>0.5)"])
            ax.set_yticklabels(["SN-dominated\n(f<=0.5)", "PISN-dominated\n(f>0.5)"])
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black",
                            fontsize=14, fontweight="bold")
            ax.set_title(f"{model_id} — {method}", fontsize=11)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Confusion matrix: PISN-dominated (f_pisn > 0.5) vs SN-dominated",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT_DIR, "confusion_matrix.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nsaved {out}")


def write_outputs(metrics):
    json_path = os.path.join(OUT_DIR, "metrics_4_3_vs_5.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved {json_path}")

    txt_path = os.path.join(OUT_DIR, "metrics_4_3_vs_5.txt")
    lines = ["Held-out performance: test_v4_3 vs test_v5_0",
             "Positive class = f_pisn > 0.5 (PISN-dominated).",
             "Detection view = f_pisn > 0 (any PISN), pred threshold 0.05.",
             ""]
    for model_id in ["4_3", "5"]:
        lines.append(f"== {model_id}: {metrics[model_id]['label']} "
                     f"(n_test={metrics[model_id]['n_test']}) ==")
        for method in ["standard", "mask_aware"]:
            m = metrics[model_id][method]
            g = m["regression"]
            s = m["strong_pisn_thr0.5"]
            d = m["detection_thr0.05"]
            lines.append(f"  [{method}]")
            lines.append(f"    R2={g['r2']:.4f}  RMSE={g['rmse']:.4f}  MAE={g['mae']:.4f}  "
                         f"rho={g['spearman_rho']:+.3f}  bias={g['bias']:+.3f}")
            lines.append(f"    strong(>0.5) CM [[TN,FP],[FN,TP]]={s['confusion_matrix']}  "
                         f"acc={s['accuracy']:.3f} prec={s['precision']:.3f} "
                         f"rec={s['recall']:.3f} F1={s['f1']:.3f}")
            lines.append(f"    detection(>0) CM={d['confusion_matrix']}  "
                         f"acc={d['accuracy']:.3f} prec={d['precision']:.3f} "
                         f"rec={d['recall']:.3f} F1={d['f1']:.3f}")
        lines.append("")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved {txt_path}")


def main():
    metrics = {}
    all_cms = {}
    for model_id in ["4_3", "5"]:
        out = evaluate_model(model_id)
        metrics[model_id] = out
        all_cms[model_id] = {
            "standard": out["standard"]["strong_pisn_thr0.5"]["confusion_matrix"],
            "mask_aware": out["mask_aware"]["strong_pisn_thr0.5"]["confusion_matrix"],
        }
    plot_confusion(all_cms)
    write_outputs(metrics)
    print(f"\ndone. outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
