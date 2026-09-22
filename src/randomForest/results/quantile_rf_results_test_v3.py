"""
Quantile Random Forest on f_pisn: training results + evaluation on Li+22.

This script (and its companion notebook) does:

  1. Load the cached f_pisn training dataset (14 elements -> 91 [X/Y] ratios).
  2. Load (or train) the full Quantile Regression Forest (QRF) and report its
     held-out performance: median-prediction R2/RMSE, prediction-interval
     coverage, and pinball loss.
  3. Report feature importances (pairwise ratio + element-level) for the full
     model.
  4. Train a *reduced* 11-element QRF restricted to the ratios observable in
     the Li+22 very-metal-poor star catalogue, then predict f_pisn (with
     68%/90% intervals) for the observed stars.
  5. Produce plots and export one row per star to CSV.

All outputs are written to src/randomForest/results/.
"""

import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
import joblib

import matplotlib

# In a notebook we keep the inline backend; headless we fall back to Agg.
try:
    get_ipython()  # noqa: F821
    _IN_NOTEBOOK = True
except NameError:
    _IN_NOTEBOOK = False
if not _IN_NOTEBOOK:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import r2_score, mean_squared_error  # noqa: E402


# --- repo-root / path setup (works from any cwd, incl. Jupyter) -------------
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

# --- paths -------------------------------------------------------------------
DATA_DIR = os.path.join(ROOT, "src", "randomForest", "data")
RESULTS_DIR = os.path.join(ROOT, "src", "randomForest", "results", "test_v3")
CACHE_PATH = os.path.join(DATA_DIR, "dataset.npz")
QUANTILE_MODEL_PATH = os.path.join(DATA_DIR, "rf_quantile_model.joblib")
CSV_PATH = os.path.join(RESULTS_DIR, "obs_li_predictions.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

QUANTILES = [0.025, 0.05, 0.16, 0.25, 0.5, 0.75, 0.84, 0.95, 0.975]
# central intervals: (nominal level, lower quantile, upper quantile)
INTERVALS = [
    (0.50, 0.25, 0.75),
    (0.68, 0.16, 0.84),
    (0.90, 0.05, 0.95),
    (0.95, 0.025, 0.975),
]

# Elements observed in Li+22 (subset of the 14 training elements).
OBS_ELEMENTS = ["C", "Na", "Mg", "Si", "Ca", "Cr", "Mn", "Fe", "Co", "Ni", "Zn"]
ELEMENT_ION = {
    "C": "C", "Na": "NaI", "Mg": "MgI", "Si": "SiI", "Ca": "CaI", "Cr": "CrI",
    "Mn": "MnI", "Fe": "FeI", "Co": "CoI", "Ni": "NiI", "Zn": "ZnI",
}


# --- helpers -----------------------------------------------------------------
def pinball(y, q_hat, q):
    diff = y - q_hat
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def element_importances(names, imps):
    """Attribute each [X/Y] importance to both elements and sum."""
    d = {}
    for nm, imp in zip(names, imps):
        x, y = nm.strip("[]").split("/")
        d[x] = d.get(x, 0.0) + float(imp)
        d[y] = d.get(y, 0.0) + float(imp)
    return d


def obs_feature_names():
    return [f"[{x}/{y}]" for x, y in combinations(OBS_ELEMENTS, 2)]


def get_xfe(star, elem):
    """Star's [X/Fe] (XFe), or NaN if missing/upper-limit. Fe is reference (0)."""
    if elem == "Fe":
        return 0.0
    rec = star["abundances"].get(ELEMENT_ION[elem])
    if rec is None:
        return np.nan
    if rec.get("l_XFe") == "<":          # upper limit -> treat as missing
        return np.nan
    xfe = rec.get("XFe")
    return np.nan if xfe is None else float(xfe)


def build_obs_features(stars):
    """Build (X_obs, feh, teff, logg); NaN where an element is missing."""
    names = obs_feature_names()
    n = len(stars)
    X = np.full((n, len(names)), np.nan)
    feh = np.full(n, np.nan)
    teff = np.full(n, np.nan)
    logg = np.full(n, np.nan)
    for i, star in enumerate(stars):
        xfe = {el: get_xfe(star, el) for el in OBS_ELEMENTS}
        for j, (x, y) in enumerate(combinations(OBS_ELEMENTS, 2)):
            vx, vy = xfe[x], xfe[y]
            X[i, j] = np.nan if (np.isnan(vx) or np.isnan(vy)) else vx - vy
        p = star.get("params", {})
        feh[i] = p.get("FeH", np.nan)
        teff[i] = p.get("Teff", np.nan)
        logg[i] = p.get("logg", np.nan)
    return X, feh, teff, logg


def subset_columns(X, full_names, subset_names):
    idx = [full_names.index(nm) for nm in subset_names]
    return X[:, idx]


def _save(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  saved {path}")


# ============================================================================
# 1. Load the cached training dataset
# ============================================================================
X_full, y, diag = load_dataset(CACHE_PATH)
full_names = list(diag["feature_names"])
n_nan = int(np.isnan(X_full).sum())
print(f"training dataset: X={X_full.shape}, y={y.shape}, "
      f"NaN cells={n_nan} ({n_nan / X_full.size:.2%})")
print(f"feature_names[0] = {full_names[0]}  ...  {len(full_names)} features")


# ============================================================================
# 2. Load (or train) the full QRF
# ============================================================================
# Always train the full QRF fresh from the current dataset so the held-out
# numbers below are self-consistent (the cached rf_quantile_model.joblib may
# predate the latest dataset.npz).
cfg = CONFIG

# impute + identical train/test split used at training time
X_full_imp = SimpleImputer(strategy="median").fit_transform(X_full)
rng = np.random.default_rng(cfg["seed"])
n = len(y)
idx = rng.permutation(n)
n_test = int(n * cfg["test_size"])
test_idx, train_idx = idx[:n_test], idx[n_test:]

X_test, y_test = X_full_imp[test_idx], y[test_idx]
X_train_full, y_train_full = X_full_imp[train_idx], y[train_idx]

qrf = QuantileRegressionForest(
    n_estimators=cfg["n_estimators"],
    min_samples_leaf=cfg["min_samples_leaf"],
    random_state=cfg["random_state"],
    n_jobs=cfg["n_jobs"],
)
qrf.fit(X_train_full, y_train_full)
print(f"trained full QRF fresh ({qrf.rf.n_estimators} trees, "
      f"min_samples_leaf={cfg['min_samples_leaf']}, {len(full_names)} features)")


# ============================================================================
# 3. Held-out performance of the full QRF
# ============================================================================
preds = qrf.predict(X_test, quantiles=QUANTILES)   # (n_test, n_q)
q_idx = {q: i for i, q in enumerate(QUANTILES)}
y_med = preds[:, q_idx[0.5]]

r2 = r2_score(y_test, y_med)
rmse = float(np.sqrt(mean_squared_error(y_test, y_med)))
print(f"\n=== Full QRF held-out performance (N={len(y_test)}) ===")
print(f"  median-prediction R2   = {r2:.4f}")
print(f"  median-prediction RMSE = {rmse:.4f}")
print(f"  target std (test)      = {float(np.std(y_test)):.4f}")

print("\n  prediction-interval coverage (nominal vs empirical):")
empirical = {}
for level, lo, hi in INTERVALS:
    y_lo = preds[:, q_idx[lo]]
    y_hi = preds[:, q_idx[hi]]
    cov = float(np.mean((y_test >= y_lo) & (y_test <= y_hi)))
    width = float(np.mean(y_hi - y_lo))
    empirical[level] = cov
    print(f"    {level*100:>5.0f}% CI [{lo:.3f},{hi:.3f}]: "
          f"coverage = {cov*100:6.2f}%   mean width = {width:.4f}")

print("\n  pinball (quantile) loss:")
pinballs = [pinball(y_test, preds[:, q_idx[q]], q) for q in QUANTILES]
for q, pb in zip(QUANTILES, pinballs):
    print(f"    q={q:5.3f}: {pb:.4f}")
print(f"    mean pinball = {np.mean(pinballs):.4f}")


# ============================================================================
# 3b. Binary classification view: PISN-enriched vs not
# ============================================================================
# Accuracy / precision / recall / F1 / confusion matrix all need ground-truth
# labels, which only exist for the synthetic held-out split (observed stars
# have no known true f_pisn). We binarize f_pisn: > 0 -> "PISN present",
# == 0 -> "no PISN" (the dataset draws f_pisn = 0 for pure SNII and
# f_pisn in [0.1, 1] otherwise), and threshold the median prediction at 0.05
# (halfway between those two regimes).
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, average_precision_score,
    confusion_matrix, classification_report,
)

THRESHOLD = 0.05
y_bin = (y_test > 0.0).astype(int)
y_pred_bin = (y_med > THRESHOLD).astype(int)

acc = accuracy_score(y_bin, y_pred_bin)
prec = precision_score(y_bin, y_pred_bin)
rec = recall_score(y_bin, y_pred_bin)
f1 = f1_score(y_bin, y_pred_bin)
auc = roc_auc_score(y_bin, y_med)
ap = average_precision_score(y_bin, y_med)
cm = confusion_matrix(y_bin, y_pred_bin)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

print(f"\n=== Binary classification: PISN-enriched (median > {THRESHOLD:.2f}) ===")
print(f"  class balance: {(y_bin == 1).mean():.1%} positive, "
      f"{(y_bin == 0).mean():.1%} negative")
print(f"  accuracy    = {acc:.4f}")
print(f"  precision   = {prec:.4f}")
print(f"  recall      = {rec:.4f}")
print(f"  specificity = {specificity:.4f}")
print(f"  F1          = {f1:.4f}")
print(f"  ROC-AUC     = {auc:.4f}")
print(f"  PR-AUC      = {ap:.4f}")
print("\n  confusion matrix (rows = true, cols = predicted):")
print(f"      [[ TN={tn:>4d}   FP={fp:>4d} ]")
print(f"       [ FN={fn:>4d}   TP={tp:>4d} ]]")
print("\n" + classification_report(y_bin, y_pred_bin,
                                target_names=["no PISN", "PISN"]))

# confusion matrix plot
fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["pred no PISN", "pred PISN"])
ax.set_yticklabels(["true no PISN", "true PISN"])
half = cm.max() / 2
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=18,
                color="white" if cm[i, j] > half else "black")
ax.set_xlabel("predicted")
ax.set_ylabel("true")
ax.set_title(f"Confusion matrix (median > {THRESHOLD:.2f})")
fig.colorbar(im, ax=ax)
_save(fig, "heldout_confusion_matrix.png")

# ROC curve
fpr, tpr, _ = roc_curve(y_bin, y_med)
fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.plot(fpr, tpr, color="tab:blue", lw=2, label=f"ROC (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
ax.set_xlabel("false positive rate")
ax.set_ylabel("true positive rate")
ax.set_title("ROC curve: PISN-enriched vs not")
ax.legend()
_save(fig, "heldout_roc.png")


# ============================================================================
# 4. Feature importance (full 14-element model)
# ============================================================================
rf_full = qrf.rf
imp_full = np.asarray(rf_full.feature_importances_, dtype=float)
order_full = np.argsort(imp_full)[::-1]

print(f"\n=== Full-model feature importances (top 20 of {len(full_names)}) ===")
for rank, i in enumerate(order_full[:20], 1):
    print(f"    {rank:>3}. {full_names[i]:<10} {imp_full[i]:.4f}")

elem_imp_full = element_importances(full_names, imp_full)
elem_sorted = sorted(elem_imp_full.items(), key=lambda kv: kv[1], reverse=True)
print("\n=== Element-level importance (full model) ===")
print("    " + "  ".join(f"{el}:{imp:.3f}" for el, imp in elem_sorted))

# plot: top-30 ratio importances
top_k = 30
top_idx = order_full[:top_k]
fig, ax = plt.subplots(figsize=(8, 9))
ax.barh(np.arange(top_k)[::-1], imp_full[top_idx][::-1], color="tab:blue")
ax.set_yticks(np.arange(top_k)[::-1])
ax.set_yticklabels([full_names[i] for i in top_idx][::-1], fontsize=8)
ax.set_xlabel("Gini importance")
ax.set_title(f"Full QRF: top {top_k} feature importances")
_save(fig, "full_feature_importance.png")

# plot: element importance
els = [e for e, _ in elem_sorted]
imps = [i for _, i in elem_sorted]
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(np.arange(len(els)), imps, color="tab:green")
ax.set_xticks(np.arange(len(els)))
ax.set_xticklabels(els)
ax.set_ylabel("summed Gini importance")
ax.set_title("Full QRF: element-level importance")
_save(fig, "full_element_importance.png")

# plot: held-out scatter (median pred vs truth)
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
ax.scatter(y_test, y_med, s=8, alpha=0.4, color="tab:blue")
ax.set_xlabel("true f_pisn")
ax.set_ylabel("predicted f_pisn (median)")
ax.set_title(f"Held-out: median pred vs truth (R2 = {r2:.3f})")
ax.legend()
_save(fig, "heldout_scatter.png")

# plot: reliability diagram (from the 4 nominal levels)
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
ax.plot([l for l, _, _ in INTERVALS], [empirical[l] for l, _, _ in INTERVALS],
        "o-", color="tab:blue", label="full QRF")
ax.set_xlabel("nominal coverage")
ax.set_ylabel("empirical coverage")
ax.set_title("Reliability diagram (held-out)")
ax.legend()
_save(fig, "heldout_reliability.png")


# ============================================================================
# 5. Reduced QRF restricted to Li+22-observable ratios (55 features)
# ============================================================================
names = obs_feature_names()
X_obs_train = subset_columns(X_full, full_names, names)
X_obs_train = SimpleImputer(strategy="median").fit_transform(X_obs_train)

X_tr, X_te = X_obs_train[train_idx], X_obs_train[test_idx]
y_tr, y_te = y[train_idx], y[test_idx]

qrf_obs = QuantileRegressionForest(
    n_estimators=cfg["n_estimators"],
    min_samples_leaf=cfg["min_samples_leaf"],
    random_state=cfg["random_state"],
    n_jobs=cfg["n_jobs"],
)
qrf_obs.fit(X_tr, y_tr)

y_med_te = qrf_obs.predict(X_te, quantiles=0.5)
r2_red = r2_score(y_te, y_med_te)
rmse_red = float(np.sqrt(mean_squared_error(y_te, y_med_te)))
print(f"\n=== Reduced QRF ({len(OBS_ELEMENTS)} elements, {len(names)} features) ===")
print(f"  test R2   = {r2_red:.4f}")
print(f"  test RMSE = {rmse_red:.4f}")

imp_red = np.asarray(qrf_obs.rf.feature_importances_, dtype=float)
order_red = np.argsort(imp_red)[::-1]
print("\n  reduced-model feature importances (top 20):")
for rank, i in enumerate(order_red[:20], 1):
    print(f"    {rank:>3}. {names[i]:<10} {imp_red[i]:.4f}")

elem_imp_red = element_importances(names, imp_red)
elem_red_sorted = sorted(elem_imp_red.items(), key=lambda kv: kv[1], reverse=True)
print("\n  reduced-model element importances:")
print("    " + "  ".join(f"{el}:{imp:.3f}" for el, imp in elem_red_sorted))

# plot: reduced feature importance
k = 25
fig, ax = plt.subplots(figsize=(8, 8))
ax.barh(np.arange(k)[::-1], imp_red[order_red[:k]][::-1], color="tab:green")
ax.set_yticks(np.arange(k)[::-1])
ax.set_yticklabels([names[i] for i in order_red[:k]][::-1], fontsize=8)
ax.set_xlabel("Gini importance")
ax.set_title("Reduced (Li+22-observable) QRF feature importances")
_save(fig, "reduced_feature_importance.png")


# ============================================================================
# 6. Load observed Li+22 stars and predict f_pisn
# ============================================================================
stars = load_obs_li()
X_obs, feh, teff, logg = build_obs_features(stars)
print(f"\n=== Li+22 catalogue: {len(stars)} stars ===")
per_elem = {el: int(sum(1 for s in stars if not np.isnan(get_xfe(s, el))))
            for el in OBS_ELEMENTS if el != "Fe"}
print("  measured [X/Fe] per element:", per_elem)
n_complete = int(sum(1 for s in stars
                     if all(not np.isnan(get_xfe(s, el)) for el in OBS_ELEMENTS)))
print(f"  stars with all {len(OBS_ELEMENTS)} elements measured: {n_complete}")

# impute observed NaN with training median (fit on training data)
imp = SimpleImputer(strategy="median").fit(X_obs_train)
X_obs_imp = imp.transform(X_obs)

q_obs = qrf_obs.predict(X_obs_imp, quantiles=[0.05, 0.16, 0.5, 0.84, 0.95])
q_obs = np.clip(q_obs, 0.0, 1.0)
pred_med = q_obs[:, 2]

print("\n=== Predicted f_pisn for observed stars ===")
print(f"  mean = {pred_med.mean():.3f}   median = {np.median(pred_med):.3f}")
print(f"  min = {pred_med.min():.3f}   max = {pred_med.max():.3f}")
print(f"  frac(f_pisn < 0.1) = {(pred_med < 0.1).mean():.3f}")
print(f"  frac(f_pisn < 0.5) = {(pred_med < 0.5).mean():.3f}")


# ============================================================================
# 7. Plots of observed predictions
# ============================================================================
# histogram
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(pred_med, bins=30, range=(0, 1), color="tab:blue", alpha=0.8)
ax.axvline(0.5, color="k", ls="--", lw=1, label="f_pisn = 0.5")
ax.set_xlabel("predicted f_pisn (median)")
ax.set_ylabel("N stars")
ax.set_title(f"Li+22: predicted f_pisn (N={len(stars)})")
ax.legend()
_save(fig, "obs_fpisn_hist.png")

# median + 68% band vs [Fe/H]
ok = np.isfinite(feh)
so = np.argsort(feh)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.fill_between(feh[so], q_obs[:, 1][so], q_obs[:, 3][so],
                color="tab:blue", alpha=0.25, label="68% CI")
ax.plot(feh[so], q_obs[:, 2][so], "o-", ms=3, color="tab:blue", label="median")
ax.set_xlabel("[Fe/H]")
ax.set_ylabel("predicted f_pisn")
ax.set_title("Li+22: QRF f_pisn vs [Fe/H] (with uncertainty)")
ax.legend()
_save(fig, "obs_fpisn_vs_feh.png")

# median vs Teff
okt = np.isfinite(teff)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(teff[okt], pred_med[okt], s=12, alpha=0.5, color="tab:red")
ax.set_xlabel("Teff (K)")
ax.set_ylabel("predicted f_pisn (median)")
ax.set_title("Li+22: predicted f_pisn vs Teff")
_save(fig, "obs_fpisn_vs_teff.png")


# ============================================================================
# 8. Export per-star predictions to CSV
# ============================================================================
n_meas = np.array([
    sum(1 for el in OBS_ELEMENTS if el != "Fe" and not np.isnan(get_xfe(s, el)))
    for s in stars
], dtype=int)

df = pd.DataFrame({
    "star": [s["label"] for s in stars],
    "FeH": feh,
    "Teff": teff,
    "logg": logg,
    "N_measured": n_meas,
    "f_pisn_q05": q_obs[:, 0],
    "f_pisn_q16": q_obs[:, 1],
    "f_pisn_q50": q_obs[:, 2],
    "f_pisn_q84": q_obs[:, 3],
    "f_pisn_q95": q_obs[:, 4],
})
df = df.sort_values("f_pisn_q50", ascending=False).reset_index(drop=True)
df.to_csv(CSV_PATH, index=False)
print(f"\n=== Exported {len(df)} stars -> {CSV_PATH} ===")
print(df.head(10).to_string(index=False))

print(f"\ndone. outputs in {RESULTS_DIR}")
