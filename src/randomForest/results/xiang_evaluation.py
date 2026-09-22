"""
Quantile Random Forest on f_pisn: evaluation on the Xiang et al. (2019)
LAMOST value-added catalogue (Vizier J/ApJS/245/34).

The Xiang catalogue provides [X/Fe] for C, N, O, Na, Mg, Al, Si, Ca, Ti, Cr,
Mn, Co, Ni, Cu, Ba.  Compared to the 14 elements the full QRF was trained on
(C, O, Na, Mg, Al, Si, Ca, Cr, Mn, Fe, Co, Ni, Cu, Zn), Xiang has every
element EXCEPT Zn -- but it *does* have O, Al and Cu, which the Li+22
catalogue lacked.  So we evaluate with a reduced 13-element QRF
(78 pairwise [X/Y] features) and drop only Zn (the least important element).

Metallicity cut is [Fe/H] in [-3, 0], matching the mock-data FEH_RANGE.
All outputs are written to src/randomForest/results/xiang/.
"""

import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
import joblib

import matplotlib

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


# --- repo-root / path setup -------------------------------------------------
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

# --- paths ------------------------------------------------------------------
DATA_DIR = os.path.join(ROOT, "src", "randomForest", "data")
XIANG_DIR = os.path.join(ROOT, "src", "randomForest", "results", "xiang")
CACHE_PATH = os.path.join(DATA_DIR, "dataset.npz")
XIANG_CACHE = os.path.join(XIANG_DIR, "xiang_catalog.csv")
os.makedirs(XIANG_DIR, exist_ok=True)

QUANTILES = [0.025, 0.05, 0.16, 0.25, 0.5, 0.75, 0.84, 0.95, 0.975]

# The 14 training elements, in the exact order used by dataset._abundance_elements().
FULL_ELEMENTS = ["C", "O", "Na", "Mg", "Al", "Si", "Ca", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"]
# Xiang measures all of these except Zn.
XIANG_ELEMENTS = [e for e in FULL_ELEMENTS if e != "Zn"]
assert XIANG_ELEMENTS == ["C", "O", "Na", "Mg", "Al", "Si", "Ca", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"]

# Metallicity range (matches dataset.FEH_RANGE = (-3.0, 0.0)).
FEH_LO, FEH_HI = -3.0, 0.0

# Vizier fetch: how many rows to pull (the full catalogue is far larger; the
# user is downloading the complete ~2 GB file separately).  Bump this if needed.
FETCH_ROW_LIMIT = 200000


# --- helpers ----------------------------------------------------------------
def xiang_feature_names():
    return [f"[{x}/{y}]" for x, y in combinations(XIANG_ELEMENTS, 2)]


def subset_columns(X, full_names, subset_names):
    idx = [full_names.index(nm) for nm in subset_names]
    return X[:, idx]


def _save(fig, name):
    path = os.path.join(XIANG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  saved {path}")


def load_xiang_catalog():
    """Return the Xiang catalogue as a DataFrame; fetch from Vizier if uncached."""
    if not os.path.exists(XIANG_CACHE):
        from astroquery.vizier import Vizier
        print(f"fetching J/ApJS/245/34 with [Fe/H] in [{FEH_LO},{FEH_HI}], "
              f"row_limit={FETCH_ROW_LIMIT} ...")
        V = Vizier(columns=["**"],
                   column_filters={"[Fe/H]": f"{FEH_LO}..{FEH_HI}"},
                   row_limit=FETCH_ROW_LIMIT)
        t = V.get_catalogs("J/ApJS/245/34")[0]
        df = t.to_pandas()
        df.to_csv(XIANG_CACHE, index=False)
        print(f"fetched {len(df)} rows -> {XIANG_CACHE}")
    return pd.read_csv(XIANG_CACHE)


def get_xfe_col(df, elem):
    """[X/Fe] as a float array (NaN where missing or unreliable). Fe -> 0."""
    if elem == "Fe":
        return np.zeros(len(df))
    col = f"[{elem}/Fe]"
    qcol = f"q_[{elem}/Fe]"
    vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    if qcol in df.columns:
        q = pd.to_numeric(df[qcol], errors="coerce").to_numpy(dtype=float)
        vals = np.where(q == 1.0, vals, np.nan)
    return vals


def build_xiang_features(df):
    """(X_obs, feh, teff, logg, snrg, name, specid, ra, dec)."""
    names = xiang_feature_names()
    n = len(df)
    X = np.full((n, len(names)), np.nan)
    xfe = {e: get_xfe_col(df, e) for e in XIANG_ELEMENTS}
    for j, (x, y) in enumerate(combinations(XIANG_ELEMENTS, 2)):
        X[:, j] = xfe[x] - xfe[y]  # NaN propagates where either side is NaN
    feh = pd.to_numeric(df["[Fe/H]"], errors="coerce").to_numpy(dtype=float)
    teff = pd.to_numeric(df["Teff"], errors="coerce").to_numpy(dtype=float)
    logg = pd.to_numeric(df["logg"], errors="coerce").to_numpy(dtype=float)
    snrg = pd.to_numeric(df["snrg"], errors="coerce").to_numpy(dtype=float)
    name = df["Name"].astype(str).to_numpy()
    specid = df["SpecID"].astype(str).to_numpy()
    ra = pd.to_numeric(df["RAJ2000"], errors="coerce").to_numpy(dtype=float)
    dec = pd.to_numeric(df["DEJ2000"], errors="coerce").to_numpy(dtype=float)
    return X, feh, teff, logg, snrg, name, specid, ra, dec


# ============================================================================
# 1. Load synthetic training data
# ============================================================================
X_full, y, diag = load_dataset(CACHE_PATH)
full_names = list(diag["feature_names"])
print(f"training dataset: X={X_full.shape}, {len(full_names)} features")

names = xiang_feature_names()
print(f"Xiang feature subset: {len(XIANG_ELEMENTS)} elements, {len(names)} [X/Y] features")


# ============================================================================
# 2. Train reduced QRF on the Xiang-observable feature subset
# ============================================================================
X_sub = subset_columns(X_full, full_names, names)
X_sub = SimpleImputer(strategy="median").fit_transform(X_sub)

rng = np.random.default_rng(CONFIG["seed"])
n = len(y)
idx = rng.permutation(n)
n_test = int(n * CONFIG["test_size"])
test_idx, train_idx = idx[:n_test], idx[n_test:]

X_tr, X_te = X_sub[train_idx], X_sub[test_idx]
y_tr, y_te = y[train_idx], y[test_idx]

qrf = QuantileRegressionForest(
    n_estimators=CONFIG["n_estimators"],
    min_samples_leaf=CONFIG["min_samples_leaf"],
    random_state=CONFIG["random_state"],
    n_jobs=CONFIG["n_jobs"],
)
qrf.fit(X_tr, y_tr)

y_med_te = qrf.predict(X_te, quantiles=0.5)
r2_red = r2_score(y_te, y_med_te)
rmse_red = float(np.sqrt(mean_squared_error(y_te, y_med_te)))
print(f"\n=== Reduced QRF ({len(XIANG_ELEMENTS)} elements, {len(names)} features, "
      f"dropped Zn) ===")
print(f"  held-out R2   = {r2_red:.4f}")
print(f"  held-out RMSE = {rmse_red:.4f}")

imp_red = np.asarray(qrf.rf.feature_importances_, dtype=float)
order_red = np.argsort(imp_red)[::-1]
print("\n  reduced-model feature importances (top 20):")
for rank, i in enumerate(order_red[:20], 1):
    print(f"    {rank:>3}. {names[i]:<10} {imp_red[i]:.4f}")


def element_importances(nms, imps):
    d = {}
    for nm, imp in zip(nms, imps):
        x, yy = nm.strip("[]").split("/")
        d[x] = d.get(x, 0.0) + float(imp)
        d[yy] = d.get(yy, 0.0) + float(imp)
    return d


elem_imp = element_importances(names, imp_red)
elem_sorted = sorted(elem_imp.items(), key=lambda kv: kv[1], reverse=True)
print("\n  reduced-model element importances:")
print("    " + "  ".join(f"{el}:{imp:.3f}" for el, imp in elem_sorted))

# plot: feature importance (top 25)
k = 25
fig, ax = plt.subplots(figsize=(8, 8))
ax.barh(np.arange(k)[::-1], imp_red[order_red[:k]][::-1], color="tab:green")
ax.set_yticks(np.arange(k)[::-1])
ax.set_yticklabels([names[i] for i in order_red[:k]][::-1], fontsize=8)
ax.set_xlabel("Gini importance")
ax.set_title(f"Xiang 13-element QRF: top {k} feature importances (Zn dropped)")
_save(fig, "xiang_feature_importance.png")

# plot: element importance
els = [e for e, _ in elem_sorted]
imps = [i for _, i in elem_sorted]
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(np.arange(len(els)), imps, color="tab:orange")
ax.set_xticks(np.arange(len(els)))
ax.set_xticklabels(els)
ax.set_ylabel("summed Gini importance")
ax.set_title("Xiang 13-element QRF: element-level importance")
_save(fig, "xiang_element_importance.png")


# ============================================================================
# 3. Load Xiang catalogue and predict f_pisn
# ============================================================================
df = load_xiang_catalog()
X_obs, feh, teff, logg, snrg, name, specid, ra, dec = build_xiang_features(df)
print(f"\n=== Xiang catalogue: {len(df)} stars (cached CSV) ===")

per_elem = {e: int(np.isfinite(get_xfe_col(df, e)).sum()) for e in XIANG_ELEMENTS if e != "Fe"}
print("  reliable (q==1) [X/Fe] per element:", per_elem)
n_complete = int(np.isfinite(X_obs).all(axis=1).sum())
print(f"  stars with all {len(XIANG_ELEMENTS)} elements reliable: {n_complete}")

imp = SimpleImputer(strategy="median").fit(X_tr)
X_obs_imp = imp.transform(X_obs)

q_obs = qrf.predict(X_obs_imp, quantiles=[0.05, 0.16, 0.5, 0.84, 0.95])
q_obs = np.clip(q_obs, 0.0, 1.0)
pred_med = q_obs[:, 2]

print("\n=== Predicted f_pisn for Xiang stars ===")
print(f"  mean = {pred_med.mean():.3f}   median = {np.median(pred_med):.3f}")
print(f"  min = {pred_med.min():.3f}   max = {pred_med.max():.3f}")
print(f"  frac(f_pisn < 0.1) = {(pred_med < 0.1).mean():.3f}")
print(f"  frac(f_pisn < 0.5) = {(pred_med < 0.5).mean():.3f}")


# ============================================================================
# 4. Plots of Xiang predictions
# ============================================================================
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(pred_med, bins=30, range=(0, 1), color="tab:blue", alpha=0.8)
ax.axvline(0.5, color="k", ls="--", lw=1, label="f_pisn = 0.5")
ax.set_xlabel("predicted f_pisn (median)")
ax.set_ylabel("N stars")
ax.set_title(f"Xiang (LAMOST): predicted f_pisn (N={len(df)})")
ax.legend()
_save(fig, "xiang_fpisn_hist.png")

# median + 68% band vs [Fe/H]
ok = np.isfinite(feh)
so = np.argsort(feh[ok])
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.fill_between(feh[ok][so], q_obs[:, 1][ok][so], q_obs[:, 3][ok][so],
                color="tab:blue", alpha=0.25, label="68% CI")
ax.plot(feh[ok][so], q_obs[:, 2][ok][so], "o-", ms=3, color="tab:blue", label="median")
ax.set_xlabel("[Fe/H]")
ax.set_ylabel("predicted f_pisn")
ax.set_title("Xiang (LAMOST): QRF f_pisn vs [Fe/H]")
ax.legend()
_save(fig, "xiang_fpisn_vs_feh.png")

# median vs Teff
okt = np.isfinite(teff)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(teff[okt], pred_med[okt], s=12, alpha=0.5, color="tab:red")
ax.set_xlabel("Teff (K)")
ax.set_ylabel("predicted f_pisn (median)")
ax.set_title("Xiang (LAMOST): predicted f_pisn vs Teff")
_save(fig, "xiang_fpisn_vs_teff.png")


# ============================================================================
# 5. Export per-star predictions to CSV
# ============================================================================
n_meas = np.array([
    sum(1 for e in XIANG_ELEMENTS if e != "Fe" and np.isfinite(get_xfe_col(df, e)[i]))
    for i in range(len(df))
], dtype=int)

out = pd.DataFrame({
    "star": name,
    "SpecID": specid,
    "RA": ra,
    "Dec": dec,
    "FeH": feh,
    "Teff": teff,
    "logg": logg,
    "snrg": snrg,
    "N_measured": n_meas,
    "f_pisn_q05": q_obs[:, 0],
    "f_pisn_q16": q_obs[:, 1],
    "f_pisn_q50": q_obs[:, 2],
    "f_pisn_q84": q_obs[:, 3],
    "f_pisn_q95": q_obs[:, 4],
})
out = out.sort_values("f_pisn_q50", ascending=False).reset_index(drop=True)
csv_path = os.path.join(XIANG_DIR, "xiang_predictions.csv")
out.to_csv(csv_path, index=False)
print(f"\n=== Exported {len(out)} stars -> {csv_path} ===")
print(out.head(10).to_string(index=False))

print(f"\ndone. outputs in {XIANG_DIR}")
