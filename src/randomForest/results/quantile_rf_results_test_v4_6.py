"""
test_v4_6: Xiang+2019 (LAMOST DR5) evaluation on a ~10,000-star subsample.

Same pipeline as test_v4_5, but loads a reproducible random subsample of the
Xiang catalogue (N_STARS ~ 5000-10000) instead of the full 178k stars, so the
seed-ensemble QRF prediction runs in well under a minute instead of minutes.

Same QRF pipeline as test_v4_3 (Takahashi PISN + Limongi18 SN, missing-element
-aware treatment: complete-star subset, completeness-stratified check, and a
mask-aware augmented + seed-ensembled QRF), but:

  * the training element set is the 15-element Xiang set (C, N, O, Na, Mg, Al,
    Si, Ca, Ti, Cr, Mn, Fe, Co, Ni, Cu -> 105 ratios), dropping Zn and Ba and
    adding N + Ti;
  * the observed catalogue is Xiang+2019 (LAMOST DR5 DDPayne recommended), loaded
    via load_obs_xiang with [Fe/H] in [-3, -1.5) and reliable flags only; and
  * the realistic per-element noise sigma is the MEDIAN Xiang measurement error
    ({EL}_FE_ERR, and FEH_ERR for Fe) instead of the Li+22 sigma_Total.

The clean (dex_error=0.0) cached dataset.npz (Takahashi + Limongi18, 105
features) is loaded and noise is added post-hoc with a fixed seed (NOISE_SEED),
mirroring test_v4_3.  All outputs are written to
src/randomForest/results/test_v4_5/.
"""

import os
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

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
from sklearn.metrics import r2_score  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


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
from src.load_data import load_obs_xiang, XIANG_ELEMENTS  # noqa: E402
from src.randomForest.dataset import add_dex_noise, _feature_pairs  # noqa: E402

DATA_DIR = os.path.join(ROOT, "src", "randomForest", "data")
RESULTS_DIR = os.path.join(ROOT, "src", "randomForest", "results", "test_v4_6")
CACHE_PATH = os.path.join(DATA_DIR, "dataset.npz")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Subsample the Xiang catalogue for speed (see docstring).
N_STARS = 10000         # ~5000-10000 stars
SUBSAMPLE_SEED = 42     # reproducible random draw

QUANTILES = [0.05, 0.16, 0.5, 0.84, 0.95]

# 15-element Xiang training set (same order as src.randomForest.dataset).
OBS_ELEMENTS = ["C", "N", "O", "Na", "Mg", "Al", "Si", "Ca", "Ti", "Cr", "Mn",
                "Fe", "Co", "Ni", "Cu"]
NON_FE = [el for el in OBS_ELEMENTS if el != "Fe"]     # 14 non-Fe elements
PAIRS = list(combinations(OBS_ELEMENTS, 2))            # 105
RATIO_NAMES = [f"[{x}/{y}]" for x, y in PAIRS]
IND_NAMES = [f"ind[{el}]" for el in NON_FE]
EL_TO_RATIOS = {el: [j for j, (x, y) in enumerate(PAIRS) if el in (x, y)] for el in OBS_ELEMENTS}

# mask-aware settings
M_AUG = 5          # masked copies per synthetic star (+1 complete copy)
K_SEEDS = 5        # mask-seed ensemble size
SEEDS = [777 + i for i in range(K_SEEDS)]
NOISE_SEED = 2024  # seed for the element-specific observational noise


def get_xfe(star, elem):
    """[X/Fe] for a Xiang star; Fe is the reference -> 0.0; unreliable -> NaN."""
    if elem == "Fe":
        return 0.0
    rec = star["abundances"].get(elem)
    if rec is None:
        return np.nan
    xfe = rec.get("XFe")
    return np.nan if xfe is None else float(xfe)


def xiang_element_sigmas(stars):
    """Median Xiang measurement error per element (dex).

    Non-Fe elements use the median {EL}_FE_ERR over reliably-measured stars;
    Fe uses the median FEH_ERR.  Mirrors li_plus_element_sigmas() but for the
    Xiang catalogue (which reports [X/Fe] errors directly, no ion stage).
    """
    per_el = defaultdict(list)
    feh_errs = []
    for s in stars:
        e = s["params"].get("e_FeH")
        if e is not None and np.isfinite(e) and e != -999.0:
            feh_errs.append(float(e))
        for el, rec in s["abundances"].items():
            sig = rec.get("sigma_Total")
            if sig is not None and np.isfinite(sig):
                per_el[el].append(float(sig))
    sigmas = {el: float(np.median(v)) for el, v in per_el.items() if v}
    if feh_errs:
        sigmas["Fe"] = float(np.median(feh_errs))
    return sigmas


def build_obs_features(stars):
    n = len(stars)
    X = np.full((n, len(PAIRS)), np.nan)
    feh = np.full(n, np.nan)
    teff = np.full(n, np.nan)
    logg = np.full(n, np.nan)
    for i, star in enumerate(stars):
        xfe = {el: get_xfe(star, el) for el in OBS_ELEMENTS}
        for j, (x, y) in enumerate(PAIRS):
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


def indicators_from_stars(stars):
    return np.array([
        [1.0 if not np.isnan(get_xfe(s, el)) else 0.0 for el in NON_FE]
        for s in stars
    ])


def mask_features(X, rng, freq):
    """Drop elements at rate 1-freq (set their ratios NaN) and emit indicators."""
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
    """M masked copies + 1 complete copy per sample -> (X_feat, y_aug)."""
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


def _save(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  saved {path}")


# ============================================================================
# Load training data (15-element / 105-ratio scheme)
# ============================================================================
X_full, y, diag = load_dataset(CACHE_PATH)
full_names = list(diag["feature_names"])
print(f"dataset: pisn={diag.get('pisn_source')} sn={diag.get('sn_source')} "
      f"X={X_full.shape} (features={len(full_names)})")

# Load the observed Xiang catalogue once (used for both noise sigma + freq).
stars = load_obs_xiang(n_max=N_STARS, seed=SUBSAMPLE_SEED)
N = len(stars)
print(f"observed Xiang stars = {N} (subsampled from full catalogue)")

# Realistic element-specific observational noise: Xiang median measurement
# error per element, added to the clean mock before subsetting.
sigmas = xiang_element_sigmas(stars)
noise_rng = np.random.default_rng(NOISE_SEED)
X_full = X_full.copy()
add_dex_noise(X_full, _feature_pairs(), sigmas, noise_rng)
print("element-specific noise sigma (dex):",
      {el: round(s, 3) for el, s in sigmas.items()})

X_obs_full = subset_columns(X_full, full_names, RATIO_NAMES)   # (2000, 105) complete
imp = SimpleImputer(strategy="median").fit(X_obs_full)         # medians from complete data
X_imp = imp.transform(X_obs_full)

freq = {el: float(np.mean([not np.isnan(get_xfe(s, el)) for s in stars])) for el in NON_FE}
print(f"training X={X_obs_full.shape} (complete), obs stars={N}")
print("observed frequencies:", {el: round(freq[el], 3) for el in NON_FE})

cfg = CONFIG
rng = np.random.default_rng(cfg["seed"])
n = len(y)
idx = rng.permutation(n)
n_test = int(n * cfg["test_size"])
test_idx, train_idx = idx[:n_test], idx[n_test:]

# ============================================================================
# Held-out benchmark (split) — characterizes the two methods
# ============================================================================
qrf_std_bench = _new_qrf(cfg)
qrf_std_bench.fit(X_imp[train_idx], y[train_idx])
r2_s = r2_score(y[test_idx], qrf_std_bench.predict(X_imp[test_idx], 0.5))

rng_mask_b = np.random.default_rng(777)
X_tr_aug, y_tr_aug = augment(X_obs_full[train_idx], y[train_idx], rng_mask_b, freq, imp)
qrf_mask_bench = _new_qrf(cfg)
qrf_mask_bench.fit(X_tr_aug, y_tr_aug)
X_te_mask, ind_te = mask_features(X_obs_full[test_idx], rng_mask_b, freq)
X_te_feat = np.hstack([imp.transform(X_te_mask), ind_te])
r2_m = r2_score(y[test_idx], qrf_mask_bench.predict(X_te_feat, 0.5))
print(f"\nheld-out R2: standard={r2_s:.4f}  mask-aware(masked test)={r2_m:.4f}")

# ============================================================================
# FINAL models trained on the FULL dataset (per "use the entire dataset")
# ============================================================================
qrf_std = _new_qrf(cfg)
qrf_std.fit(X_imp, y)

X_obs, feh, teff, logg = build_obs_features(stars)
X_obs_imp = imp.transform(X_obs)
ind_obs = indicators_from_stars(stars)
X_obs_feat = np.hstack([X_obs_imp, ind_obs])

preds_q = np.zeros((K_SEEDS, N, len(QUANTILES)))
imp_seeds = np.zeros((K_SEEDS, len(RATIO_NAMES) + len(IND_NAMES)))
for si, seed in enumerate(SEEDS):
    rng_mask = np.random.default_rng(seed)
    X_aug, y_aug = augment(X_obs_full, y, rng_mask, freq, imp)
    qrf = _new_qrf(cfg)
    qrf.fit(X_aug, y_aug)
    preds_q[si] = np.clip(qrf.predict(X_obs_feat, quantiles=QUANTILES), 0.0, 1.0)
    imp_seeds[si] = np.asarray(qrf.rf.feature_importances_, dtype=float)

q_mask = preds_q.mean(axis=0)                       # ensemble-mean quantiles
q_mask_se = preds_q[:, :, 2].std(axis=0)            # per-star seed std of q50
q_std = np.clip(qrf_std.predict(X_obs_imp, quantiles=QUANTILES), 0.0, 1.0)

has_si = np.array([not np.isnan(get_xfe(s, "Si")) for s in stars], dtype=bool)
complete = np.array([
    all(not np.isnan(get_xfe(s, el)) for el in OBS_ELEMENTS) for s in stars
], dtype=bool)
n_meas = np.array([
    sum(1 for el in NON_FE if not np.isnan(get_xfe(s, el))) for s in stars
], dtype=int)
labels = np.array([s["label"] for s in stars])

# ============================================================================
# 1. Complete-star subset (primary, no imputation)
# ============================================================================
n_comp = int(complete.sum())
if n_comp > 0:
    X_comp = X_obs[complete]             # no NaN -> imputer is a no-op
    q_comp = np.clip(qrf_std.predict(X_comp, quantiles=QUANTILES), 0.0, 1.0)
    pred_comp = q_comp[:, 2]
    feh_comp, teff_comp, logg_comp = feh[complete], teff[complete], logg[complete]

    print(f"\n===== 1. complete-star subset (N={n_comp}, no imputation) =====")
    print(f"  mean={pred_comp.mean():.3f}  median={np.median(pred_comp):.3f}  "
          f"min={pred_comp.min():.3f}  max={pred_comp.max():.3f}")
    print(f"  frac(f<0.1)={(pred_comp < 0.1).mean():.3f}  "
          f"frac(f<0.5)={(pred_comp < 0.5).mean():.3f}  "
          f"frac(f>0.5)={(pred_comp > 0.5).mean():.3f}")
else:
    print("\n===== 1. complete-star subset (N=0) =====")
    print("  no star has all 15 elements measured (Cu/Na/Mg/N too rare); "
          "skipping complete-subset analysis")

# ============================================================================
# 2. Completeness-stratified check
# ============================================================================
pred_std = q_std[:, 2]
pred_mask = q_mask[:, 2]
width_std = q_std[:, 4] - q_std[:, 0]
width_mask = q_mask[:, 4] - q_mask[:, 0]

print("\n===== 2. completeness-stratified =====")
_rho, _pv = spearmanr(n_meas, pred_std)
print(f"  Spearman(pred_fpisn, N_measured) = {_rho:+.3f}  (p={_pv:.2e})")
print(f"  {'group':<16} {'N':>4} {'std mean':>9} {'mask mean':>10} {'std frac<0.1':>12}")
for name, m in [("with Si", has_si), ("without Si", ~has_si),
                ("complete", complete), ("incomplete", ~complete)]:
    cnt = int(m.sum())
    if cnt == 0:
        print(f"  {name:<16} {cnt:>4} {'--':>9} {'--':>10} {'--':>12}")
        continue
    print(f"  {name:<16} {cnt:>4} {pred_std[m].mean():>9.3f} "
          f"{pred_mask[m].mean():>10.3f} {(pred_std[m] < 0.1).mean():>12.3f}")

# ============================================================================
# 3. Mask-aware vs standard summary (stable, seed-ensembled)
# ============================================================================
print("\n===== 3. mask-aware vs standard (overall, ensemble) =====")
print(f"  mean:      std={pred_std.mean():.3f}  mask={pred_mask.mean():.3f}")
print(f"  median:    std={np.median(pred_std):.3f}  mask={np.median(pred_mask):.3f}")
print(f"  frac<0.1:  std={(pred_std < 0.1).mean():.3f}  mask={(pred_mask < 0.1).mean():.3f}")
print(f"  mean 95% width: std={width_std.mean():.3f}  mask={width_mask.mean():.3f}")
print(f"  mask seed-uncertainty q50 (per-star std): mean={q_mask_se.mean():.4f}  "
      f"median={np.median(q_mask_se):.4f}  max={q_mask_se.max():.4f}")
print(f"  ensemble-mean SE (std/sqrt(K)): mean={q_mask_se.mean()/np.sqrt(K_SEEDS):.4f}")
flip = int(((pred_std < 0.1) & (pred_mask >= 0.1)).sum())
print(f"  stars std says f<0.1 but mask says f>=0.1: {flip}")

# ============================================================================
# Feature importances (mask-aware, averaged over seed ensemble)
# ============================================================================
imp_m = imp_seeds.mean(axis=0)
order = np.argsort(imp_m)[::-1]
n_ratio = len(RATIO_NAMES)
print("\n===== mask-aware feature importances (top 12, ensemble-mean) =====")
for i in order[:12]:
    nm = RATIO_NAMES[i] if i < n_ratio else IND_NAMES[i - n_ratio]
    print(f"  {nm:<10} {imp_m[i]:.4f}")
print(f"  (total indicator importance = {imp_m[n_ratio:].sum():.4f})")

# ============================================================================
# Export CSVs
# ============================================================================
df_all = pd.DataFrame({
    "star": labels,
    "FeH": feh, "Teff": teff, "logg": logg,
    "N_measured": n_meas,
    "has_Si": has_si.astype(int),
    "is_complete": complete.astype(int),
    "f_pisn_std_q05": q_std[:, 0], "f_pisn_std_q16": q_std[:, 1],
    "f_pisn_std_q50": q_std[:, 2], "f_pisn_std_q84": q_std[:, 3],
    "f_pisn_std_q95": q_std[:, 4],
    "f_pisn_mask_q05": q_mask[:, 0], "f_pisn_mask_q16": q_mask[:, 1],
    "f_pisn_mask_q50": q_mask[:, 2], "f_pisn_mask_q84": q_mask[:, 3],
    "f_pisn_mask_q95": q_mask[:, 4],
    "f_pisn_mask_q50_se": q_mask_se,
})
df_all.to_csv(os.path.join(RESULTS_DIR, "obs_xiang_all.csv"), index=False)

if n_comp > 0:
    df_comp = pd.DataFrame({
        "star": labels[complete],
        "FeH": feh_comp, "Teff": teff_comp, "logg": logg_comp,
        "f_pisn_q16": q_comp[:, 1], "f_pisn_q50": q_comp[:, 2], "f_pisn_q84": q_comp[:, 3],
    })
    df_comp = df_comp.sort_values("f_pisn_q50", ascending=False).reset_index(drop=True)
    df_comp.to_csv(os.path.join(RESULTS_DIR, "obs_xiang_complete.csv"), index=False)
    print(f"\nexported {len(df_all)} stars -> obs_xiang_all.csv")
    print(f"exported {len(df_comp)} complete stars -> obs_xiang_complete.csv")
else:
    print(f"\nexported {len(df_all)} stars -> obs_xiang_all.csv")
    print("no complete stars -> obs_xiang_complete.csv skipped (empty)")

# ============================================================================
# Plots
# ============================================================================
if n_comp > 0:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(pred_comp, bins=20, range=(0, 1), color="tab:orange", alpha=0.8)
    ax.axvline(0.5, color="k", ls="--", lw=1)
    ax.set_xlabel("predicted f_pisn (median)")
    ax.set_ylabel("N stars")
    ax.set_title(f"Xiang+19 complete stars ({n_comp}): f_pisn")
    _save(fig, "complete_fpisn_hist.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(feh_comp, pred_comp, s=15, alpha=0.6, color="tab:orange")
    ax.set_xlabel("[Fe/H]")
    ax.set_ylabel("predicted f_pisn (median)")
    ax.set_title("Xiang+19 complete stars: f_pisn vs [Fe/H]")
    _save(fig, "complete_fpisn_vs_feh.png")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(feh, pred_std, s=15, alpha=0.6, color="tab:orange", label="standard")
ax.scatter(feh, pred_mask, s=15, alpha=0.6, color="tab:blue", label="mask-aware")
ax.set_xlabel("[Fe/H]")
ax.set_ylabel("predicted f_pisn (median)")
ax.set_title(f"Xiang+19 all stars ({N}): f_pisn vs [Fe/H]")
ax.legend()
_save(fig, "obs_fpisn_vs_feh.png")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(n_meas + np.random.uniform(-0.2, 0.2, N), pred_std, s=8,
           alpha=0.4, color="tab:blue", label="standard")
ks = sorted(set(n_meas))
ax.plot(ks, [pred_std[n_meas == k].mean() for k in ks], "o-", color="tab:red",
        label="group mean")
ax.set_xlabel("N elements measured (of 14)")
ax.set_ylabel("predicted f_pisn (median)")
ax.set_title("Xiang+19: f_pisn vs completeness")
ax.legend()
_save(fig, "fpisn_vs_nmeasured.png")

fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.boxplot([pred_std[has_si], pred_std[~has_si]],
           tick_labels=["with Si", "without Si"])
ax.set_ylabel("predicted f_pisn (median, standard model)")
ax.set_title(f"Si present ({int(has_si.sum())}) vs absent ({int((~has_si).sum())})")
_save(fig, "si_vs_nosi_box.png")

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.scatter(pred_std[has_si], pred_mask[has_si], s=10, alpha=0.5,
           color="tab:blue", label=f"with Si ({int(has_si.sum())})")
ax.scatter(pred_std[~has_si], pred_mask[~has_si], s=10, alpha=0.5,
           color="tab:red", label=f"without Si ({int((~has_si).sum())})")
ax.set_xlabel("standard predicted f_pisn")
ax.set_ylabel("mask-aware predicted f_pisn (ensemble)")
ax.set_title("standard vs mask-aware")
ax.legend()
_save(fig, "std_vs_mask_scatter.png")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(pred_mask, bins=np.arange(0, 1.001, 0.05), color="tab:purple", alpha=0.85, edgecolor="white")
ax.axvline(0.5, color="k", ls="--", lw=1.2)
ax.set_xlabel("predicted f_pisn (mask-aware, median)")
ax.set_ylabel("N stars")
ax.set_title(f"Xiang+19 ({N} stars): mask-aware f_pisn")
ax.text(0.97, 0.95, f"mean={pred_mask.mean():.2f}\nmedian={np.median(pred_mask):.2f}\n"
        f"f>0.5: {(pred_mask > 0.5).mean() * 100:.1f}%",
        transform=ax.transAxes, ha="right", va="top", fontsize=9)
_save(fig, "mask_fpisn_hist.png")

k = 20
top = order[:k]
names_top = [RATIO_NAMES[i] if i < n_ratio else IND_NAMES[i - n_ratio] for i in top]
fig, ax = plt.subplots(figsize=(8, 8))
ax.barh(np.arange(k)[::-1], imp_m[top][::-1], color="tab:green")
ax.set_yticks(np.arange(k)[::-1])
ax.set_yticklabels([names_top[i] for i in range(k)][::-1], fontsize=8)
ax.set_xlabel("Gini importance (ensemble mean)")
ax.set_title("Mask-aware QRF feature importances (top 20)")
_save(fig, "mask_feature_importance.png")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(q_mask_se, bins=30, color="tab:purple", alpha=0.8)
ax.set_xlabel("per-star seed std of f_pisn(q50)")
ax.set_ylabel("N stars")
ax.set_title(f"mask-aware missingness uncertainty (K={K_SEEDS} seeds)")
_save(fig, "mask_seed_uncertainty.png")

print(f"\ndone. outputs in {RESULTS_DIR}")
