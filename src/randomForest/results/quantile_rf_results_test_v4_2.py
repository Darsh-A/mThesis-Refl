"""
Hurdle (two-stage zero-inflated) model for f_pisn in [0,1] from stellar
abundance ratios (test_v4_2): Takahashi PISN yields, clean (dex_error=0.0)
dataset, applied to the same Li+22 very-metal-poor star catalogue as v4_1.

The v4_1 QRF regresses f_pisn directly, but many Li+22 stars are predicted
at/near f_pisn=0, so the conditional distribution carries a point mass at zero
that a single continuous QRF models poorly. v4_2 replaces that single QRF with
a HURDLE (two-stage zero-inflated) model:

  1. STAGE 1 (zero vs positive): a calibrated RandomForestClassifier predicts
     pi = P(f_pisn > 0) for each star (isotonic calibration, 5-fold CV).
  2. STAGE 2 (positive magnitude): a QRF trained only on the positive-f_pisn
     training stars predicts the positive-component quantiles.

The two stages are combined into one mixed distribution. hurdle_quantiles()
maps a requested level tau to its quantile: if tau <= P(f_pisn=0)=1-pi the
quantile is exactly 0, otherwise it interpolates the positive-component
quantile at the rescaled level (tau - (1-pi)) / pi.

All v4_1 helpers and the standard QRF + mask-aware QRF sections are kept
UNCHANGED for comparison. The hurdle model, its held-out evaluation (AUC /
Brier / CRPS / pinball / coverage), the full-data Li+22 application, and the
hurdle-specific exports + plots are ADDED below the feature-importance block.
Outputs are written to src/randomForest/results/test_v4_2/.
"""

import os
import sys
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
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.metrics import r2_score, roc_auc_score, brier_score_loss  # noqa: E402
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
from src.load_data import load_obs_li  # noqa: E402

DATA_DIR = os.path.join(ROOT, "src", "randomForest", "data")
RESULTS_DIR = os.path.join(ROOT, "src", "randomForest", "results", "test_v4_2")
CACHE_PATH = os.path.join(DATA_DIR, "dataset.npz")
os.makedirs(RESULTS_DIR, exist_ok=True)

QUANTILES = [0.05, 0.16, 0.5, 0.84, 0.95]

OBS_ELEMENTS = ["C", "Na", "Mg", "Si", "Ca", "Cr", "Mn", "Fe", "Co", "Ni", "Zn"]
ELEMENT_ION = {
    "C": "C", "Na": "NaI", "Mg": "MgI", "Si": "SiI", "Ca": "CaI", "Cr": "CrI",
    "Mn": "MnI", "Fe": "FeI", "Co": "CoI", "Ni": "NiI", "Zn": "ZnI",
}
NON_FE = [el for el in OBS_ELEMENTS if el != "Fe"]
PAIRS = list(combinations(OBS_ELEMENTS, 2))                 # 55
RATIO_NAMES = [f"[{x}/{y}]" for x, y in PAIRS]
IND_NAMES = [f"ind[{el}]" for el in NON_FE]
EL_TO_RATIOS = {el: [j for j, (x, y) in enumerate(PAIRS) if el in (x, y)] for el in OBS_ELEMENTS}

# mask-aware settings
M_AUG = 5          # masked copies per synthetic star (+1 complete copy)
K_SEEDS = 5        # mask-seed ensemble size
SEEDS = [777 + i for i in range(K_SEEDS)]


def get_xfe(star, elem):
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


def hurdle_quantiles(pi, q_pos_grid, grid_q, quantiles):
    # pi: (n,) P(f_pisn>0). q_pos_grid: (n,m) positive-component quantiles
    # at grid_q levels (m ascending). grid_q: (m,). quantiles: list of tau.
    n = pi.shape[0]
    q_out = np.zeros((n, len(quantiles)))
    for k, tau in enumerate(quantiles):
        p0 = 1.0 - pi            # P(f_pisn=0)
        u = np.clip((tau - p0) / np.where(pi > 0, pi, 1e-12), 0.0, 1.0)
        interp = np.array([np.interp(u[i], grid_q, q_pos_grid[i]) for i in range(n)])
        q_out[:, k] = np.where(tau <= p0, 0.0, interp)
    return q_out


def _crps(y_true, q_grid, tau_grid):
    # q_grid: (n,m) quantiles at levels tau_grid (m,). Pinball decomposition.
    diff = y_true[:, None] - q_grid
    pin = diff * (tau_grid[None, :] - (diff < 0))
    return float(2 * np.mean(pin))


# ============================================================================
# Load training data, restrict to Li+22-observable ratios (55 features)
# ============================================================================
X_full, y, diag = load_dataset(CACHE_PATH)
full_names = list(diag["feature_names"])
X_obs_full = subset_columns(X_full, full_names, RATIO_NAMES)   # (2000, 55) complete
imp = SimpleImputer(strategy="median").fit(X_obs_full)         # medians from complete data
X_imp = imp.transform(X_obs_full)

stars = load_obs_li()
N = len(stars)
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
# standard reduced QRF
qrf_std_bench = _new_qrf(cfg)
qrf_std_bench.fit(X_imp[train_idx], y[train_idx])
r2_s = r2_score(y[test_idx], qrf_std_bench.predict(X_imp[test_idx], 0.5))

# mask-aware (single augmented seed, for speed) evaluated on a masked test set
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
# standard reduced QRF (full data)
qrf_std = _new_qrf(cfg)
qrf_std.fit(X_imp, y)

# mask-aware augmented + seed-ensemble (full data)
X_obs, feh, teff, logg = build_obs_features(stars)
X_obs_imp = imp.transform(X_obs)
ind_obs = indicators_from_stars(stars)
X_obs_feat = np.hstack([X_obs_imp, ind_obs])

preds_q = np.zeros((K_SEEDS, N, len(QUANTILES)))
imp_seeds = np.zeros((K_SEEDS, 65))
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
X_comp = X_obs[complete]                 # no NaN -> imputer is a no-op
q_comp = np.clip(qrf_std.predict(X_comp, quantiles=QUANTILES), 0.0, 1.0)
pred_comp = q_comp[:, 2]
feh_comp, teff_comp, logg_comp = feh[complete], teff[complete], logg[complete]

print(f"\n===== 1. complete-star subset (N={int(complete.sum())}, no imputation) =====")
print(f"  mean={pred_comp.mean():.3f}  median={np.median(pred_comp):.3f}  "
      f"min={pred_comp.min():.3f}  max={pred_comp.max():.3f}")
print(f"  frac(f<0.1)={(pred_comp < 0.1).mean():.3f}  "
      f"frac(f<0.5)={(pred_comp < 0.5).mean():.3f}  frac(f>0.5)={(pred_comp > 0.5).mean():.3f}")

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
    print(f"  {name:<16} {int(m.sum()):>4} {pred_std[m].mean():>9.3f} "
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
print("\n===== mask-aware feature importances (top 12, ensemble-mean) =====")
for i in order[:12]:
    nm = RATIO_NAMES[i] if i < 55 else IND_NAMES[i - 55]
    print(f"  {nm:<10} {imp_m[i]:.4f}")
ind_sum = imp_m[55:].sum()
print(f"  (total indicator importance = {ind_sum:.4f})")

# ============================================================================
# 4. Hurdle (two-stage zero-inflated) model
# ============================================================================
# --- construction ---
grid_q = np.linspace(0.0, 1.0, 201)
y_bin = (y > 0).astype(int)
pos = y > 0
X_pos = X_obs_full[pos]
y_pos = y[pos]

clf = CalibratedClassifierCV(
    RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                           random_state=42, n_jobs=-1),
    method="isotonic", cv=5)
qrf_pos = _new_qrf(cfg)
qrf_pos.fit(X_pos, y_pos)

# ============================================================================
# 4a. Held-out evaluation (reuse train_idx / test_idx from the base split)
# ============================================================================
print("\n===== 4a. hurdle held-out evaluation =====")
clf.fit(X_imp[train_idx], y_bin[train_idx])
pi_te = clf.predict_proba(X_imp[test_idx])[:, 1]
print(f"  classifier AUC = {roc_auc_score(y_bin[test_idx], pi_te):.4f}  "
      f"Brier = {brier_score_loss(y_bin[test_idx], pi_te):.4f}")

qrf_pos.fit(X_obs_full[train_idx][y[train_idx] > 0], y[train_idx][y[train_idx] > 0])
q_pos_te = np.clip(qrf_pos.predict(X_imp[test_idx], quantiles=grid_q), 0, 1)
q_h_te = hurdle_quantiles(pi_te, q_pos_te, grid_q, QUANTILES)

q_std_te = qrf_std_bench.predict(X_imp[test_idx], quantiles=QUANTILES)

grid_crps = np.linspace(0.01, 0.99, 99)
q_h_crps = hurdle_quantiles(pi_te, q_pos_te, grid_q, grid_crps)
q_std_crps = qrf_std_bench.predict(X_imp[test_idx], quantiles=grid_crps)
print(f"  CRPS (hurdle) = {_crps(y[test_idx], q_h_crps, grid_crps):.5f}  "
      f"CRPS (std) = {_crps(y[test_idx], q_std_crps, grid_crps):.5f}")

pos_te = y[test_idx] > 0
print(f"  CRPS | y>0 (hurdle) = {_crps(y[test_idx][pos_te], q_h_crps[pos_te], grid_crps):.5f}  "
      f"CRPS | y>0 (std) = {_crps(y[test_idx][pos_te], q_std_crps[pos_te], grid_crps):.5f}")

print(f"  R2 (hurdle median) = {r2_score(y[test_idx], q_h_te[:, 2]):.4f}")

print("  pinball loss (lower is better):")
print(f"  {'tau':>6} {'hurdle':>10} {'std':>10}")
for k, tau in enumerate(QUANTILES):
    pb_h = np.mean((y[test_idx] - q_h_te[:, k]) * (tau - (y[test_idx] < q_h_te[:, k])))
    pb_s = np.mean((y[test_idx] - q_std_te[:, k]) * (tau - (y[test_idx] < q_std_te[:, k])))
    print(f"  {tau:>6.2f} {pb_h:>10.4f} {pb_s:>10.4f}")

cov_h = np.array([np.mean(y[test_idx] <= q_h_te[:, k]) for k in range(len(QUANTILES))])
cov_s = np.array([np.mean(y[test_idx] <= q_std_te[:, k]) for k in range(len(QUANTILES))])
print("  empirical coverage P(y <= q_tau):")
print(f"  {'tau':>6} {'hurdle':>10} {'std':>10}")
for k, tau in enumerate(QUANTILES):
    print(f"  {tau:>6.2f} {cov_h[k]:>10.3f} {cov_s[k]:>10.3f}")

# ============================================================================
# 4b. Full-data application (Li+22, all 385 + complete 118)
# ============================================================================
clf.fit(X_imp, y_bin)
qrf_pos.fit(X_pos, y_pos)

pi_obs = clf.predict_proba(X_obs_imp)[:, 1]
q_pos_obs = np.clip(qrf_pos.predict(X_obs_imp, quantiles=grid_q), 0, 1)
q_h_obs = hurdle_quantiles(pi_obs, q_pos_obs, grid_q, QUANTILES)

pi_comp = clf.predict_proba(X_comp)[:, 1]
q_pos_comp = np.clip(qrf_pos.predict(X_comp, quantiles=grid_q), 0, 1)
q_h_comp = hurdle_quantiles(pi_comp, q_pos_comp, grid_q, QUANTILES)

print("\n===== 4b. hurdle full-data summary =====")
print(f"  all 385:      hurdle q50 mean={q_h_obs[:, 2].mean():.3f}  "
      f"median={np.median(q_h_obs[:, 2]):.3f}  "
      f"frac<0.1={(q_h_obs[:, 2] < 0.1).mean():.3f}  "
      f"frac>0.5={(q_h_obs[:, 2] > 0.5).mean():.3f}  "
      f"mean(pi)={pi_obs.mean():.3f}  std q50 mean={pred_std.mean():.3f}")
print(f"  complete 118: hurdle q50 mean={q_h_comp[:, 2].mean():.3f}  "
      f"median={np.median(q_h_comp[:, 2]):.3f}  "
      f"frac<0.1={(q_h_comp[:, 2] < 0.1).mean():.3f}  "
      f"frac>0.5={(q_h_comp[:, 2] > 0.5).mean():.3f}  "
      f"mean(pi)={pi_comp.mean():.3f}  std q50 mean={pred_comp.mean():.3f}")

# --- hurdle CSV export ---
df_hurdle = pd.DataFrame({
    "star": labels,
    "FeH": feh, "Teff": teff, "logg": logg,
    "N_measured": n_meas,
    "has_Si": has_si.astype(int),
    "is_complete": complete.astype(int),
    "pi_hat": pi_obs,
    "f_pisn_hurdle_q05": q_h_obs[:, 0], "f_pisn_hurdle_q16": q_h_obs[:, 1],
    "f_pisn_hurdle_q50": q_h_obs[:, 2], "f_pisn_hurdle_q84": q_h_obs[:, 3],
    "f_pisn_hurdle_q95": q_h_obs[:, 4],
    "f_pisn_std_q50": pred_std,
})
df_hurdle.to_csv(os.path.join(RESULTS_DIR, "obs_li_hurdle.csv"), index=False)
print(f"\nexported {len(df_hurdle)} stars -> obs_li_hurdle.csv")

# --- hurdle plots ---
# hurdle f_pisn histogram (all 385)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(q_h_obs[:, 2], bins=np.arange(0, 1.001, 0.05), color="tab:cyan",
        alpha=0.85, edgecolor="white")
ax.axvline(0.5, color="k", ls="--", lw=1.2)
ax.set_xlabel("predicted f_pisn (hurdle, median)")
ax.set_ylabel("N stars")
ax.set_title(f"Li+22 ({N} stars): hurdle f_pisn  mean={q_h_obs[:, 2].mean():.2f}  "
             f"median={np.median(q_h_obs[:, 2]):.2f}  "
             f"f>0.5: {(q_h_obs[:, 2] > 0.5).mean() * 100:.1f}%")
_save(fig, "hurdle_fpisn_hist.png")

# hurdle vs standard scatter
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.scatter(pred_std[has_si], q_h_obs[:, 2][has_si], s=10, alpha=0.5,
           color="tab:blue", label=f"with Si ({int(has_si.sum())})")
ax.scatter(pred_std[~has_si], q_h_obs[:, 2][~has_si], s=10, alpha=0.5,
           color="tab:red", label=f"without Si ({int((~has_si).sum())})")
ax.set_xlabel("standard predicted f_pisn (median)")
ax.set_ylabel("hurdle predicted f_pisn (median)")
ax.set_title("hurdle vs standard")
ax.legend()
_save(fig, "hurdle_vs_std_scatter.png")

# coverage calibration
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
ax.plot(QUANTILES, cov_h, "o-", color="tab:cyan", label="hurdle")
ax.plot(QUANTILES, cov_s, "s-", color="tab:blue", label="standard")
ax.set_xlabel("nominal quantile level tau")
ax.set_ylabel("empirical coverage P(y <= q_tau)")
ax.set_title("coverage calibration (held-out)")
ax.legend()
_save(fig, "coverage_calibration.png")

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
df_all.to_csv(os.path.join(RESULTS_DIR, "obs_li_all.csv"), index=False)

df_comp = pd.DataFrame({
    "star": labels[complete],
    "FeH": feh_comp, "Teff": teff_comp, "logg": logg_comp,
    "f_pisn_q16": q_comp[:, 1], "f_pisn_q50": q_comp[:, 2], "f_pisn_q84": q_comp[:, 3],
})
df_comp = df_comp.sort_values("f_pisn_q50", ascending=False).reset_index(drop=True)
df_comp.to_csv(os.path.join(RESULTS_DIR, "obs_li_complete.csv"), index=False)
print(f"\nexported {len(df_all)} stars -> obs_li_all.csv")
print(f"exported {len(df_comp)} complete stars -> obs_li_complete.csv")

# ============================================================================
# Plots
# ============================================================================
# complete-star histogram
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(pred_comp, bins=20, range=(0, 1), color="tab:orange", alpha=0.8)
ax.axvline(0.5, color="k", ls="--", lw=1)
ax.set_xlabel("predicted f_pisn (median)")
ax.set_ylabel("N stars")
ax.set_title(f"Li+22 complete stars ({int(complete.sum())}): f_pisn")
_save(fig, "complete_fpisn_hist.png")

# complete-star f_pisn vs [Fe/H]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(feh_comp, pred_comp, s=15, alpha=0.6, color="tab:orange")
ax.set_xlabel("[Fe/H]")
ax.set_ylabel("predicted f_pisn (median)")
ax.set_title("Li+22 complete stars: f_pisn vs [Fe/H]")
_save(fig, "complete_fpisn_vs_feh.png")

# all-star f_pisn vs [Fe/H]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(feh, pred_std, s=15, alpha=0.6, color="tab:orange", label="standard")
ax.scatter(feh, pred_mask, s=15, alpha=0.6, color="tab:blue", label="mask-aware")
ax.set_xlabel("[Fe/H]")
ax.set_ylabel("predicted f_pisn (median)")
ax.set_title(f"Li+22 all stars ({N}): f_pisn vs [Fe/H]")
ax.legend()
_save(fig, "obs_fpisn_vs_feh.png")

# f_pisn vs N_measured
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(n_meas + np.random.uniform(-0.2, 0.2, N), pred_std, s=8,
           alpha=0.4, color="tab:blue", label="standard")
ks = sorted(set(n_meas))
ax.plot(ks, [pred_std[n_meas == k].mean() for k in ks], "o-", color="tab:red",
        label="group mean")
ax.set_xlabel("N elements measured (of 10)")
ax.set_ylabel("predicted f_pisn (median)")
ax.set_title("Li+22: f_pisn vs completeness")
ax.legend()
_save(fig, "fpisn_vs_nmeasured.png")

# with-Si vs without-Si
fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.boxplot([pred_std[has_si], pred_std[~has_si]],
           tick_labels=["with Si", "without Si"])
ax.set_ylabel("predicted f_pisn (median, standard model)")
ax.set_title(f"Si present ({int(has_si.sum())}) vs absent ({int((~has_si).sum())})")
_save(fig, "si_vs_nosi_box.png")

# standard vs mask-aware scatter (ensemble)
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

# mask-aware f_pisn histogram (main plot, linear scale only)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(pred_mask, bins=np.arange(0, 1.001, 0.05), color="tab:purple", alpha=0.85, edgecolor="white")
ax.axvline(0.5, color="k", ls="--", lw=1.2)
ax.set_xlabel("predicted f_pisn (mask-aware, median)")
ax.set_ylabel("N stars")
ax.set_title(f"Li+22 ({N} stars): mask-aware f_pisn")
ax.text(0.97, 0.95, f"mean={pred_mask.mean():.2f}\nmedian={np.median(pred_mask):.2f}\n"
        f"f>0.5: {(pred_mask > 0.5).mean() * 100:.1f}%",
        transform=ax.transAxes, ha="right", va="top", fontsize=9)
_save(fig, "mask_fpisn_hist.png")

# mask-aware feature importance (top 20, ensemble-mean)
k = 20
top = order[:k]
names_top = [RATIO_NAMES[i] if i < 55 else IND_NAMES[i - 55] for i in top]
fig, ax = plt.subplots(figsize=(8, 8))
ax.barh(np.arange(k)[::-1], imp_m[top][::-1], color="tab:green")
ax.set_yticks(np.arange(k)[::-1])
ax.set_yticklabels([names_top[i] for i in range(k)][::-1], fontsize=8)
ax.set_xlabel("Gini importance (ensemble mean)")
ax.set_title("Mask-aware QRF feature importances (top 20)")
_save(fig, "mask_feature_importance.png")

# mask-aware seed uncertainty (per-star std of q50)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(q_mask_se, bins=30, color="tab:purple", alpha=0.8)
ax.set_xlabel("per-star seed std of f_pisn(q50)")
ax.set_ylabel("N stars")
ax.set_title(f"mask-aware missingness uncertainty (K={K_SEEDS} seeds)")
_save(fig, "mask_seed_uncertainty.png")

print(f"\ndone. outputs in {RESULTS_DIR}")
