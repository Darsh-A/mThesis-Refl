"""Evaluate the f_pisn Random Forest on the observed Li+22 catalogue.

The training model uses pairwise [X/Y] abundance ratios over 14 elements:
    C, O, Na, Mg, Al, Si, Ca, Cr, Mn, Fe, Co, Ni, Cu, Zn.
The Li+22 very metal-poor star catalogue measures a subset of these
(missing O, Al, Cu), so evaluation proceeds as follows:

  1. The 11 elements observed in Li+22 are kept: C, Na, Mg, Si, Ca, Cr, Mn,
     Fe, Co, Ni, Zn  ->  C(11,2) = 55 pairwise [X/Y] features.
  2. A companion RF (and QRF) is trained on the *same* cached dataset but
     restricted to those 55 features, so predictions are on the exact feature
     space that real stars provide.  (The full 14-element model's top feature
     [Al/Si] is NOT observable in Li+22; the reduced model re-ranks the
     observable ratios.)
  3. Observed features are built as [X/Y] = [X/Fe]_X - [X/Fe]_Y, using each
     star's [X/Fe] (XFe) columns.  Fe is the reference ([Fe/Fe] = 0), so the
     [X/Fe] ratios are read directly and non-Fe pairs are pairwise differences.
  4. Missing / upper-limit abundances are treated as NaN and imputed with the
     training median before prediction (completeness is reported separately).

Plots written to src/randomForest/diagnostic/new/.

Run:
    python src/randomForest/evaluate_obs.py
"""

import os
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from src.load_data import load_obs_li
from src.randomForest.train_rf import CONFIG, load_dataset
from src.randomForest.quantile_rf import QuantileRegressionForest

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG_DIR = os.path.join(HERE, "diagnostic", "new")
os.makedirs(DIAG_DIR, exist_ok=True)

# Training element order (from dataset._abundance_elements), restricted to the
# elements present in Li+22.
OBS_ELEMENTS = ["C", "Na", "Mg", "Si", "Ca", "Cr", "Mn", "Fe", "Co", "Ni", "Zn"]

# Ion key in load_obs_li()'s "abundances" dict for each element's [X/Fe].
ELEMENT_ION = {
    "C": "C",
    "Na": "NaI",
    "Mg": "MgI",
    "Si": "SiI",
    "Ca": "CaI",
    "Cr": "CrI",
    "Mn": "MnI",
    "Fe": "FeI",   # reference; [Fe/Fe]=0 by construction
    "Co": "CoI",
    "Ni": "NiI",
    "Zn": "ZnI",
}


def obs_feature_names():
    return [f"[{x}/{y}]" for x, y in combinations(OBS_ELEMENTS, 2)]


def get_xfe(star: dict, elem: str) -> float:
    """Return the star's [X/Fe] (XFe) for `elem`, or NaN if missing/upper-limit.

    Fe is the reference: [Fe/Fe] = 0.
    """
    if elem == "Fe":
        return 0.0
    rec = star["abundances"].get(ELEMENT_ION[elem])
    if rec is None:
        return np.nan
    if rec.get("l_XFe") == "<":          # upper limit -> treat as missing
        return np.nan
    xfe = rec.get("XFe")
    return np.nan if xfe is None else float(xfe)


def build_obs_features(stars: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X_obs, feh, teff) for all stars; NaN where an element is missing."""
    names = obs_feature_names()
    n = len(stars)
    X = np.full((n, len(names)), np.nan)
    feh = np.full(n, np.nan)
    teff = np.full(n, np.nan)

    for i, star in enumerate(stars):
        xfe = {el: get_xfe(star, el) for el in OBS_ELEMENTS}
        for j, (x, y) in enumerate(combinations(OBS_ELEMENTS, 2)):
            vx, vy = xfe[x], xfe[y]
            if np.isnan(vx) or np.isnan(vy):
                X[i, j] = np.nan
            else:
                X[i, j] = vx - vy
        p = star.get("params", {})
        feh[i] = p.get("FeH", np.nan)
        teff[i] = p.get("Teff", np.nan)

    return X, feh, teff


def _subset_columns(X: np.ndarray, full_names: list[str], subset_names: list[str]) -> np.ndarray:
    idx = [full_names.index(nm) for nm in subset_names]
    return X[:, idx]


def main():
    # --- training data (55-feature subset of the cached dataset) ---
    X_full, y, diag = load_dataset(CONFIG["cache_path"])
    full_names = list(diag["feature_names"])
    names = obs_feature_names()
    X_train_full = _subset_columns(X_full, full_names, names)
    X_train_full = SimpleImputer(strategy="median").fit_transform(X_train_full)

    # identical split to train_rf.py
    rng = np.random.default_rng(CONFIG["seed"])
    n = len(y)
    idx = rng.permutation(n)
    n_test = int(n * CONFIG["test_size"])
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    X_tr, X_te = X_train_full[train_idx], X_train_full[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    rf = RandomForestRegressor(
        n_estimators=CONFIG["n_estimators"],
        min_samples_leaf=CONFIG["min_samples_leaf"],
        random_state=CONFIG["random_state"],
        n_jobs=CONFIG["n_jobs"],
    )
    rf.fit(X_tr, y_tr)
    from sklearn.metrics import r2_score, mean_squared_error
    y_pred_te = rf.predict(X_te)
    print(f"=== Reduced (11-element, {len(names)}-feature) RF ===")
    print(f"  test R2  = {r2_score(y_te, y_pred_te):.4f}")
    print(f"  test RMSE= {float(np.sqrt(mean_squared_error(y_te, y_pred_te))):.4f}")

    print("\n=== Reduced-model feature importances (top 20) ===")
    order = np.argsort(rf.feature_importances_)[::-1]
    for rank, i in enumerate(order[:20], 1):
        print(f"    {rank:>3}. {names[i]:<10} {rf.feature_importances_[i]:.4f}")

    # --- observed stars ---
    stars = load_obs_li()
    X_obs, feh, teff = build_obs_features(stars)
    print(f"\n=== Li+22 catalogue: {len(stars)} stars ===")
    per_elem = {el: int(sum(1 for s in stars if not np.isnan(get_xfe(s, el))))
                for el in OBS_ELEMENTS if el != "Fe"}
    print("  measured [X/Fe] per element:", per_elem)
    n_complete = int(sum(1 for s in stars
                         if all(not np.isnan(get_xfe(s, el)) for el in OBS_ELEMENTS)))
    print(f"  stars with all 11 elements measured: {n_complete}")

    # impute obs NaN with training median (fit on training data)
    imp = SimpleImputer(strategy="median").fit(X_train_full)
    X_obs_imp = imp.transform(X_obs)

    pred_obs = rf.predict(X_obs_imp)
    pred_obs = np.clip(pred_obs, 0.0, 1.0)

    # QRF uncertainty on obs
    qrf = QuantileRegressionForest(
        n_estimators=CONFIG["n_estimators"],
        min_samples_leaf=CONFIG["min_samples_leaf"],
        random_state=CONFIG["random_state"],
        n_jobs=CONFIG["n_jobs"],
    )
    qrf.fit(X_tr, y_tr)
    q_obs = qrf.predict(X_obs_imp, quantiles=[0.16, 0.5, 0.84])

    # --- summary ---
    print("\n=== Predicted f_pisn for observed stars ===")
    print(f"  mean = {pred_obs.mean():.3f}   median = {np.median(pred_obs):.3f}")
    print(f"  min = {pred_obs.min():.3f}   max = {pred_obs.max():.3f}")
    print(f"  frac(f_pisn < 0.1) = {(pred_obs < 0.1).mean():.3f}")
    print(f"  frac(f_pisn < 0.5) = {(pred_obs < 0.5).mean():.3f}")

    # --- plots ---
    # 1. histogram of predicted f_pisn
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(pred_obs, bins=30, range=(0, 1), color="tab:blue", alpha=0.8)
    ax.axvline(0.5, color="k", ls="--", lw=1, label="f_pisn = 0.5")
    ax.set_xlabel("predicted f_pisn")
    ax.set_ylabel("N stars")
    ax.set_title(f"Li+22: predicted f_pisn (N={len(stars)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(DIAG_DIR, "obs_eval_fpisn_hist.png"))
    plt.close(fig)

    # 2. f_pisn vs [Fe/H]
    ok = np.isfinite(feh)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(feh[ok], pred_obs[ok], s=12, alpha=0.5, color="tab:blue")
    ax.set_xlabel("[Fe/H]")
    ax.set_ylabel("predicted f_pisn")
    ax.set_title("Li+22: predicted f_pisn vs [Fe/H]")
    fig.tight_layout()
    fig.savefig(os.path.join(DIAG_DIR, "obs_eval_fpisn_vs_feh.png"))
    plt.close(fig)

    # 3. f_pisn vs Teff
    okt = np.isfinite(teff)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(teff[okt], pred_obs[okt], s=12, alpha=0.5, color="tab:red")
    ax.set_xlabel("Teff (K)")
    ax.set_ylabel("predicted f_pisn")
    ax.set_title("Li+22: predicted f_pisn vs Teff")
    fig.tight_layout()
    fig.savefig(os.path.join(DIAG_DIR, "obs_eval_fpisn_vs_teff.png"))
    plt.close(fig)

    # 4. reduced-model feature importance
    fig, ax = plt.subplots(figsize=(8, 8))
    k = 25
    ax.barh(np.arange(k)[::-1], rf.feature_importances_[order[:k]][::-1], color="tab:green")
    ax.set_yticks(np.arange(k)[::-1])
    ax.set_yticklabels([names[i] for i in order[:k]][::-1], fontsize=8)
    ax.set_xlabel("Gini importance")
    ax.set_title("Reduced (Li+22-observable) model feature importances")
    fig.tight_layout()
    fig.savefig(os.path.join(DIAG_DIR, "obs_eval_feature_importance.png"))
    plt.close(fig)

    # 5. QRF median + 68% band vs [Fe/H]
    so = np.argsort(feh)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(feh[so], q_obs[:, 0][so], q_obs[:, 2][so],
                    color="tab:blue", alpha=0.25, label="68% CI")
    ax.plot(feh[so], q_obs[:, 1][so], "o-", ms=3, color="tab:blue", label="median")
    ax.set_xlabel("[Fe/H]")
    ax.set_ylabel("predicted f_pisn")
    ax.set_title("Li+22: QRF f_pisn vs [Fe/H] (with uncertainty)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(DIAG_DIR, "obs_eval_qrf_vs_feh.png"))
    plt.close(fig)

    print(f"\nsaved plots to {DIAG_DIR}")


if __name__ == "__main__":
    main()
