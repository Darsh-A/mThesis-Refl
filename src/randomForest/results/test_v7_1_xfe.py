"""[X/Fe]-feature model + mass-gated review (test_v7.1).

The deployed model uses 91 pairwise [X/Y] ratios; a single element's signal is
spread over its ~13 ratios, which can dilute feature importance.  This test
switches to the 13 [X/Fe] ratios (one per non-Fe element) so each element maps
to exactly one feature, then re-ranks importance and re-checks held-out skill.

Because the PISN yield signal is strongly mass dependent, the skill and
importance are also reviewed mass-gated: combined, then the PISN samples split
at the dominant yield jump (see test_v7_1_mass_dependence.py), which falls
between the 220 and 240 Msun Takahashi grid models -> split at 230 Msun.

[ X/Fe ] is reconstructed exactly from the cached pairwise ratios:
  * [X/Fe] for X lighter than Fe is already a stored column.
  * [X/Fe] = -[Fe/X] for X heavier than Fe.

Datasets (already cached, no regeneration):
  * 4_3 -> dataset_TakaLimon_realisticNoise_v4_3.npz  (Takahashi + Limongi18 + noise)
  * 5   -> dataset_Nomotov5.npz                       (Takahashi + Nomoto13, clean)

Outputs (src/randomForest/results/test_v7_1/):
  * xfe_feature_importance.png  13 [X/Fe] Gini importances
  * mass_gated_importance.png   [X/Fe] importance in the <230 vs >=230 PISN regimes
  * xfe_metrics.json / .txt     combined + mass-gated held-out skill
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score, cross_val_predict
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
OUT_DIR = os.path.join(ROOT, "src", "randomForest", "results", "test_v7_1")
os.makedirs(OUT_DIR, exist_ok=True)

ELEMENTS = ["C", "O", "Na", "Mg", "Al", "Si", "Ca", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"]
NON_FE = [e for e in ELEMENTS if e != "Fe"]
MASS_SPLIT = 230.0  # dominant Takahashi NR yield jump (220->240 grid models)

DATASETS = {
    "4_3": os.path.join(DATA_DIR, "dataset_TakaLimon_realisticNoise_v4_3.npz"),
    "5": os.path.join(DATA_DIR, "dataset_Nomotov5.npz"),
}


def build_xfe(X_full, full_names, fe="Fe"):
    """Reconstruct the 13 [X/Fe] columns from the cached pairwise ratios."""
    Xxfe = np.empty((X_full.shape[0], len(NON_FE)))
    for k, x in enumerate(NON_FE):
        col = sign = None
        for j, nm in enumerate(full_names):
            a, b = nm.strip("[]").split("/")
            if {a, b} == {x, fe}:
                col = j
                sign = 1.0 if a == x else -1.0  # [X/Fe] = +[X/Fe] or -[Fe/X]
                break
        if col is None:
            raise KeyError(f"no {x}/{fe} ratio in feature names")
        Xxfe[:, k] = sign * X_full[:, col]
    return Xxfe, [f"[{x}/Fe]" for x in NON_FE]


def reg_metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "spearman_rho": float(spearmanr(y_true, y_pred)[0]),
        "bias": float(np.mean(y_pred - y_true)),
        "n": int(len(y_true)),
    }


def _new_rf(cfg):
    return RandomForestRegressor(
        n_estimators=cfg["n_estimators"],
        min_samples_leaf=cfg["min_samples_leaf"],
        random_state=cfg["random_state"],
        n_jobs=cfg["n_jobs"],
    )


def _new_qrf(cfg):
    return QuantileRegressionForest(
        n_estimators=cfg["n_estimators"], min_samples_leaf=cfg["min_samples_leaf"],
        random_state=cfg["random_state"], n_jobs=cfg["n_jobs"])


def run_dataset(dataset_id, cache):
    X_full, y, diag = load_dataset(cache)
    full_names = list(diag["feature_names"])
    Xxfe, xfe_names = build_xfe(X_full, full_names)
    imp = SimpleImputer(strategy="median")
    Xxfe = imp.fit_transform(Xxfe)

    pisn_mass = np.asarray(diag["pisn_mass"], dtype=float)
    is_pisn = np.isfinite(pisn_mass)  # SN-only samples carry NaN pisn_mass

    cfg = CONFIG
    rng = np.random.default_rng(cfg["seed"])
    n = len(y)
    idx = rng.permutation(n)
    n_test = int(n * cfg["test_size"])
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    # ---- combined model (all samples: PISN + SN-only) ----
    qrf = _new_qrf(cfg)
    qrf.fit(Xxfe[train_idx], y[train_idx])
    y_te = qrf.predict(Xxfe[test_idx], 0.5)
    combined = reg_metrics(y[test_idx], y_te)
    combined["importance"] = {
        nm: float(v) for nm, v in zip(xfe_names, qrf.rf.feature_importances_)
    }

    # combined model's skill on the PISN-only held-out, split by mass regime
    pisn_te = is_pisn[test_idx]
    combined_by_mass = {}
    for label, m in [("low<230", pisn_mass < MASS_SPLIT), ("high>=230", pisn_mass >= MASS_SPLIT)]:
        sel = test_idx[pisn_te & m[test_idx]]
        combined_by_mass[label] = (
            reg_metrics(y[sel], qrf.predict(Xxfe[sel], 0.5)) if len(sel) else None
        )

    # ---- mass-gated review via out-of-fold predictions of a point RF ----
    # out-of-fold f_pisn for every PISN sample -> stable per-regime skill
    rf_full = _new_rf(cfg)
    oof = cross_val_predict(rf_full, Xxfe, y, cv=5)
    mass_skill = {}
    for label, m in [("low<230", pisn_mass < MASS_SPLIT), ("high>=230", pisn_mass >= MASS_SPLIT)]:
        sel = is_pisn & m
        mass_skill[label] = reg_metrics(y[sel], oof[sel])

    # ---- regime-specific [X/Fe] importance (PISN samples only) ----
    regime_imp = {}
    for label, m in [("low<230", pisn_mass < MASS_SPLIT), ("high>=230", pisn_mass >= MASS_SPLIT)]:
        sel = is_pisn & m
        rf = _new_rf(cfg).fit(Xxfe[sel], y[sel])
        regime_imp[label] = {
            nm: float(v) for nm, v in zip(xfe_names, rf.feature_importances_)
        }
        # learnability of f_pisn within the regime alone (5-fold CV R2)
        regime_imp[label]["_cv_r2"] = float(
            cross_val_score(rf, Xxfe[sel], y[sel], cv=5, scoring="r2").mean()
        )

    print(f"\n[{dataset_id}] combined held-out R2 = {combined['r2']:.4f}  "
          f"RMSE = {combined['rmse']:.4f}  (n_test={combined['n']})")
    print(f"  mass-gated (out-of-fold, PISN only):")
    for label in ("low<230", "high>=230"):
        m = mass_skill[label]
        print(f"    {label:<12} R2={m['r2']:+.4f}  RMSE={m['rmse']:.4f}  "
              f"rho={m['spearman_rho']:+.3f}  n={m['n']}")
    print(f"  regime-specific CV R2 (PISN only): "
          f"low<230={regime_imp['low<230']['_cv_r2']:.4f}  "
          f"high>=230={regime_imp['high>=230']['_cv_r2']:.4f}")

    return {
        "dataset_id": dataset_id,
        "xfe_names": xfe_names,
        "combined": combined,
        "combined_by_mass": combined_by_mass,
        "mass_skill": mass_skill,
        "regime_importance": regime_imp,
        "n_total": int(n),
        "n_pisn": int(is_pisn.sum()),
        "n_low": int((is_pisn & (pisn_mass < MASS_SPLIT)).sum()),
        "n_high": int((is_pisn & (pisn_mass >= MASS_SPLIT)).sum()),
    }


def plot(res):
    xfe = res["xfe_names"]

    # 1. [X/Fe] Gini importance
    imp = res["combined"]["importance"]
    order = sorted(xfe, key=lambda nm: imp[nm], reverse=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(np.arange(len(order)), [imp[nm] for nm in order], color="tab:green")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_ylabel("Gini importance")
    ax.set_title(f"test_v7.1 [{res['dataset_id']}]: [X/Fe] feature importance")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"xfe_feature_importance_{res['dataset_id']}.png"), dpi=130)
    plt.close(fig)

    # 2. regime-specific importance side by side
    lo = res["regime_importance"]["low<230"]
    hi = res["regime_importance"]["high>=230"]
    x = np.arange(len(xfe))
    w = 0.4
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, [lo[nm] for nm in xfe], w, label="pisn_mass < 230", color="tab:blue")
    ax.bar(x + w / 2, [hi[nm] for nm in xfe], w, label="pisn_mass >= 230", color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels(xfe, rotation=45, ha="right")
    ax.set_ylabel("Gini importance")
    ax.set_title(f"test_v7.1 [{res['dataset_id']}]: [X/Fe] importance by PISN mass regime")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"mass_gated_importance_{res['dataset_id']}.png"), dpi=130)
    plt.close(fig)


def main():
    all_results = {}
    for ds, cache in DATASETS.items():
        all_results[ds] = run_dataset(ds, cache)
        plot(all_results[ds])

    with open(os.path.join(OUT_DIR, "xfe_metrics.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    lines = ["[X/Fe]-feature model + mass-gated review (test_v7.1)",
             "Combined held-out = QRF median; mass-gated/CV = point RF (same tree ensemble).",
             "Mass split: pisn_mass < 230 vs >= 230 (dominant Takahashi NR yield jump, 220->240).",
             ""]
    for ds, r in all_results.items():
        c = r["combined"]
        lines.append(f"== {ds}: combined held-out R2={c['r2']:.4f} RMSE={c['rmse']:.4f} "
                     f"rho={c['spearman_rho']:+.3f} (n_test={c['n']}) ==")
        lines.append(f"  n_total={r['n_total']}  n_pisn={r['n_pisn']}  "
                     f"low<230={r['n_low']}  high>=230={r['n_high']}")
        lines.append("  combined [X/Fe] importance:")
        for nm in sorted(r["xfe_names"], key=lambda k: -c["importance"][k]):
            lines.append(f"    {nm:<10} {c['importance'][nm]:.4f}")
        lines.append("  mass-gated skill (out-of-fold, PISN only):")
        for label in ("low<230", "high>=230"):
            m = r["mass_skill"][label]
            lines.append(f"    {label:<12} R2={m['r2']:+.4f} RMSE={m['rmse']:.4f} "
                         f"rho={m['spearman_rho']:+.3f} n={m['n']}")
        lines.append("  regime-specific CV R2 (PISN only): "
                     f"low<230={r['regime_importance']['low<230']['_cv_r2']:.4f}  "
                     f"high>=230={r['regime_importance']['high>=230']['_cv_r2']:.4f}")
        lines.append("")

    with open(os.path.join(OUT_DIR, "xfe_metrics.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\ndone. outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
