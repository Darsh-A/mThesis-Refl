"""Grouped permutation importance (test_v7).

The deployed f_pisn model is trained on 91 pairwise [X/Y] ratios built from 14
elements.  A per-feature permutation importance spreads each element's true
signal across its ~13 ratios, so an element that matters can look unimportant
on every individual ratio.  This test permutes *all* ratios sharing an element
together (a "grouped" permutation), which preserves the within-element
correlation while destroying the element's link to f_pisn, and reports the
resulting element-level importance.

Groups of interest (flagged by the user as suspected dilution): Cu, Zn, Al.
Every element is still reported for context.

Datasets (already cached, no regeneration):
  * 4_3 -> dataset_TakaLimon_realisticNoise_v4_3.npz  (Takahashi + Limongi18 + noise)
  * 5   -> dataset_Nomotov5.npz                       (Takahashi + Nomoto13, clean)

Model: a standard RandomForestRegressor (point prediction) so permutation
scoring is cheap; the tree structure and Gini importance are identical to the
QRF the pipeline deploys.

Outputs (src/randomForest/results/test_v7/):
  * grouped_permutation_importance.png   element-level grouped importance (Cu/Zn/Al highlighted)
  * grouped_vs_per_feature.png           grouped importance vs sum of per-feature importance
  * grouped_permutation.json / .txt
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
from sklearn.metrics import r2_score


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

from src.randomForest.train_rf import CONFIG, load_dataset  # noqa: E402

DATA_DIR = os.path.join(ROOT, "src", "randomForest", "data")
OUT_DIR = os.path.join(ROOT, "src", "randomForest", "results", "test_v7")
os.makedirs(OUT_DIR, exist_ok=True)

# The 14 elements the deployed model is built from (in atomic-number order, the
# exact order used to build the pairwise feature names).
ELEMENTS = ["C", "O", "Na", "Mg", "Al", "Si", "Ca", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"]
HIGHLIGHT = {"Cu", "Zn", "Al"}

DATASETS = {
    "4_3": os.path.join(DATA_DIR, "dataset_TakaLimon_realisticNoise_v4_3.npz"),
    "5": os.path.join(DATA_DIR, "dataset_Nomotov5.npz"),
}

N_REPEATS = 15


def element_groups(feature_names, elements):
    """Map element -> column indices of every ratio containing that element."""
    groups = {}
    for el in elements:
        cols = []
        for j, nm in enumerate(feature_names):
            if el in nm.strip("[]").split("/"):
                cols.append(j)
        groups[el] = cols
    return groups


def grouped_permutation_importance(rf, X_te, y_te, groups, rng, n_repeats=N_REPEATS):
    """Baseline R2 minus permuted R2 for each element group (higher = more important).

    A single row permutation is applied to every column of a group at once, so
    the group's internal correlation is preserved while its tie to f_pisn is cut.
    """
    baseline = float(r2_score(y_te, rf.predict(X_te)))
    out = {}
    n = len(y_te)
    for gname, cols in groups.items():
        drops = np.empty(n_repeats)
        for r in range(n_repeats):
            Xp = X_te.copy()
            perm = rng.permutation(n)
            Xp[:, cols] = Xp[perm][:, cols]
            drops[r] = baseline - r2_score(y_te, rf.predict(Xp))
        out[gname] = (float(drops.mean()), float(drops.std()))
    return baseline, out


def per_feature_permutation_importance(rf, X_te, y_te, rng, n_repeats=N_REPEATS):
    """Single-feature permutation importance, summed to element level for comparison."""
    baseline = float(r2_score(y_te, rf.predict(X_te)))
    n_feat = X_te.shape[1]
    n = len(y_te)
    imp = np.empty(n_feat)
    for j in range(n_feat):
        drops = np.empty(n_repeats)
        for r in range(n_repeats):
            Xp = X_te.copy()
            Xp[:, j] = Xp[rng.permutation(n), j]
            drops[r] = baseline - r2_score(y_te, rf.predict(Xp))
        imp[j] = float(drops.mean())
    return baseline, imp


def run_dataset(dataset_id, cache):
    X_full, y, diag = load_dataset(cache)
    full_names = list(diag["feature_names"])
    X = SimpleImputer(strategy="median").fit_transform(X_full)
    groups = element_groups(full_names, ELEMENTS)

    cfg = CONFIG
    rng = np.random.default_rng(cfg["seed"])
    n = len(y)
    idx = rng.permutation(n)
    n_test = int(n * cfg["test_size"])
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    rf = RandomForestRegressor(
        n_estimators=cfg["n_estimators"],
        min_samples_leaf=cfg["min_samples_leaf"],
        random_state=cfg["random_state"],
        n_jobs=cfg["n_jobs"],
    ).fit(X[train_idx], y[train_idx])

    X_te, y_te = X[test_idx], y[test_idx]

    rng_pi = np.random.default_rng(0)
    baseline, grouped = grouped_permutation_importance(rf, X_te, y_te, groups, rng_pi)
    _, per_feat = per_feature_permutation_importance(rf, X_te, y_te, rng_pi)

    # sum per-feature importances to element level for the "dilution" comparison
    per_elem = {el: 0.0 for el in ELEMENTS}
    for el in ELEMENTS:
        per_elem[el] = float(sum(per_feat[j] for j in groups[el]))

    order = sorted(ELEMENTS, key=lambda e: grouped[e][0], reverse=True)

    print(f"\n[{dataset_id}] baseline R2 = {baseline:.4f}  "
          f"(n_train={len(train_idx)}, n_test={len(test_idx)})")
    print(f"  {'element':<6}{'grouped':>12}{'sum-per-feat':>16}{'diff':>10}")
    for el in order:
        g = grouped[el][0]
        s = per_elem[el]
        flag = "  <--" if el in HIGHLIGHT else ""
        print(f"  {el:<6}{g:>12.4f}{s:>16.4f}{g - s:>10.4f}{flag}")

    return {
        "dataset_id": dataset_id,
        "baseline_r2": baseline,
        "grouped": {el: {"mean": grouped[el][0], "std": grouped[el][1]} for el in ELEMENTS},
        "per_feature_sum": per_elem,
        "order": order,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }


def plot(results):
    # 1. grouped importance, all elements, Cu/Zn/Al highlighted
    fig, ax = plt.subplots(figsize=(8, 5))
    el = results["order"]
    vals = np.array([results["grouped"][e]["mean"] for e in el])
    errs = np.array([results["grouped"][e]["std"] for e in el])
    colors = ["tab:red" if e in HIGHLIGHT else "tab:blue" for e in el]
    ax.bar(np.arange(len(el)), vals, yerr=errs, color=colors, alpha=0.9)
    ax.set_xticks(np.arange(len(el)))
    ax.set_xticklabels(el)
    ax.set_ylabel("grouped permutation importance (R2 drop)")
    ax.set_title("test_v7: element-level grouped permutation importance")
    ax.axhline(0, color="k", lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "grouped_permutation_importance.png"), dpi=130)
    plt.close(fig)

    # 2. grouped vs sum-of-per-feature (dilution view)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(el))
    w = 0.4
    gv = [results["grouped"][e]["mean"] for e in el]
    sv = [results["per_feature_sum"][e] for e in el]
    ax.bar(x - w / 2, gv, w, label="grouped (permuted together)", color="tab:blue")
    ax.bar(x + w / 2, sv, w, label="sum of per-feature", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(el)
    ax.set_ylabel("importance (R2 drop)")
    ax.set_title("test_v7: grouped vs per-feature importance (dilution check)")
    ax.axhline(0, color="k", lw=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "grouped_vs_per_feature.png"), dpi=130)
    plt.close(fig)


def main():
    all_results = {}
    for ds, cache in DATASETS.items():
        all_results[ds] = run_dataset(ds, cache)

    with open(os.path.join(OUT_DIR, "grouped_permutation.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    lines = ["Grouped permutation importance (test_v7)",
             "Positive importance = R2 drop when permuting the element's ratios together.",
             "Higher grouped-than-sum-per-feature => signal diluted across ratios.", ""]
    for ds in DATASETS:
        r = all_results[ds]
        lines.append(f"== {ds}: baseline R2 = {r['baseline_r2']:.4f} "
                     f"(n_train={r['n_train']}, n_test={r['n_test']}) ==")
        for el in r["order"]:
            g = r["grouped"][el]
            s = r["per_feature_sum"][el]
            lines.append(f"  {el:<6} grouped={g['mean']:+.4f}(+/-{g['std']:.4f})  "
                         f"sum-per-feat={s:+.4f}  diff={g['mean'] - s:+.4f}")
        lines.append("")
    with open(os.path.join(OUT_DIR, "grouped_permutation.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    for ds in DATASETS:
        plot(all_results[ds])

    print(f"\ndone. outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
