"""Ground-truth single-feature and pairwise separability of PISN-dominated
vs CCSN-only stars, per PISN mass bin (test_v9).

Answers directly: *which abundance-ratio signs a PISN*, alone and in pairs.

For each cached dataset we reconstruct the single-ratio families
    FE = {[X/Fe] : X != Fe},  MG = {[X/Mg] : X != Mg}
from the cached pairwise [X/Y] columns, then compute:

  * single-feature metrics (signed AUC + separability, KS test, bootstrap CI)
  * pairwise metrics (5-fold CV logistic AUC over the FE U MG pool)

for four class definitions (positives = PISN-dominated, f_pisn >= 0.5, in a
mass bin; negatives = CCSN-only, f_pisn == 0):
    "210-230", ">=230", "all" (any mass), "<210".

Datasets (cached, no regeneration):
  * 4_3 -> dataset_TakaLimon_realisticNoise_v4_3.npz  (Zn set, 91 pairs)
  * 5   -> dataset_Nomotov5.npz                        (Zn set, 91 pairs)
  * L   -> dataset_Limongi_tpop42.npz                  (N set, 105 pairs)
  * N   -> dataset_Nomoto_tpop42.npz                   (N set, 105 pairs)

Outputs (src/randomForest/results/test_v9/):
  * single_feature_auc.csv / .json
  * pairwise_top_auc.csv / pairwise_matrices.json
  * single_auc_heatmap_<ds>.png, single_auc_bars_<ds>_<bin>.png,
    roc_curves_top_<ds>_<bin>.png, dist_top_elements_<ds>_<bin>.png,
    pair_auc_matrix_<ds>_<bin>.png, pair_top_<ds>_<bin>.png,
    scatter_best_pair_<ds>_<bin>.png, summary_best_single_vs_bin.png
  * signature_report.txt
"""

import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import ks_2samp


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

from src.randomForest.train_rf import load_dataset  # noqa: E402

DATA_DIR = os.path.join(ROOT, "src", "randomForest", "data")
OUT_DIR = os.path.join(ROOT, "src", "randomForest", "results", "test_v9")
os.makedirs(OUT_DIR, exist_ok=True)

_PLOT_SUBDIRS = {
    "single_auc_heatmap_": "01_single_heatmaps",
    "single_auc_bars_": "02_single_bars",
    "roc_curves_top_": "03_single_roc",
    "dist_top_elements_": "04_single_distributions",
    "pair_auc_matrix_": "05_pair_matrices",
    "pair_top_": "06_pair_top",
    "scatter_best_pair_": "07_pair_scatter",
    "summary_best_single_vs_bin": "08_summary",
}


def _plot_path(name):
    """Map a figure filename to OUT_DIR/<subfolder>/name (created on demand)."""
    for prefix, sub in _PLOT_SUBDIRS.items():
        if name.startswith(prefix):
            folder = os.path.join(OUT_DIR, sub)
            os.makedirs(folder, exist_ok=True)
            return os.path.join(folder, name)
    return os.path.join(OUT_DIR, name)


DATASETS = {
    "4_3": os.path.join(DATA_DIR, "dataset_TakaLimon_realisticNoise_v4_3.npz"),
    "5": os.path.join(DATA_DIR, "dataset_Nomotov5.npz"),
    "L": os.path.join(DATA_DIR, "dataset_Limongi_tpop42.npz"),
    "N": os.path.join(DATA_DIR, "dataset_Nomoto_tpop42.npz"),
}

# Bins: (label, lo, hi) with hi=None meaning +inf for the upper bound and
# lo=None meaning "no mass restriction" for the "all" context bin.
BINS = [
    ("210-230", 210.0, 230.0),
    (">=230", 230.0, None),
    ("all", None, None),
    ("<210", None, 210.0),
]
BIN_ORDER = ["<210", "210-230", ">=230", "all"]

F_STRONG = 0.5
BOOTSTRAP_N = 500
BOOTSTRAP_SEED = 0
N_FOLDS = 5
TOP_PAIRS = 25
TOP_PAIRS_PLOT = 12
TOP_SINGLE_PLOT = 6
DELTA_THRESH = 0.02


# ---------------------------------------------------------------- helpers
def _reconstruct(X, feature_names, el, ref):
    """Column of [el/ref] from the cached pairwise ratios."""
    for j, nm in enumerate(feature_names):
        a, b = nm.strip("[]").split("/")
        if {a, b} == {el, ref}:
            return (1.0 if a == el else -1.0) * X[:, j]
    raise KeyError(f"no {el}/{ref} ratio in feature names")


def build_pool(X, feature_names):
    """Return (elements, pool_names, pool_cols) with FE U MG (deduped)."""
    elems = set()
    for nm in feature_names:
        a, b = nm.strip("[]").split("/")
        elems.add(a)
        elems.add(b)
    elems = sorted(elems)

    def family(ref):
        out = {}
        for el in elems:
            if el == ref:
                continue
            out[f"[{el}/{ref}]"] = _reconstruct(X, feature_names, el, ref)
        return out

    fe = family("Fe")
    mg = family("Mg")

    # pool = FE U MG, dedupe identical columns
    pool_names, pool_cols = [], []
    for name, col in list(fe.items()) + list(mg.items()):
        if any(np.array_equal(col, existing) for existing in pool_cols):
            continue
        pool_names.append(name)
        pool_cols.append(col)
    pool = np.column_stack(pool_cols)

    return elems, fe, mg, pool_names, pool


def positive_mask(y, pm, is_pisn, lo, hi):
    """PISN-dominated samples in the mass bin. lo/hi None = no bound."""
    pos = (y >= F_STRONG) & is_pisn
    if lo is not None:
        pos = pos & (pm >= lo)
    if hi is not None:
        pos = pos & (pm < hi)
    return pos


def bootstrap_auc_ci(pos_vals, neg_vals, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    """95% CI of the signed ROC-AUC via stratified bootstrap."""
    rng = np.random.default_rng(seed)
    pos_vals = np.asarray(pos_vals)
    neg_vals = np.asarray(neg_vals)
    aucs = np.empty(n)
    ybin = np.r_[np.ones(len(pos_vals)), np.zeros(len(neg_vals))]
    for i in range(n):
        p = rng.choice(pos_vals, size=len(pos_vals), replace=True)
        q = rng.choice(neg_vals, size=len(neg_vals), replace=True)
        aucs[i] = roc_auc_score(ybin, np.r_[p, q])
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def single_metrics(pos_vals, neg_vals, n_pos, n_neg):
    """Signed AUC, separability, direction, KS, bootstrap CI."""
    vals = np.r_[pos_vals, neg_vals]
    ybin = np.r_[np.ones(n_pos), np.zeros(n_neg)].astype(int)
    auc = float(roc_auc_score(ybin, vals))
    sep = max(auc, 1.0 - auc)
    direction = "high-in-PISN" if auc >= 0.5 else "low-in-PISN"
    ks = ks_2samp(pos_vals, neg_vals)
    ci = (None, None)
    if n_pos >= 10:
        ci = bootstrap_auc_ci(pos_vals, neg_vals)
    return {
        "auc": auc,
        "separability": sep,
        "direction": direction,
        "ks_stat": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "pos_mean": float(np.mean(pos_vals)),
        "pos_std": float(np.std(pos_vals)),
        "neg_mean": float(np.mean(neg_vals)),
        "neg_std": float(np.std(neg_vals)),
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "ci_low": ci[0],
        "ci_high": ci[1],
    }


def pair_auc(Xstd, binary, cols):
    """5-fold CV logistic AUC for a pair of standardized columns."""
    X2 = Xstd[:, cols]
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    oof = cross_val_predict(clf, X2, binary, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(binary, oof))


# ---------------------------------------------------------------- dataset run
def run_dataset(dataset_id, cache):
    X, y, diag = load_dataset(cache)
    y = np.asarray(y, dtype=float)
    pm = np.asarray(diag["pisn_mass"], dtype=float)
    is_pisn = np.isfinite(pm)
    feature_names = list(diag["feature_names"])

    elems, fe, mg, pool_names, pool = build_pool(X, feature_names)
    pool = SimpleImputer(strategy="median").fit_transform(pool)

    neg = (y == 0.0)
    n_neg = int(neg.sum())

    # single-feature metrics
    single_rows = []
    single_by = {b: {} for b, _, _ in BINS}  # single_by[bin][name] = metrics
    for lo_hi in BINS:
        b, lo, hi = lo_hi
        pos = positive_mask(y, pm, is_pisn, lo, hi)
        n_pos = int(pos.sum())
        if n_pos == 0:
            continue
        for name in pool_names:
            idx = pool_names.index(name)
            col = pool[:, idx]
            m = single_metrics(col[pos], col[neg], n_pos, n_neg)
            m.update({"dataset": dataset_id, "bin": b,
                      "ref": name.strip("[]").split("/")[1],
                      "element": name.strip("[]").split("/")[0],
                      "feature": name})
            single_rows.append(m)
            single_by[b][name] = m

    # pairwise metrics
    pair_rows = []
    pair_matrices = {}
    for lo_hi in BINS:
        b, lo, hi = lo_hi
        pos = positive_mask(y, pm, is_pisn, lo, hi)
        n_pos = int(pos.sum())
        if n_pos == 0:
            continue
        binary = (pos).astype(int)  # only pos|neg samples? NO -> build subset below
        sel = pos | neg
        binary_sel = pos[sel].astype(int)
        Xsub = pool[sel]
        scaler = StandardScaler().fit(Xsub)
        Xstd = scaler.transform(Xsub)

        # single AUCs on the same subset (for max_single + delta)
        single_auc_signed = {}
        single_sep = {}
        for name in pool_names:
            j = pool_names.index(name)
            auc = roc_auc_score(binary_sel, Xsub[:, j])
            single_auc_signed[name] = auc
            single_sep[name] = max(auc, 1 - auc)

        n_pool = len(pool_names)
        mat = np.full((n_pool, n_pool), np.nan)
        pairs = []
        for i in range(n_pool):
            for j in range(i + 1, n_pool):
                pa = pair_auc(Xstd, binary_sel, (i, j))
                ms = max(single_sep[pool_names[i]], single_sep[pool_names[j]])
                delta = pa - ms
                mat[i, j] = mat[j, i] = pa
                pairs.append({
                    "feat_a": pool_names[i], "feat_b": pool_names[j],
                    "pair_auc": pa, "max_single_auc": ms, "delta": delta,
                })
        pairs.sort(key=lambda r: -r["pair_auc"])
        pair_matrices[b] = {"pool_names": pool_names, "matrix": mat.tolist(),
                            "n_pos": n_pos, "n_neg": n_neg}
        for rank, r in enumerate(pairs[:TOP_PAIRS], 1):
            row = dict(r)
            row.update({"dataset": dataset_id, "bin": b, "rank": rank})
            pair_rows.append(row)
        pair_matrices[b]["all_pairs"] = pairs

    print(f"[{dataset_id}] pool={len(pool_names)} features "
          f"({len(fe)} FE + {len(mg)} MG); elements={elems}")

    return {
        "dataset_id": dataset_id,
        "elements": elems,
        "pool_names": pool_names,
        "pool": pool,
        "fe": fe,
        "mg": mg,
        "y": y,
        "pm": pm,
        "is_pisn": is_pisn,
        "neg": neg,
        "single_rows": single_rows,
        "single_by": single_by,
        "pair_rows": pair_rows,
        "pair_matrices": pair_matrices,
    }


# ---------------------------------------------------------------- plots
def _style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        try:
            plt.style.use("ggplot")
        except Exception:
            pass


def plot_heatmap(res):
    ds = res["dataset_id"]
    by = res["single_by"]
    cols = ["210-230", ">=230", "all"]

    def panel(ref, elems_for_ref):
        # rows = elements sorted by >=230 separability desc
        order = sorted(elems_for_ref,
                       key=lambda e: by[">=230"].get(f"[{e}/{ref}]", {}).get("separability", 0.0),
                       reverse=True)
        M = np.empty((len(order), len(cols)))
        for r, e in enumerate(order):
            for c, b in enumerate(cols):
                M[r, c] = by[b].get(f"[{e}/{ref}]", {}).get("separability", np.nan)
        return order, M

    fe_elems = [e for e in res["elements"] if e != "Fe"]
    mg_elems = [e for e in res["elements"] if e != "Mg"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 6), constrained_layout=True)
    for ax, ref, elems, title in [
        (axes[0], "Fe", fe_elems, "reference Fe ([X/Fe])"),
        (axes[1], "Mg", mg_elems, "reference Mg ([X/Mg])"),
    ]:
        order, M = panel(ref, elems)
        im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0.5, vmax=1.0)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        ax.set_title(title)
        for r in range(len(order)):
            for c in range(len(cols)):
                v = M[r, c]
                ax.text(c, r, f"{v:.2f}" if np.isfinite(v) else "nan",
                        ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04, label="separability")
    fig.suptitle(f"test_v9 [{ds}]: single-ratio separability (PISN vs CCSN-only)",
                 fontsize=12)
    fig.savefig(os.path.join(OUT_DIR, f"single_auc_heatmap_{ds}.png"), dpi=130)
    plt.close(fig)


def plot_bars(res, b):
    ds = res["dataset_id"]
    by = res["single_by"][b]
    common = [e for e in res["elements"] if e not in ("Fe", "Mg")]
    order = sorted(common, key=lambda e: max(
        by.get(f"[{e}/Fe]", {}).get("separability", 0.0),
        by.get(f"[{e}/Mg]", {}).get("separability", 0.0)), reverse=True)
    fe_sep = [by[f"[{e}/Fe]"]["separability"] for e in order]
    mg_sep = [by[f"[{e}/Mg]"]["separability"] for e in order]

    def err(e, ref):
        m = by[f"[{e}/{ref}]"]
        auc, lo, hi = m["auc"], m["ci_low"], m["ci_high"]
        if lo is None:
            return None
        if auc >= 0.5:
            sep, elo, ehi = auc, lo, hi
        else:
            sep, elo, ehi = 1 - auc, 1 - hi, 1 - lo
        return [sep - elo, ehi - sep]

    x = np.arange(len(order))
    w = 0.38
    fe_err = np.array([err(e, "Fe") for e in order], dtype=object)
    mg_err = np.array([err(e, "Mg") for e in order], dtype=object)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, fe_sep, w, yerr=_transpose_err(fe_err),
           label="[X/Fe]", color="tab:blue", capsize=3)
    ax.bar(x + w / 2, mg_sep, w, yerr=_transpose_err(mg_err),
           label="[X/Mg]", color="tab:orange", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_ylabel("separability (max(AUC, 1-AUC))")
    ax.set_ylim(0.5, 1.0)
    ax.legend()
    ax.set_title(f"test_v9 [{ds}] bin={b}: single-ratio separability Fe vs Mg")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"single_auc_bars_{ds}_{b}.png"), dpi=130)
    plt.close(fig)


def _transpose_err(arr):
    """list of [lo, hi] -> ([lo...], [hi...]); None-safe."""
    lo, hi = [], []
    for v in arr:
        if v is None:
            lo.append(0.0)
            hi.append(0.0)
        else:
            lo.append(v[0])
            hi.append(v[1])
    return lo, hi


def _top_single(res, b, k=TOP_SINGLE_PLOT):
    by = res["single_by"][b]
    names = sorted(by, key=lambda n: -by[n]["separability"])
    return names[:k]


def plot_roc(res, b):
    ds = res["dataset_id"]
    by = res["single_by"][b]
    y = res["y"]
    pm = res["pm"]
    is_pisn = res["is_pisn"]
    neg = res["neg"]
    lo = dict((lbl, lo) for lbl, lo, _ in BINS)[b]
    hi = dict((lbl, hi) for lbl, _, hi in BINS)[b]
    pos = positive_mask(y, pm, is_pisn, lo, hi)
    ybin = pos.astype(int)

    top = _top_single(res, b)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    for name in top:
        idx = res["pool_names"].index(name)
        col = res["pool"][:, idx]
        # orient so higher = more PISN -> AUC == separability
        m = by[name]
        sign = 1.0 if m["auc"] >= 0.5 else -1.0
        score = sign * col
        fpr, tpr, _ = roc_curve(ybin, score)
        ax.plot(fpr, tpr, lw=1.6, label=f"{name}  AUC={m['separability']:.2f}")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"test_v9 [{ds}] bin={b}: top single-feature ROC")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"roc_curves_top_{ds}_{b}.png"), dpi=130)
    plt.close(fig)


def plot_dist(res, b):
    ds = res["dataset_id"]
    by = res["single_by"][b]
    y = res["y"]
    pm = res["pm"]
    is_pisn = res["is_pisn"]
    neg = res["neg"]
    lo = dict((lbl, lo) for lbl, lo, _ in BINS)[b]
    hi = dict((lbl, hi) for lbl, _, hi in BINS)[b]
    pos = positive_mask(y, pm, is_pisn, lo, hi)

    top = _top_single(res, b)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, name in zip(axes.flat, top):
        idx = res["pool_names"].index(name)
        col = res["pool"][:, idx]
        pv = col[pos]
        nv = col[neg]
        ax.hist(nv, bins=35, alpha=0.6, color="tab:gray", label="CCSN-only", density=True)
        ax.hist(pv, bins=35, alpha=0.6, color="tab:red", label="PISN", density=True)
        ax.set_title(f"{name}  sep={by[name]['separability']:.2f}")
        ax.tick_params(labelsize=7)
    handles = [Line2D([0], [0], color="tab:gray", lw=4, label="CCSN-only"),
               Line2D([0], [0], color="tab:red", lw=4, label="PISN")]
    fig.legend(handles=handles, loc="upper right")
    fig.suptitle(f"test_v9 [{ds}] bin={b}: top single-feature distributions", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT_DIR, f"dist_top_elements_{ds}_{b}.png"), dpi=130)
    plt.close(fig)


def plot_pair_matrix(res, b):
    ds = res["dataset_id"]
    pm = res["pair_matrices"][b]
    names = pm["pool_names"]
    M = np.array(pm["matrix"])
    mask = np.eye(len(names), dtype=bool)
    Mdisp = np.ma.masked_where(mask, M)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(Mdisp, cmap="magma", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j and np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=5.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="pair AUC")
    ax.set_title(f"test_v9 [{ds}] bin={b}: pairwise logistic AUC (FE U MG pool)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"pair_auc_matrix_{ds}_{b}.png"), dpi=130)
    plt.close(fig)


def plot_pair_top(res, b):
    ds = res["dataset_id"]
    pairs = [r for r in res["pair_rows"] if r["bin"] == b][:TOP_PAIRS_PLOT]
    names = [f"{r['feat_a']} & {r['feat_b']}" for r in pairs]
    aucs = [r["pair_auc"] for r in pairs]
    deltas = [r["delta"] for r in pairs]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["tab:red" if d > DELTA_THRESH else "tab:blue" for d in deltas]
    ax.bar(np.arange(len(pairs)), aucs, color=colors)
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xticks(np.arange(len(pairs)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("pair AUC")
    ax.set_ylim(0.5, 1.0)
    # legend for delta>0.02 hatch
    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(color="tab:red", label=f"delta > {DELTA_THRESH}"),
        mpatches.Patch(color="tab:blue", label=f"delta <= {DELTA_THRESH}"),
    ], fontsize=8)
    ax.set_title(f"test_v9 [{ds}] bin={b}: top pairs (red = beats best single)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"pair_top_{ds}_{b}.png"), dpi=130)
    plt.close(fig)


def plot_scatter_best_pair(res, b):
    ds = res["dataset_id"]
    pm = res["pair_matrices"][b]
    pairs = [r for r in res["pair_rows"] if r["bin"] == b]
    if not pairs:
        return
    best = pairs[0]
    ia = res["pool_names"].index(best["feat_a"])
    ib = res["pool_names"].index(best["feat_b"])

    y = res["y"]
    pmass = res["pm"]
    is_pisn = res["is_pisn"]
    neg = res["neg"]
    lo = dict((lbl, lo) for lbl, lo, _ in BINS)[b]
    hi = dict((lbl, hi) for lbl, _, hi in BINS)[b]
    pos = positive_mask(y, pmass, is_pisn, lo, hi)
    sel = pos | neg

    Xsub = res["pool"][sel]
    scaler = StandardScaler().fit(Xsub[:, [ia, ib]])
    Xstd = scaler.transform(Xsub[:, [ia, ib]])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
        Xstd, pos[sel].astype(int))

    a = Xsub[:, 0]
    bb = Xsub[:, 1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(a[~pos[sel]], bb[~pos[sel]], s=8, c="tab:gray", alpha=0.5, label="CCSN-only")
    ax.scatter(a[pos[sel]], bb[pos[sel]], s=8, c="tab:red", alpha=0.7, label="PISN")

    # decision boundary (standardized) -> raw coordinates
    w0, w1 = clf.coef_[0]
    b0 = clf.intercept_[0]
    # std2 = -(w0*std1 + b0)/w1 ; raw = std*scale + mean
    mean0, mean1 = scaler.mean_
    s0, s1 = scaler.scale_
    xmin, xmax = a.min(), a.max()
    std1_lo = (xmin - mean0) / s0
    std1_hi = (xmax - mean0) / s0
    std2_lo = -(w0 * std1_lo + b0) / w1
    std2_hi = -(w0 * std1_hi + b0) / w1
    y_lo = std2_lo * s1 + mean1
    y_hi = std2_hi * s1 + mean1
    ax.plot([xmin, xmax], [y_lo, y_hi], "k--", lw=1.5, label="logistic boundary")
    ax.set_xlabel(best["feat_a"])
    ax.set_ylabel(best["feat_b"])
    ax.legend(fontsize=8)
    ax.set_title(f"test_v9 [{ds}] bin={b}: best pair "
                 f"{best['feat_a']} & {best['feat_b']} (AUC={best['pair_auc']:.3f})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"scatter_best_pair_{ds}_{b}.png"), dpi=130)
    plt.close(fig)


def plot_summary(all_res):
    fig, ax = plt.subplots(figsize=(7, 5))
    for ds, res in all_res.items():
        xs = []
        ys = []
        for b in BIN_ORDER:
            if b not in res["single_by"]:
                continue
            by = res["single_by"][b]
            if not by:
                continue
            best = max(by[n]["separability"] for n in by)
            xs.append(BIN_ORDER.index(b))
            ys.append(best)
        ax.plot(xs, ys, "o-", lw=1.8, label=ds)
    ax.set_xticks(range(len(BIN_ORDER)))
    ax.set_xticklabels(BIN_ORDER)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("best single-feature separability")
    ax.set_xlabel("PISN mass bin")
    ax.legend()
    ax.set_title("test_v9: best single-ratio separability vs mass bin")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "summary_best_single_vs_bin.png"), dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- report
def write_report(all_res, single_rows, pair_rows):
    lines = [
        "PISN vs CCSN-only abundance-ratio signature analysis (test_v9)",
        "Positive = PISN-dominated (f_pisn >= 0.5) in mass bin; "
        "Negative = CCSN-only (f_pisn == 0), always included.",
        "Single separability = max(signed AUC, 1 - AUC); "
        "direction 'high-in-PISN' means positives have larger values.",
        "Pair AUC = 5-fold CV logistic (balanced) over FE U Mg pool; "
        "delta = pair_auc - max single separability of the two members.",
        "",
    ]
    for ds in DATASETS:
        res = all_res[ds]
        for b, lo, hi in BINS:
            if b not in res["single_by"]:
                continue
            by = res["single_by"][b]
            names = sorted(by, key=lambda n: -by[n]["separability"])
            prs = [r for r in pair_rows if r["dataset"] == ds and r["bin"] == b]
            lines.append(f"== {ds} / bin {b} (n_pos={by[names[0]]['n_pos']}, "
                         f"n_neg={by[names[0]]['n_neg']}) ==")
            lines.append("  top-5 single features:")
            for n in names[:5]:
                m = by[n]
                lines.append(f"    {n:<10} sep={m['separability']:.3f} "
                             f"{m['direction']:<13} KS_p={m['ks_p']:.2e}")
            lines.append(f"  top-10 pairs:")
            for r in prs[:10]:
                lines.append(f"    {r['feat_a']:<8} & {r['feat_b']:<8} "
                             f"pair_auc={r['pair_auc']:.3f} delta={r['delta']:+.3f}")
            best_single = by[names[0]]["separability"]
            max_delta = max((r["delta"] for r in prs), default=0.0)
            beats = max_delta > DELTA_THRESH
            if beats:
                best_d = max(prs, key=lambda r: r["delta"])
                lines.append(f"  PAIR BEATS ITS OWN BEST SINGLE by >{DELTA_THRESH}: YES "
                             f"(max delta={max_delta:+.3f} from "
                             f"{best_d['feat_a']} & {best_d['feat_b']})")
            else:
                lines.append(f"  PAIR BEATS ITS OWN BEST SINGLE by >{DELTA_THRESH}: NO "
                             f"(max delta={max_delta:+.3f})")
            lines.append(f"  (global best single sep={best_single:.3f}; "
                         f"a pair can only exceed a single when that single < 1.000)")
            lines.append("")
    with open(os.path.join(OUT_DIR, "signature_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------- main
def main():
    _style()
    all_res = {}
    all_single_rows = []
    all_pair_rows = []

    for ds, cache in DATASETS.items():
        res = run_dataset(ds, cache)
        all_res[ds] = res
        all_single_rows.extend(res["single_rows"])
        all_pair_rows.extend(res["pair_rows"])

    # --- CSVs ---
    with open(os.path.join(OUT_DIR, "single_feature_auc.csv"), "w", newline="") as f:
        fieldnames = ["dataset", "bin", "ref", "element", "feature", "auc",
                      "separability", "direction", "ks_stat", "ks_p",
                      "pos_mean", "pos_std", "neg_mean", "neg_std",
                      "n_pos", "n_neg", "ci_low", "ci_high"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_single_rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    with open(os.path.join(OUT_DIR, "pairwise_top_auc.csv"), "w", newline="") as f:
        fieldnames = ["dataset", "bin", "rank", "feat_a", "feat_b",
                      "pair_auc", "max_single_auc", "delta"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_pair_rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    # --- JSONs ---
    with open(os.path.join(OUT_DIR, "single_feature_auc.json"), "w") as f:
        json.dump(all_single_rows, f, indent=2)

    pair_json = {ds: {b: {"pool_names": pm["pool_names"], "matrix": pm["matrix"],
                          "n_pos": pm["n_pos"], "n_neg": pm["n_neg"],
                          "all_pairs": pm["all_pairs"]}
                      for b, pm in res["pair_matrices"].items()}
                 for ds, res in all_res.items()}
    with open(os.path.join(OUT_DIR, "pairwise_matrices.json"), "w") as f:
        json.dump(pair_json, f, indent=2)

    # --- plots ---
    for ds, res in all_res.items():
        plot_heatmap(res)
        for b, lo, hi in BINS:
            if b not in res["single_by"]:
                continue
            plot_bars(res, b)
            plot_roc(res, b)
            plot_dist(res, b)
            plot_pair_matrix(res, b)
            plot_pair_top(res, b)
            plot_scatter_best_pair(res, b)
    plot_summary(all_res)

    write_report(all_res, all_single_rows, all_pair_rows)

    # --- summary of produced files ---
    files = sorted(os.listdir(OUT_DIR))
    print("\nProduced files (size bytes):")
    for fn in files:
        p = os.path.join(OUT_DIR, fn)
        print(f"  {fn:<40} {os.path.getsize(p):>8}")
    print(f"\nTotal: {len(files)} files in {OUT_DIR}")


if __name__ == "__main__":
    main()
