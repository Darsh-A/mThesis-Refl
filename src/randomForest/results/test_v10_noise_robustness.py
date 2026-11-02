"""Observational-noise robustness of the PISN-vs-CCSN signature (test_v10).

test_v9 measured ground-truth single-feature and pairwise separability on
*noiseless* forward-model datasets (dex_error = 0.0), where many abundance
ratios achieved perfect (AUC = 1.0) separation.  This follow-up makes the
signatures observationally meaningful by injecting Gaussian dex noise into
every cell of X and re-measuring which signatures SURVIVE.

Class definition and bins are IDENTICAL to test_v9:
  positives = (f_pisn >= 0.5) & finite(pisn_mass) & mass in bin
  negatives = (f_pisn == 0)
  bins: 210-230, >=230, all, <210
  feature families FE = {[X/Fe]}, MG = {[X/Mg]} (reconstructed as in v9).

Noise injection (per dataset 4_3 / 5 / L / N):
  sigma in {0.05, 0.10, 0.20} dex, R = 10 realizations,
  seed = 1000 * round(sigma * 100) + r.  sigma = 0.0 is the noiseless baseline
  (R = 1).  taka_0.1 (dataset_takahashi_noise.npz, dex_error = 0.1) is run once
  as a real noisy reference.

Metrics:
  * singles: signed AUC -> separability, direction, KS test; aggregated
    mean +/- std over realizations -> noise_single_metrics.csv
  * pairs (sigma = 0.10, R = 5, + sigma = 0.0 baseline): 5-fold CV logistic
    (balanced) pair AUC + delta vs best single -> noise_pair_metrics.csv
  * survival table -> signature_survival.csv (sep @ 0/0.05/0.10/0.20)

Reuses helpers (build_pool, positive_mask, pair_auc, BINS, F_STRONG, DATA_DIR)
from test_v9 via importlib; test_v9 is NOT modified.

Outputs (src/randomForest/results/test_v10/):
  noise_single_metrics.csv, noise_pair_metrics.csv, signature_survival.csv,
  degradation_<ds>_<bin>.png, degradation_all_bins.png,
  survival_heatmap_<ds>_<bin>.png, noise_pair_top_<ds>_<bin>.png,
  taka_vs_injected_<bin>.png, summary_survivors.png,
  noise_robustness_report.txt
"""

import csv
import importlib.util
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
from joblib import Parallel, delayed


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

# --- import test_v9 helpers without modifying it ---
V9_PATH = os.path.join(ROOT, "src", "randomForest", "results",
                       "test_v9_signature_separability.py")
_spec = importlib.util.spec_from_file_location("v9sig", V9_PATH)
v9 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v9)

OUT_DIR = os.path.join(ROOT, "src", "randomForest", "results", "test_v10")
os.makedirs(OUT_DIR, exist_ok=True)

_PLOT_SUBDIRS = {
    "degradation_": "01_degradation",
    "survival_heatmap_": "02_survival_heatmaps",
    "noise_pair_top_": "03_noise_pair_top",
    "taka_vs_injected_": "04_taka_vs_injected",
    "summary_survivors": "05_summary",
}


def _plot_path(name):
    """Map a figure filename to OUT_DIR/<subfolder>/name (created on demand)."""
    for prefix, sub in _PLOT_SUBDIRS.items():
        if name.startswith(prefix):
            folder = os.path.join(OUT_DIR, sub)
            os.makedirs(folder, exist_ok=True)
            return os.path.join(folder, name)
    return os.path.join(OUT_DIR, name)


DATA_DIR = v9.DATA_DIR
F_STRONG = v9.F_STRONG
BINS = list(v9.BINS)  # [(label, lo, hi), ...]

DATASETS = {
    "4_3": os.path.join(DATA_DIR, "dataset_TakaLimon_realisticNoise_v4_3.npz"),
    "5": os.path.join(DATA_DIR, "dataset_Nomotov5.npz"),
    "L": os.path.join(DATA_DIR, "dataset_Limongi_tpop42.npz"),
    "N": os.path.join(DATA_DIR, "dataset_Nomoto_tpop42.npz"),
}
TAKA_PATH = os.path.join(DATA_DIR, "dataset_takahashi_noise.npz")

SIGMAS = [0.0, 0.05, 0.10, 0.20]
R_PER_SIGMA = 10
PAIR_SIGMAS = (0.0, 0.10)
PAIR_R = 5
TOP_N_PLOT = 8
TOP_PAIRS_PLOT = 12
DELTA_THRESH = 0.02
SURVIVE_SEP = 0.90
SURVIVE_P = 0.05
NOISE_FLOOR = 0.10
N_JOBS = -1
N_CHUNKS = 16


def seed_for(sigma, r):
    return 1000 * int(round(sigma * 100)) + r


def build_mapping(feature_names, pool_names):
    """[(col_index, sign), ...] so pool[:, k] = sign * X[:, col_index]."""
    mapping = []
    for nm in pool_names:
        el, ref = nm.strip("[]").split("/")
        for j, full in enumerate(feature_names):
            a, b = full.strip("[]").split("/")
            if {a, b} == {el, ref}:
                mapping.append((j, 1.0 if a == el else -1.0))
                break
        else:
            raise KeyError(f"no ratio for {nm}")
    return mapping


def reconstruct(X, mapping):
    return np.column_stack([s * X[:, j] for (j, s) in mapping])


def light_single(pos_vals, neg_vals):
    n_pos = len(pos_vals)
    ybin = np.r_[np.ones(n_pos), np.zeros(len(neg_vals))].astype(int)
    vals = np.r_[pos_vals, neg_vals]
    auc = float(roc_auc_score(ybin, vals))
    sep = max(auc, 1.0 - auc)
    ks = ks_2samp(pos_vals, neg_vals)
    return {"auc": auc, "sep": sep, "ks_stat": float(ks.statistic),
            "ks_p": float(ks.pvalue)}


# Local replica of v9.pair_auc (identical logic) so it is picklable by joblib
# (v9 is loaded via importlib under a non-importable module name).
def _pair_auc_local(Xstd, binary, i, j):
    X2 = Xstd[:, [i, j]]
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = cross_val_predict(clf, X2, binary, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(binary, oof))


def _pair_chunk(chunk, Xstd, binary):
    return [(i, j, _pair_auc_local(Xstd, binary, i, j)) for (i, j) in chunk]


# ---------------------------------------------------------------- data load
def load_all():
    data = {}
    for ds, path in DATASETS.items():
        X, y, diag = load_dataset(path)
        data[ds] = _prep(X, y, diag)
    Xt, yt, dt = load_dataset(TAKA_PATH)
    data["taka_0.1"] = _prep(Xt, yt, dt)
    return data


def _prep(X, y, diag):
    y = np.asarray(y, dtype=float)
    pm = np.asarray(diag["pisn_mass"], dtype=float)
    is_pisn = np.isfinite(pm)
    feature_names = list(diag["feature_names"])
    elems, fe, mg, pool_names, pool = v9.build_pool(X, feature_names)
    mapping = build_mapping(feature_names, pool_names)
    neg = (y == 0.0)
    return {
        "X": X, "y": y, "pm": pm, "is_pisn": is_pisn,
        "feature_names": feature_names, "pool_names": pool_names,
        "mapping": mapping, "neg": neg, "elems": elems,
        "pisn_source": str(diag["pisn_source"]),
        "sn_source": str(diag["sn_source"]),
        "dex_error": float(diag["dex_error"]),
    }


def _std(xs):
    return float(np.std(xs)) if len(xs) else float("nan")


# ---------------------------------------------------------------- core compute
def run():
    data = load_all()
    # real single metrics per (ds, sigma, r, bin, feature)
    real_single = {}   # real_single[ds][sigma][r][bin][feature] = dict
    # pair metrics per (ds, sigma, r, bin) -> list of (i, j, pair_auc, max_single, delta)
    real_pair = {}     # real_pair[ds][sigma][r][bin] = list[...]

    t0 = time.time()
    for ds in ["4_3", "5", "L", "N"]:
        d = data[ds]
        X = d["X"]
        y, pm, is_pisn = d["y"], d["pm"], d["is_pisn"]
        neg = d["neg"]
        pool_names = d["pool_names"]
        mapping = d["mapping"]
        n_pool = len(pool_names)
        pair_idx = [(i, j) for i in range(n_pool) for j in range(i + 1, n_pool)]
        print(f"[{ds}] start (n_features={n_pool}, n_pairs={len(pair_idx)})", flush=True)

        # ---- singles (sequential; R=10 for sigma>0, R=1 for sigma=0) ----
        for sigma in SIGMAS:
            r_list = [0] if sigma == 0.0 else list(range(R_PER_SIGMA))
            for r in r_list:
                if sigma == 0.0:
                    X_noisy = X
                else:
                    X_noisy = X + np.random.default_rng(seed_for(sigma, r)).normal(
                        0.0, sigma, size=X.shape)
                pool = reconstruct(X_noisy, mapping)
                real_single.setdefault(ds, {}).setdefault(sigma, {})[r] = {}
                for b, lo, hi in BINS:
                    pos = v9.positive_mask(y, pm, is_pisn, lo, hi)
                    real_single[ds][sigma][r][b] = {}
                    for k, name in enumerate(pool_names):
                        real_single[ds][sigma][r][b][name] = light_single(
                            pool[pos, k], pool[neg, k])

        # ---- pairs (parallel; R=5 at sigma=0.10, R=1 at sigma=0.0) ----
        for sigma in PAIR_SIGMAS:
            r_list = [0] if sigma == 0.0 else list(range(PAIR_R))
            for r in r_list:
                if sigma == 0.0:
                    X_noisy = X
                else:
                    X_noisy = X + np.random.default_rng(seed_for(sigma, r)).normal(
                        0.0, sigma, size=X.shape)
                pool = reconstruct(X_noisy, mapping)
                real_pair.setdefault(ds, {}).setdefault(sigma, {})[r] = {}
                for b, lo, hi in BINS:
                    pos = v9.positive_mask(y, pm, is_pisn, lo, hi)
                    sel = pos | neg
                    binary = pos[sel].astype(int)
                    Xstd = StandardScaler().fit_transform(pool[sel])
                    chunks = [pair_idx[k::N_CHUNKS] for k in range(N_CHUNKS)]
                    res = Parallel(n_jobs=N_JOBS)(
                        delayed(_pair_chunk)(c, Xstd, binary) for c in chunks)
                    flat = [x for sub in res for x in sub]
                    sep_here = {name: real_single[ds][sigma][r][b][name]["sep"]
                                for name in pool_names}
                    pairs = []
                    for (i, j, pa) in flat:
                        ms = max(sep_here[pool_names[i]], sep_here[pool_names[j]])
                        pairs.append((i, j, pa, ms, pa - ms))
                    real_pair[ds][sigma][r][b] = pairs
        print(f"[{ds}] done in {time.time()-t0:.1f}s", flush=True)

    # --- taka_0.1: run once (intrinsic 0.1 dex) ---
    d = data["taka_0.1"]
    X = d["X"]
    y, pm, is_pisn = d["y"], d["pm"], d["is_pisn"]
    neg = d["neg"]
    pool_names = d["pool_names"]
    mapping = d["mapping"]
    pool = reconstruct(X, mapping)
    ds = "taka_0.1"
    sigma = 0.1
    r = 0
    real_single.setdefault(ds, {}).setdefault(sigma, {})[r] = {}
    real_pair.setdefault(ds, {}).setdefault(sigma, {})[r] = {}
    for b, lo, hi in BINS:
        pos = v9.positive_mask(y, pm, is_pisn, lo, hi)
        real_single[ds][sigma][r][b] = {}
        for k, name in enumerate(pool_names):
            real_single[ds][sigma][r][b][name] = light_single(pool[pos, k], pool[neg, k])
        sel = pos | neg
        binary = pos[sel].astype(int)
        Xstd = StandardScaler().fit_transform(pool[sel])
        sep_here = {name: real_single[ds][sigma][r][b][name]["sep"] for name in pool_names}
        pairs = []
        for i in range(len(pool_names)):
            for j in range(i + 1, len(pool_names)):
                pa = _pair_auc_local(Xstd, binary, i, j)
                ms = max(sep_here[pool_names[i]], sep_here[pool_names[j]])
                pairs.append((i, j, pa, ms, pa - ms))
        real_pair[ds][sigma][r][b] = pairs

    return data, real_single, real_pair


# ---------------------------------------------------------------- aggregation
def aggregate(data, real_single, real_pair):
    single_rows = []
    agg_single = {}  # agg_single[ds][sigma][bin][feature] = dict (mean/std)
    for ds, by_sigma in real_single.items():
        agg_single.setdefault(ds, {})
        for sigma, by_r in by_sigma.items():
            agg_single[ds].setdefault(sigma, {})
            for r, by_bin in by_r.items():
                for b, feats in by_bin.items():
                    agg_single[ds][sigma].setdefault(b, {})
                    for name, m in feats.items():
                        agg_single[ds][sigma][b].setdefault(name, []).append(m)
            for b, feats in agg_single[ds][sigma].items():
                for name, ms in feats.items():
                    seps = [m["sep"] for m in ms]
                    aucs = [m["auc"] for m in ms]
                    ks_stat = [m["ks_stat"] for m in ms]
                    ks_p = [m["ks_p"] for m in ms]
                    a_mean = float(np.mean(aucs))
                    agg = {
                        "dataset": ds, "sigma": sigma, "R": len(ms),
                        "bin": b, "feature": name,
                        "ref": name.strip("[]").split("/")[1],
                        "element": name.strip("[]").split("/")[0],
                        "sep_mean": float(np.mean(seps)),
                        "sep_std": _std(seps),
                        "auc_mean": a_mean,
                        "auc_std": _std(aucs),
                        "direction": "high-in-PISN" if a_mean >= 0.5 else "low-in-PISN",
                        "ks_stat_mean": float(np.mean(ks_stat)),
                        "ks_p_mean": float(np.mean(ks_p)),
                        "n_pos": None, "n_neg": None,
                    }
                    agg_single[ds][sigma][b][name] = agg
                    single_rows.append(agg)

    # n_pos/n_neg (same for all features in a bin; from the data)
    for ds in DATASETS:
        d = data[ds]
        y, pm, is_pisn = d["y"], d["pm"], d["is_pisn"]
        for b, lo, hi in BINS:
            n_pos = int(v9.positive_mask(y, pm, is_pisn, lo, hi).sum())
            n_neg = int(d["neg"].sum())
            for sigma in agg_single[ds]:
                if b in agg_single[ds][sigma]:
                    for name in agg_single[ds][sigma][b]:
                        agg_single[ds][sigma][b][name]["n_pos"] = n_pos
                        agg_single[ds][sigma][b][name]["n_neg"] = n_neg

    # --- pairs aggregation ---
    pair_rows = []
    agg_pair = {}  # agg_pair[ds][sigma][bin] = list of dicts (sorted)
    for ds, by_sigma in real_pair.items():
        agg_pair.setdefault(ds, {})
        for sigma, by_r in by_sigma.items():
            agg_pair[ds].setdefault(sigma, {})
            for b, lo, hi in BINS:
                # gather per-pair lists keyed by (i,j)
                acc = {}
                for r, by_bin in by_r.items():
                    if b not in by_bin:
                        continue
                    for (i, j, pa, ms, delta) in by_bin[b]:
                        acc.setdefault((i, j), []).append((pa, ms, delta))
                out = []
                for (i, j), vals in acc.items():
                    pa_mean = float(np.mean([v[0] for v in vals]))
                    pa_std = _std([v[0] for v in vals])
                    ms_mean = float(np.mean([v[1] for v in vals]))
                    delta_mean = float(np.mean([v[2] for v in vals]))
                    delta_std = _std([v[2] for v in vals])
                    out.append({
                        "dataset": ds, "sigma": sigma, "R": len(vals), "bin": b,
                        "feat_a": data[ds]["pool_names"][i],
                        "feat_b": data[ds]["pool_names"][j],
                        "pair_auc_mean": pa_mean, "pair_auc_std": pa_std,
                        "max_single_auc_mean": ms_mean,
                        "delta_mean": delta_mean, "delta_std": delta_std,
                    })
                out.sort(key=lambda x: -x["pair_auc_mean"])
                for rank, row in enumerate(out[:25], 1):
                    r2 = dict(row); r2["rank"] = rank
                    pair_rows.append(r2)
                agg_pair[ds][sigma][b] = out

    return single_rows, agg_single, pair_rows, agg_pair


# ---------------------------------------------------------------- survival table
def survival_table(agg_single):
    rows = []
    for ds in ["4_3", "5", "L", "N"]:
        for b, lo, hi in BINS:
            s0 = agg_single[ds][0.0][b]
            for name in s0:
                sep0 = s0[name]["sep_mean"]
                sep05 = agg_single[ds][0.05][b][name]["sep_mean"]
                sep10 = agg_single[ds][0.10][b][name]["sep_mean"]
                sep20 = agg_single[ds][0.20][b][name]["sep_mean"]
                ks_p10 = agg_single[ds][0.10][b][name]["ks_p_mean"]
                survives = (sep10 >= SURVIVE_SEP) and (ks_p10 < SURVIVE_P)
                rows.append({
                    "dataset": ds, "bin": b, "feature": name,
                    "ref": name.strip("[]").split("/")[1],
                    "element": name.strip("[]").split("/")[0],
                    "sep_0": sep0, "sep_0p05": sep05, "sep_0p10": sep10,
                    "sep_0p20": sep20, "ks_p_0p10": ks_p10,
                    "survives_0p1": bool(survives),
                })
    return rows


# ---------------------------------------------------------------- plots
def _style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        try:
            plt.style.use("ggplot")
        except Exception:
            pass


def top_noiseless(agg_single, ds, b, k=TOP_N_PLOT):
    s0 = agg_single[ds][0.0][b]
    names = sorted(s0, key=lambda n: (-s0[n]["sep_mean"], n))
    return names[:k]


def plot_degradation(agg_single, ds, b):
    top = top_noiseless(agg_single, ds, b)
    sigmas = [s for s in SIGMAS if s in agg_single[ds]]
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in top:
        sep_mean = [agg_single[ds][s][b][name]["sep_mean"] for s in sigmas]
        sep_std = [agg_single[ds][s][b][name]["sep_std"] for s in sigmas]
        ax.errorbar(sigmas, sep_mean, yerr=sep_std, fmt="o-", lw=1.5, ms=4,
                    label=name, capsize=2)
    ax.axvline(NOISE_FLOOR, color="k", ls="--", lw=1, label=f"{NOISE_FLOOR} dex floor")
    ax.set_xlabel("injected noise sigma (dex)")
    ax.set_ylabel("separability (max AUC, 1-AUC)")
    ax.set_ylim(0.4, 1.01)
    ax.set_title(f"test_v10 [{ds}] bin={b}: top-{TOP_N_PLOT} noiseless features degrade")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"degradation_{ds}_{b}.png"), dpi=130)
    plt.close(fig)


def plot_degradation_all(agg_single):
    sigmas = [s for s in SIGMAS if 0.0 in agg_single["4_3"]]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, ds in zip(axes.flat, ["4_3", "5", "L", "N"]):
        for b, lo, hi in BINS:
            best = [max(agg_single[ds][s][b][n]["sep_mean"] for n in agg_single[ds][s][b])
                    for s in sigmas]
            ax.plot(sigmas, best, "o-", lw=1.6, label=b)
        ax.set_xlabel("noise sigma (dex)")
        ax.set_ylabel("best single sep")
        ax.set_ylim(0.4, 1.01)
        ax.set_title(ds)
        ax.legend(fontsize=8)
    fig.suptitle("test_v10: best single-feature separability vs noise (per bin)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT_DIR, "degradation_all_bins.png"), dpi=130)
    plt.close(fig)


def plot_survival_heatmap(agg_single, ds, b):
    sigmas = [0.0, 0.05, 0.10, 0.20]
    s0 = agg_single[ds][0.0][b]
    order = sorted(s0, key=lambda n: (-s0[n]["sep_mean"], n))
    M = np.empty((len(order), len(sigmas)))
    for r, name in enumerate(order):
        for c, s in enumerate(sigmas):
            M[r, c] = agg_single[ds][s][b][name]["sep_mean"]
    fig, ax = plt.subplots(figsize=(6, max(4, len(order) * 0.42)))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(sigmas)))
    ax.set_xticklabels([f"{s:.2f}" for s in sigmas])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("noise sigma (dex)")
    for r in range(len(order)):
        for c in range(len(sigmas)):
            ax.text(c, r, f"{M[r,c]:.2f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean separability")
    ax.set_title(f"test_v10 [{ds}] bin={b}: separability vs noise")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"survival_heatmap_{ds}_{b}.png"), dpi=130)
    plt.close(fig)


def plot_pair_top(agg_pair, ds, b):
    pairs = [p for p in agg_pair[ds][0.10][b]][:TOP_PAIRS_PLOT]
    names = [f"{p['feat_a']} & {p['feat_b']}" for p in pairs]
    aucs = [p["pair_auc_mean"] for p in pairs]
    deltas = [p["delta_mean"] for p in pairs]
    errs = [p["pair_auc_std"] for p in pairs]
    colors = ["tab:red" if d > DELTA_THRESH else "tab:blue" for d in deltas]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(np.arange(len(pairs)), aucs, yerr=errs, color=colors, capsize=3)
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xticks(np.arange(len(pairs)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("pair AUC (0.10 dex)")
    ax.set_ylim(0.5, 1.0)
    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(color="tab:red", label=f"delta > {DELTA_THRESH}"),
        mpatches.Patch(color="tab:blue", label=f"delta <= {DELTA_THRESH}"),
    ], fontsize=8)
    ax.set_title(f"test_v10 [{ds}] bin={b}: top pairs @ 0.10 dex")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"noise_pair_top_{ds}_{b}.png"), dpi=130)
    plt.close(fig)


def plot_taka_vs_injected(agg_single, b):
    # 91-feat set: taka_0.1 sep vs 4_3 injected 0.10 sep (mean)
    injected = {n: agg_single["4_3"][0.10][b][n]["sep_mean"]
                for n in agg_single["4_3"][0.10][b]}
    taka = {n: agg_single["taka_0.1"][0.1][b][n]["sep_mean"]
            for n in agg_single["taka_0.1"][0.1][b]}
    names = sorted(set(injected) & set(taka))
    x = [injected[n] for n in names]
    yv = [taka[n] for n in names]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, yv, s=22, color="tab:blue", alpha=0.8)
    ax.plot([0.5, 1.0], [0.5, 1.0], "k--", lw=1, label="y = x")
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("4_3 injected 0.10 dex separability")
    ax.set_ylabel("taka_0.1 separability")
    ax.set_title(f"test_v10 bin={b}: real 0.1-dex dataset vs injected (91-feat set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"taka_vs_injected_{b}.png"), dpi=130)
    plt.close(fig)


def plot_summary_survivors(agg_single, surv_rows):
    ds_list = ["4_3", "5", "L", "N"]
    bins = [b for b, _, _ in BINS]
    rows = [f"{ds}/{b}" for ds in ds_list for b in bins]
    n_rank = TOP_N_PLOT
    grid = np.zeros((len(rows), n_rank), dtype=bool)
    labels = np.empty((len(rows), n_rank), dtype=object)
    surv_by = {}
    for r in surv_rows:
        surv_by[(r["dataset"], r["bin"], r["feature"])] = r["survives_0p1"]
    for ri, (ds, b) in enumerate([(d, bb) for d in ds_list for bb in bins]):
        top = top_noiseless(agg_single, ds, b)
        for c, name in enumerate(top):
            grid[ri, c] = surv_by.get((ds, b, name), False)
            labels[ri, c] = name.strip("[]").replace("/", "/")
    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.get_cmap("RdYlGn")
    ax.imshow(grid.astype(float), cmap=cmap, aspect="auto", vmin=0, vmax=1)
    for ri in range(len(rows)):
        for c in range(n_rank):
            ax.text(c, ri, labels[ri, c], ha="center", va="center", fontsize=8,
                    color="black")
    ax.set_xticks(range(n_rank))
    ax.set_xticklabels([f"rank {i+1}" for i in range(n_rank)])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_title("test_v10: which top-8 noiseless features survive 0.1 dex "
                 "(green=survives, red=collapses)")
    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(color=cmap(1.0), label="survives (sep>=0.90, KS p<0.05)"),
        mpatches.Patch(color=cmap(0.0), label="collapses"),
    ], loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "summary_survivors.png"), dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- report
def write_report(data, agg_single, agg_pair, surv_rows):
    lines = [
        "Observational-noise robustness of PISN-vs-CCSN signatures (test_v10)",
        "Class/bins identical to test_v9 (pos = f_pisn>=0.5 & finite mass & in bin; "
        "neg = f_pisn==0).",
        "Noise: Gaussian(0, sigma) dex added to every cell of X, R=10 realizations, "
        "seed = 1000*round(sigma*100)+r.",
        "Survival @0.1 dex: mean separability >= 0.90 AND KS p < 0.05.",
        "",
        "== dataset source check ==",
    ]
    for ds in ["4_3", "5", "L", "N", "taka_0.1"]:
        d = data[ds]
        lines.append(f"  {ds:<8} pisn_source={d['pisn_source']}  "
                     f"sn_source={d['sn_source']}  dex_error={d['dex_error']}")
    same = (data["taka_0.1"]["pisn_source"] == data["4_3"]["pisn_source"] and
            data["taka_0.1"]["sn_source"] == data["4_3"]["sn_source"])
    lines.append(f"  taka_0.1 has the SAME yield sources as 4_3: {same}. "
                 f"(It is an independent realization, not 4_3+noise: pisn_mass differs.)")
    lines.append("")

    surv_by = {(r["dataset"], r["bin"], r["feature"]): r for r in surv_rows}
    for ds in ["4_3", "5", "L", "N"]:
        for b, lo, hi in BINS:
            top = top_noiseless(agg_single, ds, b)
            lines.append(f"== {ds} / bin {b} ==")
            lines.append(f"  top-8 @0.0 dex -> sep@0.10 -> sep@0.20 -> survives?")
            for name in top:
                r = surv_by[(ds, b, name)]
                flag = "SURVIVES" if r["survives_0p1"] else "collapse"
                lines.append(f"    {name:<9} {r['sep_0']:.3f} -> "
                             f"{r['sep_0p10']:.3f} -> {r['sep_0p20']:.3f}  "
                             f"(KS_p@0.10={r['ks_p_0p10']:.1e})  {flag}")
            # top-5 pairs at 0.10
            pairs = [p for p in agg_pair[ds][0.10][b]][:5]
            lines.append("  top-5 pairs @0.10 dex:")
            for p in pairs:
                lines.append(f"    {p['feat_a']:<8} & {p['feat_b']:<8} "
                             f"pair_auc={p['pair_auc_mean']:.3f} "
                             f"delta={p['delta_mean']:+.3f}")
            lines.append("")
    with open(os.path.join(OUT_DIR, "noise_robustness_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------- main
def main():
    _style()
    t0 = time.time()
    data, real_single, real_pair = run()
    print(f"compute done in {time.time()-t0:.1f}s", flush=True)
    single_rows, agg_single, pair_rows, agg_pair = aggregate(data, real_single, real_pair)
    surv_rows = survival_table(agg_single)

    # --- CSVs ---
    with open(os.path.join(OUT_DIR, "noise_single_metrics.csv"), "w", newline="") as f:
        fieldnames = ["dataset", "sigma", "R", "bin", "feature", "ref", "element",
                      "sep_mean", "sep_std", "auc_mean", "auc_std", "direction",
                      "ks_stat_mean", "ks_p_mean", "n_pos", "n_neg"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in single_rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    with open(os.path.join(OUT_DIR, "noise_pair_metrics.csv"), "w", newline="") as f:
        fieldnames = ["dataset", "sigma", "R", "bin", "rank", "feat_a", "feat_b",
                      "pair_auc_mean", "pair_auc_std", "max_single_auc_mean",
                      "delta_mean", "delta_std"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in pair_rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    with open(os.path.join(OUT_DIR, "signature_survival.csv"), "w", newline="") as f:
        fieldnames = ["dataset", "bin", "feature", "ref", "element",
                      "sep_0", "sep_0p05", "sep_0p10", "sep_0p20",
                      "ks_p_0p10", "survives_0p1"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in surv_rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    # --- plots ---
    for ds in ["4_3", "5", "L", "N"]:
        for b, lo, hi in BINS:
            plot_degradation(agg_single, ds, b)
            plot_survival_heatmap(agg_single, ds, b)
            plot_pair_top(agg_pair, ds, b)
    plot_degradation_all(agg_single)
    for b, lo, hi in BINS:
        plot_taka_vs_injected(agg_single, b)
    plot_summary_survivors(agg_single, surv_rows)

    write_report(data, agg_single, agg_pair, surv_rows)

    files = sorted(os.listdir(OUT_DIR))
    print("\nProduced files (size bytes):")
    for fn in files:
        p = os.path.join(OUT_DIR, fn)
        print(f"  {fn:<40} {os.path.getsize(p):>8}")
    print(f"\nTotal: {len(files)} files in {OUT_DIR}")
    print(f"single rows: {len(single_rows)}  pair rows: {len(pair_rows)}  "
          f"survival rows: {len(surv_rows)}")


if __name__ == "__main__":
    main()
