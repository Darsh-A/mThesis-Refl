"""Exhaustive all-[X/Y] permutation OOD study (test_v11_2).

Extends v11.1 from a hand-picked 10-pair list to EVERY unordered element-pair
ratio [X/Y] (Fe as the reference denominator) over the curated signature set
{Ca, Mg, Si, Ti, Ni, Cr, Mn, Co, Al, Zn, Cu, C, N, O}, against the simulated
f_pisn=0 (CCSN-only) and f_pisn>=0.5 (PISN) clouds, WITHOUT the Mg bottleneck.

Per-star completeness: a real point is used only if ALL component abundances
of BOTH ratios are measured & reliable for the SAME star (Li22: l_XFe != "<"
and finite; Xiang: q_[X/Fe]==1 and finite).  NaN propagates; no imputation.

Sim ratios come from the stored pairwise columns (sign * X[:,j], the same
reconstruction test_v9 uses).  Real [X/Y] = [X/Fe] - [Y/Fe].

Outputs (src/randomForest/results/test_v11_2/):
  element_presence.csv, single_ratio_ood.csv, ratio_pair_ood.csv,
  test_v11_2_report.txt,
  01_single_1d/, 02_top_pair_2d/, 03_ranking/, 04_coverage/
"""

import csv
import importlib.util
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
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
from src.load_data import load_obs_li  # noqa: E402

# --- import test_v9 helpers ---
V9_PATH = os.path.join(ROOT, "src", "randomForest", "results",
                       "test_v9_signature_separability.py")
_spec = importlib.util.spec_from_file_location("v9sig", V9_PATH)
v9 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v9)

OUT_DIR = os.path.join(ROOT, "src", "randomForest", "results", "test_v11_2")
os.makedirs(OUT_DIR, exist_ok=True)

_PLOT_SUBDIRS = {
    "single_1d_": "01_single_1d",
    "top_pair_2d_": "02_top_pair_2d",
    "ranking_": "03_ranking",
    "coverage_": "04_coverage",
}


def _plot_path(name):
    for prefix, sub in _PLOT_SUBDIRS.items():
        if name.startswith(prefix):
            folder = os.path.join(OUT_DIR, sub)
            os.makedirs(folder, exist_ok=True)
            return os.path.join(folder, name)
    return os.path.join(OUT_DIR, name)


DATA_DIR = v9.DATA_DIR
DATASETS = {
    "4_3": os.path.join(DATA_DIR, "dataset_TakaLimon_realisticNoise_v4_3.npz"),
    "5": os.path.join(DATA_DIR, "dataset_Nomotov5.npz"),
    "L": os.path.join(DATA_DIR, "dataset_Limongi_tpop42.npz"),
    "N": os.path.join(DATA_DIR, "dataset_Nomoto_tpop42.npz"),
}
PRED_CSV = os.path.join(ROOT, "src", "randomForest", "results", "obs_li_predictions.csv")
XIANG_CSV = os.path.join(ROOT, "src", "randomForest", "results", "xiang", "xiang_catalog.csv")

CURATED = ["Ca", "Mg", "Si", "Ti", "Ni", "Cr", "Mn", "Co", "Al", "Zn", "Cu", "C", "N", "O"]
FE = "Fe"

# Li22 ion key per curated element (None = element not measurable in Li22)
LI_ION = {"Ca": "CaI", "Mg": "MgI", "Si": "SiI", "Ti": "TiII", "Ni": "NiI",
          "Cr": "CrI", "Mn": "MnI", "Co": "CoI", "Al": None, "Zn": "ZnI",
          "Cu": None, "C": "C", "N": None, "O": None}

MIN_N_SINGLE = 10
MIN_N_PAIR = 10
TOP_SINGLE = 20
TOP_PAIR = 30
N_CHUNKS = 16

# representative dataset per survey (for plots)
REP_DS = {"Li22": "4_3", "Xiang": "L"}


def _style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        try:
            plt.style.use("ggplot")
        except Exception:
            pass


# ---------------------------------------------------------------- ratio utils
def _sort_key(e):
    return (e == FE, e)


def canonical_ratio(x, y):
    a, b = sorted([x, y], key=_sort_key)
    return f"[{a}/{b}]"


def ratio_xy(ratio):
    return tuple(ratio.strip("[]").split("/"))


def generate_ratios(elems):
    els = sorted(set(elems), key=_sort_key)
    return [canonical_ratio(els[i], els[j])
            for i in range(len(els)) for j in range(i + 1, len(els))]


def ion_to_element(ion):
    return re.sub(r"(I|V|X)+$", "", ion)


# ---------------------------------------------------------------- sim load
def load_sim():
    sim = {}
    for ds, path in DATASETS.items():
        X, y, diag = load_dataset(path)
        y = np.asarray(y, dtype=float)
        pm = np.asarray(diag["pisn_mass"], dtype=float)
        is_pisn = np.isfinite(pm)
        fn = list(diag["feature_names"])
        # ratio -> column (both orientations via sign flip), NO imputation
        ratio_col = {}
        elems = set()
        for j, nm in enumerate(fn):
            a, b = nm.strip("[]").split("/")
            ratio_col[f"[{a}/{b}]"] = X[:, j]
            ratio_col[f"[{b}/{a}]"] = -X[:, j]
            elems.add(a)
            elems.add(b)
        # restrict to curated signature set + Fe
        avail_elems = (elems & set(CURATED)) | {FE}
        ratio_names = generate_ratios(avail_elems)
        pool = np.column_stack([ratio_col[n] for n in ratio_names])
        idx = {n: k for k, n in enumerate(ratio_names)}
        neg = (y == 0.0)
        pos = v9.positive_mask(y, pm, is_pisn, None, None)
        sel = neg | pos
        Xstd = StandardScaler().fit_transform(pool[sel])
        binary = pos[sel].astype(int)
        sim[ds] = {
            "ratio_names": ratio_names, "pool": pool, "idx": idx,
            "neg": neg, "pos": pos, "elems": elems,
            "Xstd": Xstd, "binary": binary, "sel": sel,
        }
    return sim


# ---------------------------------------------------------------- real load
def load_li22():
    stars = load_obs_li()
    # dynamically enumerate element presence (ion -> element)
    counts = {}
    for s in stars:
        for ion, rec in s["abundances"].items():
            v = rec.get("XFe")
            l = rec.get("l_XFe")
            if v is None or l == "<":
                continue
            el = ion_to_element(ion)
            counts[el] = counts.get(el, 0) + 1
    # per-star arrays for curated elements + Fe
    arr = {el: np.full(len(stars), np.nan) for el in CURATED}
    arr[FE] = np.zeros(len(stars))
    for i, s in enumerate(stars):
        for el in CURATED:
            ion = LI_ION.get(el)
            if ion is None:
                continue
            rec = s["abundances"].get(ion)
            if not rec:
                continue
            v = rec.get("XFe")
            l = rec.get("l_XFe")
            if v is None or l == "<":
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                arr[el][i] = v
    pred = {}
    with open(PRED_CSV) as f:
        for r in csv.DictReader(f):
            pred[r["star"]] = float(r["f_pisn_q50"])
    labels = [s["label"] for s in stars]
    fpisn = np.array([pred.get(l, np.nan) for l in labels], dtype=float)
    present = {el for el in CURATED if np.isfinite(arr[el]).any()}
    return arr, fpisn, counts, present


def load_xiang():
    import pandas as pd
    els = [e for e in CURATED if e in
           {"C", "N", "O", "Mg", "Al", "Si", "Ca", "Ti", "Cr", "Mn", "Co", "Ni", "Cu"}]
    cols = ["[Fe/H]", "q_[Fe/H]"]
    for el in els:
        cols += [f"[{el}/Fe]", f"q_[{el}/Fe]"]
    df = pd.read_csv(XIANG_CSV, usecols=cols)
    df = df[df["q_[Fe/H]"] == 1]
    df = df[pd.to_numeric(df["[Fe/H]"], errors="coerce") < -2.0].copy()
    arr = {el: np.full(len(df), np.nan) for el in CURATED}
    arr[FE] = np.zeros(len(df))
    counts = {}
    for el in els:
        v = pd.to_numeric(df[f"[{el}/Fe]"], errors="coerce").to_numpy()
        q = df[f"q_[{el}/Fe]"].to_numpy()
        reliable = np.where((q == 1) & np.isfinite(v), v, np.nan)
        arr[el] = reliable
        counts[el] = int(np.isfinite(reliable).sum())
    present = {el for el in CURATED if np.isfinite(arr[el]).any()}
    return arr, counts, present


def real_ratio_arr(arr, ratio):
    x, y = ratio_xy(ratio)
    xv = arr.get(x)
    yv = arr.get(y)
    if xv is None or yv is None:
        return None
    return xv - yv


# ---------------------------------------------------------------- metrics
def mask2d(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def sim_auc_1d(col, neg, pos):
    vals = np.r_[col[neg], col[pos]]
    ybin = np.r_[np.zeros(int(neg.sum())), np.ones(int(pos.sum()))].astype(int)
    auc = float(roc_auc_score(ybin, vals))
    return max(auc, 1.0 - auc)


def _auc2d_local(Xstd, binary, i, j):
    X2 = Xstd[:, [i, j]]
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = cross_val_predict(clf, X2, binary, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(binary, oof))


def _pair_chunk(chunk, Xstd, binary):
    return [(i, j, _auc2d_local(Xstd, binary, i, j)) for (i, j) in chunk]


def compute_sim_auc2d(sim, ds):
    d = sim[ds]
    n = len(d["ratio_names"])
    pair_idx = [(i, j) for i in range(n) for j in range(i + 1, n)]
    chunks = [pair_idx[k::N_CHUNKS] for k in range(N_CHUNKS)]
    res = Parallel(n_jobs=-1)(
        delayed(_pair_chunk)(c, d["Xstd"], d["binary"]) for c in chunks)
    flat = [x for sub in res for x in sub]
    return {(i, j): a for (i, j, a) in flat}


def overlap_1d(real_vals, neg_vals, pos_vals):
    rv = np.asarray(real_vals, dtype=float)
    rv = rv[np.isfinite(rv)]
    n_real = len(rv)
    n_p05, n_p95 = np.percentile(neg_vals, [5, 95])
    p_p05, p_p95 = np.percentile(pos_vals, [5, 95])
    in_neg = (rv >= n_p05) & (rv <= n_p95)
    in_pos = (rv >= p_p05) & (rv <= p_p95)
    in_neither = ~(in_neg | in_pos)
    return {
        "n_real": n_real,
        "frac_in_neg": float(in_neg.mean()) if n_real else float("nan"),
        "frac_in_pos": float(in_pos.mean()) if n_real else float("nan"),
        "frac_in_neither": float(in_neither.mean()) if n_real else float("nan"),
    }


def overlap_2d(real_x, real_y, neg_x, neg_y, pos_x, pos_y):
    rx, ry = mask2d(real_x, real_y)
    n_real = len(rx)
    n_x05, n_x95 = np.percentile(neg_x, [5, 95])
    n_y05, n_y95 = np.percentile(neg_y, [5, 95])
    p_x05, p_x95 = np.percentile(pos_x, [5, 95])
    p_y05, p_y95 = np.percentile(pos_y, [5, 95])
    in_neg = (rx >= n_x05) & (rx <= n_x95) & (ry >= n_y05) & (ry <= n_y95)
    in_pos = (rx >= p_x05) & (rx <= p_x95) & (ry >= p_y05) & (ry <= p_y95)
    in_neither = ~(in_neg | in_pos)
    return {
        "n_real": n_real,
        "frac_in_neg": float(in_neg.mean()) if n_real else float("nan"),
        "frac_in_pos": float(in_pos.mean()) if n_real else float("nan"),
        "frac_in_neither": float(in_neither.mean()) if n_real else float("nan"),
    }


# ---------------------------------------------------------------- main
def main():
    _style()
    sim = load_sim()
    li_arr, li_fp, li_counts, li_present = load_li22()
    xiang_arr, xiang_counts, xiang_present = load_xiang()

    surveys = {
        "Li22": {"arr": li_arr, "present": li_present, "fp": li_fp},
        "Xiang": {"arr": xiang_arr, "present": xiang_present, "fp": None},
    }

    # --- element presence CSV ---
    presence_rows = []
    for el in CURATED + [FE]:
        for ds in DATASETS:
            presence_rows.append({
                "element": el, "source_type": "dataset", "source_name": ds,
                "present": bool(el in sim[ds]["elems"]),
                "reliable_count": 2000 if el in sim[ds]["elems"] else 0,
            })
        presence_rows.append({
            "element": el, "source_type": "survey", "source_name": "Li22",
            "present": bool(el in li_present or el == FE),
            "reliable_count": li_counts.get(el, 0) if el != FE else len(li_arr[FE]),
        })
        presence_rows.append({
            "element": el, "source_type": "survey", "source_name": "Xiang",
            "present": bool(el in xiang_present or el == FE),
            "reliable_count": xiang_counts.get(el, 0) if el != FE else len(xiang_arr[FE]),
        })
    with open(os.path.join(OUT_DIR, "element_presence.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["element", "source_type", "source_name",
                                          "present", "reliable_count"])
        w.writeheader()
        for r in presence_rows:
            w.writerow(r)

    # --- sim 2D AUC (per dataset, all sim pairs) ---
    print("\nComputing sim-side 2D AUC ...", flush=True)
    sim_auc2d = {ds: compute_sim_auc2d(sim, ds) for ds in DATASETS}

    # --- singles + pairs OOD ---
    single_rows = []
    pair_rows = []
    for survey, sdict in surveys.items():
        arr = sdict["arr"]
        present = sdict["present"]
        for ds in DATASETS:
            d = sim[ds]
            common_elems = (present & d["elems"]) | {FE}
            avail_ratios = generate_ratios(common_elems)
            # sim-side 1D AUC for all sim ratios (cache)
            sim_auc1 = {n: sim_auc_1d(d["pool"][:, d["idx"][n]], d["neg"], d["pos"])
                        for n in d["ratio_names"]}
            for rname in avail_ratios:
                j = d["idx"][rname]
                col = d["pool"][:, j]
                rv = real_ratio_arr(arr, rname)
                if rv is None:
                    continue
                ov = overlap_1d(rv, col[d["neg"]], col[d["pos"]])
                single_rows.append({
                    "survey": survey, "dataset": ds, "ratio": rname,
                    "n_real": ov["n_real"], "frac_in_neg": ov["frac_in_neg"],
                    "frac_in_pos": ov["frac_in_pos"],
                    "frac_in_neither": ov["frac_in_neither"],
                    "sim_auc": sim_auc1[rname],
                })
            # pairs
            rlist = avail_ratios
            pair_list = [(rlist[i], rlist[j]) for i in range(len(rlist))
                         for j in range(i + 1, len(rlist))]
            scored = []
            for (r1, r2) in pair_list:
                i, j = d["idx"][r1], d["idx"][r2]
                a2 = sim_auc2d[ds].get((min(i, j), max(i, j)), sim_auc2d[ds].get((i, j), np.nan))
                rx = real_ratio_arr(arr, r1)
                ry = real_ratio_arr(arr, r2)
                neg_x, neg_y = mask2d(d["pool"][d["neg"], i], d["pool"][d["neg"], j])
                pos_x, pos_y = mask2d(d["pool"][d["pos"], i], d["pool"][d["pos"], j])
                ov = overlap_2d(rx, ry, neg_x, neg_y, pos_x, pos_y)
                scored.append((r1, r2, a2, ov))
            scored.sort(key=lambda t: -t[2])
            for rank, (r1, r2, a2, ov) in enumerate(scored, 1):
                pair_rows.append({
                    "survey": survey, "dataset": ds, "ratio1": r1, "ratio2": r2,
                    "n_real": ov["n_real"], "frac_in_neg": ov["frac_in_neg"],
                    "frac_in_pos": ov["frac_in_pos"],
                    "frac_in_neither": ov["frac_in_neither"],
                    "sim_2d_auc": a2, "rank": rank,
                })
            print(f"[{survey} x {ds}] ratios={len(avail_ratios)} pairs={len(scored)}", flush=True)

    # --- CSVs ---
    with open(os.path.join(OUT_DIR, "single_ratio_ood.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["survey", "dataset", "ratio", "n_real",
                                          "frac_in_neg", "frac_in_pos",
                                          "frac_in_neither", "sim_auc"])
        w.writeheader()
        for r in single_rows:
            w.writerow(r)

    with open(os.path.join(OUT_DIR, "ratio_pair_ood.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["survey", "dataset", "ratio1", "ratio2",
                                          "n_real", "frac_in_neg", "frac_in_pos",
                                          "frac_in_neither", "sim_2d_auc", "rank"])
        w.writeheader()
        for r in pair_rows:
            w.writerow(r)

    # --- plots ---
    for survey in ["Li22", "Xiang"]:
        plot_single_1d(sim, surveys, single_rows, survey)
        plot_top_pair_2d(sim, surveys, pair_rows, survey)
        plot_ranking(single_rows, pair_rows, survey)
        plot_coverage(sim, surveys, survey)

    write_report(single_rows, pair_rows, presence_rows)

    files = sorted(os.listdir(OUT_DIR))
    print("\nProduced files (size bytes):")
    for fn in files:
        p = os.path.join(OUT_DIR, fn)
        print(f"  {fn:<30} {os.path.getsize(p):>8}")
    sub = []
    for folder in sorted(_PLOT_SUBDIRS.values()):
        fp = os.path.join(OUT_DIR, folder)
        if os.path.isdir(fp):
            sub.extend((folder, p) for p in sorted(os.listdir(fp)))
    print(f"\nTotal: {len(files)} root entries + {len(sub)} pngs in subfolders")


# ---------------------------------------------------------------- plots
def _top_singles(single_rows, survey, ds, k=TOP_SINGLE):
    rows = [r for r in single_rows if r["survey"] == survey and r["dataset"] == ds
            and r["n_real"] >= MIN_N_SINGLE]
    rows.sort(key=lambda r: -r["sim_auc"])
    return rows[:k]


def _top_pairs(pair_rows, survey, ds, k=TOP_PAIR):
    rows = [r for r in pair_rows if r["survey"] == survey and r["dataset"] == ds
            and r["n_real"] >= MIN_N_PAIR]
    rows.sort(key=lambda r: -r["sim_2d_auc"])
    return rows[:k]


def plot_single_1d(sim, surveys, single_rows, survey):
    ds = REP_DS[survey]
    top = _top_singles(single_rows, survey, ds)
    if not top:
        return
    arr = surveys[survey]["arr"]
    d = sim[ds]
    cols = 5
    rows = int(np.ceil(len(top) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for k, r in enumerate(top):
        ax = axes.flat[k]
        j = d["idx"][r["ratio"]]
        neg_vals = d["pool"][d["neg"], j]
        pos_vals = d["pool"][d["pos"], j]
        ax.hist(neg_vals, bins=30, density=True, alpha=0.5, color="tab:gray",
                label=f"f=0 ({len(neg_vals)})")
        ax.hist(pos_vals, bins=30, density=True, alpha=0.5, color="tab:red",
                label=f"f>=0.5 ({len(pos_vals)})")
        rv_full = real_ratio_arr(arr, r["ratio"])
        if rv_full is None:
            continue
        mask = np.isfinite(rv_full)
        rv = rv_full[mask]
        n = len(rv)
        ymin, ymax = ax.get_ylim()
        if survey == "Li22" and surveys["Li22"]["fp"] is not None:
            fp = surveys["Li22"]["fp"][mask]
            ax.scatter(rv, np.full(n, ymin + 0.02 * (ymax - ymin)), c=fp,
                       cmap="viridis", vmin=0, vmax=1, marker="|", s=60, alpha=0.8)
        else:
            ax.scatter(rv, np.full(n, ymin + 0.02 * (ymax - ymin)), c="tab:green",
                       marker="|", s=60, alpha=0.8)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(r["ratio"] + " (dex)")
        ax.set_title(f"{r['ratio']}  n={n}  auc={r['sim_auc']:.2f}")
        ax.legend(fontsize=5)
    for ax in axes.flat[len(top):]:
        ax.set_visible(False)
    fig.suptitle(f"test_v11_2 [{survey} x {ds}]: top single-ratio 1D OOD", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(_plot_path(f"single_1d_{survey}.png"), dpi=130)
    plt.close(fig)


def plot_top_pair_2d(sim, surveys, pair_rows, survey):
    ds = REP_DS[survey]
    top = _top_pairs(pair_rows, survey, ds)
    if not top:
        return
    arr = surveys[survey]["arr"]
    d = sim[ds]
    cols = 5
    rows = int(np.ceil(len(top) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for k, r in enumerate(top):
        ax = axes.flat[k]
        i, j = d["idx"][r["ratio1"]], d["idx"][r["ratio2"]]
        neg_x, neg_y = mask2d(d["pool"][d["neg"], i], d["pool"][d["neg"], j])
        pos_x, pos_y = mask2d(d["pool"][d["pos"], i], d["pool"][d["pos"], j])
        ax.scatter(neg_x, neg_y, s=4, c="tab:gray", alpha=0.3)
        ax.scatter(pos_x, pos_y, s=4, c="tab:red", alpha=0.4)
        rx = real_ratio_arr(arr, r["ratio1"])
        ry = real_ratio_arr(arr, r["ratio2"])
        rxm, rym = mask2d(rx, ry)
        ax.scatter(rxm, rym, s=18, c="tab:blue" if survey == "Li22" else "tab:green",
                   edgecolor="k", linewidth=0.3)
        ax.set_xlabel(r["ratio1"])
        ax.set_ylabel(r["ratio2"])
        ax.set_title(f"{r['ratio1']} vs {r['ratio2']}\nn={len(rxm)} auc={r['sim_2d_auc']:.2f}",
                     fontsize=6)
    for ax in axes.flat[len(top):]:
        ax.set_visible(False)
    fig.suptitle(f"test_v11_2 [{survey} x {ds}]: top ratio-pair 2D OOD", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(_plot_path(f"top_pair_2d_{survey}.png"), dpi=130)
    plt.close(fig)


def plot_ranking(single_rows, pair_rows, survey):
    ds = REP_DS[survey]
    s_top = _top_singles(single_rows, survey, ds)
    p_top = _top_pairs(pair_rows, survey, ds)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    labels = [r["ratio"] for r in s_top]
    ax.barh(range(len(s_top)), [r["sim_auc"] for r in s_top], color="tab:blue")
    ax.set_yticks(range(len(s_top)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("sim 1D AUC")
    ax.set_title(f"{survey}: top single ratios (n_real>={MIN_N_SINGLE})")
    ax = axes[1]
    labels = [f"{r['ratio1']}/{r['ratio2'].split('/')[1]}" if False else
              f"{r['ratio1']} & {r['ratio2']}" for r in p_top]
    ax.barh(range(len(p_top)), [r["sim_2d_auc"] for r in p_top], color="tab:orange")
    ax.set_yticks(range(len(p_top)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("sim 2D AUC")
    ax.set_title(f"{survey}: top ratio-pairs (n_real>={MIN_N_PAIR})")
    fig.suptitle(f"test_v11_2 [{survey} x {ds}] ranking", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(_plot_path(f"ranking_{survey}.png"), dpi=130)
    plt.close(fig)


def plot_coverage(sim, surveys, survey):
    ds = REP_DS[survey]
    arr = surveys[survey]["arr"]
    d = sim[ds]
    present = surveys[survey]["present"]
    elems = sorted((present & d["elems"]) | {FE}, key=_sort_key)
    n = len(elems)
    M = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            rname = canonical_ratio(elems[i], elems[j])
            rv = real_ratio_arr(arr, rname)
            M[i, j] = M[j, i] = int(np.isfinite(rv).sum()) if rv is not None else 0
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(M, cmap="viridis", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels(elems, fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(elems, fontsize=8)
    for i in range(n):
        for j in range(n):
            if i != j and np.isfinite(M[i, j]):
                ax.text(j, i, f"{int(M[i,j])}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label="n_real (per-star complete)")
    ax.set_title(f"test_v11_2 [{survey} x {ds}]: n_real coverage across [X/Y] ratios")
    fig.tight_layout()
    fig.savefig(_plot_path(f"coverage_{survey}.png"), dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- report
def write_report(single_rows, pair_rows, presence_rows):
    lines = [
        "test_v11_2: exhaustive all-[X/Y] OOD study",
        "Curated set: " + ",".join(CURATED) + " (+ Fe reference).",
        "Ratios = unordered pairs; [X/Y] = [X/Fe]-[Y/Fe] for real stars.",
        "Per-star completeness; no imputation; 5-95% sim envelopes.",
        "",
        "== element presence ==",
    ]
    for r in presence_rows:
        if r["source_name"] in ("Li22", "Xiang", "4_3", "L"):
            lines.append(f"  {r['source_type']:<7} {r['source_name']:<6} {r['element']:<3} "
                         f"present={str(r['present']):<5} count={r['reliable_count']}")
    lines.append("")
    for survey in ["Li22", "Xiang"]:
        ds = REP_DS[survey]
        lines.append(f"== {survey} (dataset {ds}): top-10 single ratios ==")
        top = [r for r in single_rows if r["survey"] == survey and r["dataset"] == ds
               and r["n_real"] >= MIN_N_SINGLE]
        top.sort(key=lambda r: -r["sim_auc"])
        for r in top[:10]:
            lines.append(f"  {r['ratio']:<9} n={r['n_real']:>4} auc={r['sim_auc']:.3f} "
                         f"neg={r['frac_in_neg']:.2f} pos={r['frac_in_pos']:.2f} "
                         f"neither={r['frac_in_neither']:.2f}")
        lines.append(f"== {survey} (dataset {ds}): top-10 ratio-pairs ==")
        ptop = [r for r in pair_rows if r["survey"] == survey and r["dataset"] == ds
                and r["n_real"] >= MIN_N_PAIR]
        ptop.sort(key=lambda r: -r["sim_2d_auc"])
        for r in ptop[:10]:
            lines.append(f"  {r['ratio1']:<9} vs {r['ratio2']:<9} n={r['n_real']:>4} "
                         f"auc={r['sim_2d_auc']:.3f} neither={r['frac_in_neither']:.2f}")
        lines.append("")
    # non-Mg high-n pairs
    lines.append("== best HIGH-n NON-Mg ratio-pairs (vs Mg-based P1) ==")
    p1_names = {canonical_ratio("Ca", "Mg"), canonical_ratio("Si", "Mg")}
    for survey in ["Li22", "Xiang"]:
        ds = REP_DS[survey]
        nonmg = [r for r in pair_rows if r["survey"] == survey and r["dataset"] == ds
                 and "Mg" not in r["ratio1"] and "Mg" not in r["ratio2"]
                 and r["n_real"] >= 50]
        nonmg.sort(key=lambda r: -r["sim_2d_auc"])
        p1 = [r for r in pair_rows if r["survey"] == survey and r["dataset"] == ds
              and {r["ratio1"], r["ratio2"]} == p1_names]
        p1 = p1[0] if p1 else None
        if p1:
            lines.append(f"  {survey}: P1 {p1['ratio1']} vs {p1['ratio2']} = "
                         f"(n={p1['n_real']}, auc={p1['sim_2d_auc']:.3f}, "
                         f"neither={p1['frac_in_neither']:.2f})")
        else:
            lines.append(f"  {survey}: P1 n/a")
        for r in nonmg[:8]:
            lines.append(f"    {r['ratio1']:<9} vs {r['ratio2']:<9} n={r['n_real']:>4} "
                         f"auc={r['sim_2d_auc']:.3f} neither={r['frac_in_neither']:.2f}")
    lines.append("")
    with open(os.path.join(OUT_DIR, "test_v11_2_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
