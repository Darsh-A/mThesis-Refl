"""Corrected multi-ratio 2D OOD overlay (test_v11_1).

test_v11's ood_2d plots had a bug: for the Xiang source, [Ca/Mg] and [Si/Mg]
were computed from DIFFERENT row subsets (one filtered on q_[Ca/Fe]==1, the
other on q_[Si/Fe]==1) and then zipped after truncating to min length, so real
stars were mis-paired.  This follow-up fixes it: every real star is kept only
when ALL components of BOTH ratios are measured for the SAME star, and we now
show MULTIPLE ratio pairs (including Cu for Xiang, Zn for Li22), never
imputing missing values for either the sim or the real data.

Simulated features are reconstructed from the stored pairwise columns via
test_v9's build_pool (which returns the FE / MG families directly, no imputer).
The two needed ratio columns are taken as-is; non-finite values are MASKED
(never filled).

Real sources:
  * Li+22 (load_obs_li): [X/Fe]=XFe of ion X; [X/Mg]=[X/Fe]-[Mg/Fe] (same star).
    Upper limits (l_XFe=="<") and missing -> NaN.  No Cu.
  * Xiang/LAMOST (xiang_catalog.csv): q==1 & [Fe/H]<-2; [X/Mg]=[X/Fe]-[Mg/Fe]
    (same row).  Has Cu/Ti/Ni; no Zn.

Outputs (src/randomForest/results/test_v11_1/):
  availability.csv, ood_2d_overlap.csv, test_v11_1_report.txt
  01_xiang_2d/ood_2d_xiang_<ds>.png, 02_li22_2d/ood_2d_li22_<ds>.png,
  03_overlay_all/ood_2d_all_<ds>.png
"""

import csv
import importlib.util
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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

# --- import test_v9 helpers (build_pool, positive_mask, DATA_DIR) ---
V9_PATH = os.path.join(ROOT, "src", "randomForest", "results",
                       "test_v9_signature_separability.py")
_spec = importlib.util.spec_from_file_location("v9sig", V9_PATH)
v9 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v9)

OUT_DIR = os.path.join(ROOT, "src", "randomForest", "results", "test_v11_1")
os.makedirs(OUT_DIR, exist_ok=True)

_PLOT_SUBDIRS = {
    "ood_2d_xiang_": "01_xiang_2d",
    "ood_2d_li22_": "02_li22_2d",
    "ood_2d_all_": "03_overlay_all",
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

# ratio pairs (priority order)
PAIRS = [
    ("[Ca/Mg]", "[Si/Mg]"),
    ("[Ca/Mg]", "[Cu/Mg]"),
    ("[Ca/Mg]", "[Cu/Fe]"),
    ("[Ca/Fe]", "[Cu/Fe]"),
    ("[Si/Mg]", "[Cu/Fe]"),
    ("[Ca/Mg]", "[Zn/Mg]"),
    ("[Ca/Fe]", "[Zn/Fe]"),
    ("[Ca/Mg]", "[Ni/Fe]"),
    ("[Ca/Mg]", "[Ti/Mg]"),
    ("[Si/Mg]", "[Ti/Mg]"),
]

# Li22 ion key for each element of interest (Cu is absent in Li22)
LI_ION = {"Ca": "CaI", "Mg": "MgI", "Si": "SiI", "Fe": "FeI",
          "Zn": "ZnI", "Ni": "NiI", "Ti": "TiII", "Cu": None}


def _style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        try:
            plt.style.use("ggplot")
        except Exception:
            pass


# ---------------------------------------------------------------- ratio helpers
def ratio_elements(ratio):
    x, y = ratio.strip("[]").split("/")
    els = {x}
    if y != "Fe":
        els.add(y)
    return els


def pair_elements(pair):
    return ratio_elements(pair[0]) | ratio_elements(pair[1])


# ---------------------------------------------------------------- sim load
def load_sim():
    sim = {}
    for ds, path in DATASETS.items():
        X, y, diag = load_dataset(path)
        y = np.asarray(y, dtype=float)
        pm = np.asarray(diag["pisn_mass"], dtype=float)
        is_pisn = np.isfinite(pm)
        feature_names = list(diag["feature_names"])
        elems, fe, mg, pool_names, pool = v9.build_pool(X, feature_names)
        # NOTE: fe/mg are raw sign-reconstructed columns; NO imputation here.
        sim[ds] = {
            "fe": fe, "mg": mg, "elems": set(elems),
            "neg": (y == 0.0),
            "pos": v9.positive_mask(y, pm, is_pisn, None, None),
        }
    return sim


def sim_col(d, ratio):
    x, y = ratio.strip("[]").split("/")
    if y == "Fe":
        return d["fe"].get(f"[{x}/Fe]")
    return d["mg"].get(f"[{x}/Mg]")


# ---------------------------------------------------------------- real load
def li_xfe(star, el):
    ion = LI_ION.get(el)
    if ion is None:
        return np.nan
    rec = star["abundances"].get(ion)
    if not rec:
        return np.nan
    v = rec.get("XFe")
    l = rec.get("l_XFe")
    if v is None or l == "<":
        return np.nan
    try:
        v = float(v)
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def load_li22():
    stars = load_obs_li()
    els = ["Ca", "Mg", "Si", "Zn", "Ni", "Ti", "Cu"]
    arr = {el: np.full(len(stars), np.nan) for el in els}
    labels = [s["label"] for s in stars]
    for i, s in enumerate(stars):
        for el in els:
            arr[el][i] = li_xfe(s, el)
    # predicted f_pisn (q50), joined on label
    pred = {}
    with open(PRED_CSV) as f:
        for r in csv.DictReader(f):
            pred[r["star"]] = float(r["f_pisn_q50"])
    fpisn = np.array([pred.get(l, np.nan) for l in labels], dtype=float)
    return arr, labels, fpisn


def li_ratio_arr(arr, ratio):
    x, y = ratio.strip("[]").split("/")
    xv = arr.get(x)
    if xv is None:
        return np.full(len(next(iter(arr.values()))), np.nan)
    if y == "Fe":
        return xv.copy()
    mv = arr["Mg"]
    return xv - mv  # NaN propagates where either missing


def load_xiang():
    import pandas as pd
    els = ["Ca", "Mg", "Si", "Cu", "Ti", "Ni"]
    cols = ["[Fe/H]", "q_[Fe/H]"]
    for el in els:
        cols += [f"[{el}/Fe]", f"q_[{el}/Fe]"]
    df = pd.read_csv(XIANG_CSV, usecols=cols)
    df = df[df["q_[Fe/H]"] == 1]
    df = df[pd.to_numeric(df["[Fe/H]"], errors="coerce") < -2.0].copy()
    arr = {}
    for el in els:
        v = pd.to_numeric(df[f"[{el}/Fe]"], errors="coerce")
        q = df[f"q_[{el}/Fe]"]
        arr[el] = np.where((q.to_numpy() == 1) & np.isfinite(v.to_numpy()), v.to_numpy(), np.nan)
    return arr


def xiang_ratio_arr(arr, ratio):
    x, y = ratio.strip("[]").split("/")
    xv = arr.get(x)
    if xv is None:
        return np.full(len(next(iter(arr.values()))), np.nan)
    if y == "Fe":
        return xv.copy()
    mv = arr["Mg"]
    return xv - mv


# ---------------------------------------------------------------- overlap
def mask2d(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def overlap2d(real_x, real_y, neg_x, neg_y, pos_x, pos_y):
    rx, ry = mask2d(real_x, real_y)
    n_real = len(rx)
    np_x05, np_x95 = np.percentile(neg_x, [5, 95])
    np_y05, np_y95 = np.percentile(neg_y, [5, 95])
    pp_x05, pp_x95 = np.percentile(pos_x, [5, 95])
    pp_y05, pp_y95 = np.percentile(pos_y, [5, 95])
    in_neg = (rx >= np_x05) & (rx <= np_x95) & (ry >= np_y05) & (ry <= np_y95)
    in_pos = (rx >= pp_x05) & (rx <= pp_x95) & (ry >= pp_y05) & (ry <= pp_y95)
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
    li_arr, li_labels, li_fp = load_li22()
    xiang_arr = load_xiang()

    # availability tables
    li_available_elems = {el for el, a in li_arr.items() if np.isfinite(a).any()}
    xiang_available_elems = {el for el, a in xiang_arr.items() if np.isfinite(a).any()}
    print("\nLi22 ion availability (element -> n measured / n stars):")
    for el, a in li_arr.items():
        print(f"  {el:>3} ({LI_ION.get(el)}): {int(np.isfinite(a).sum())}/{len(a)}")
    print("\nXiang element availability (n reliable / n VMP stars):")
    for el, a in xiang_arr.items():
        print(f"  {el:>3}: {int(np.isfinite(a).sum())}/{len(a)}")

    avail_rows = []
    overlap_rows = []

    for ds in DATASETS:
        d = sim[ds]
        # precompute sim 2D points per pair once per dataset
        for pidx, (r1, r2) in enumerate(PAIRS, 1):
            plabel = f"P{pidx}"
            sc1 = sim_col(d, r1)
            sc2 = sim_col(d, r2)
            for real_source in ["Xiang", "Li22"]:
                # --- availability ---
                needed = pair_elements((r1, r2))
                sim_missing = [e for e in needed if e not in d["elems"]]
                if real_source == "Xiang":
                    real_elems = xiang_available_elems
                    real_arr = xiang_arr
                    real_ratio = xiang_ratio_arr
                else:
                    real_elems = li_available_elems
                    real_arr = li_arr
                    real_ratio = li_ratio_arr
                real_missing = [e for e in needed if e not in real_elems]
                available = (sc1 is not None and sc2 is not None
                             and not sim_missing and not real_missing)
                reason = ""
                if sc1 is None or sc2 is None or sim_missing:
                    miss = sim_missing or [e for e in needed
                                           if (e != "Mg" and e not in d["elems"])]
                    reason = f"sim {ds} missing element(s) {sorted(set(miss))}"
                elif real_missing:
                    reason = f"{real_source} missing element(s) {sorted(real_missing)}"
                else:
                    reason = "ok"
                avail_rows.append({
                    "dataset": ds, "pair": f"{plabel} {r1} vs {r2}",
                    "real_source": real_source, "available": bool(available),
                    "reason": reason,
                })

                if not available:
                    continue

                # sim 2D points (masked)
                neg_x, neg_y = mask2d(sc1[d["neg"]], sc2[d["neg"]])
                pos_x, pos_y = mask2d(sc1[d["pos"]], sc2[d["pos"]])
                # real 2D points (per-star, same row)
                rx_arr = real_ratio(real_arr, r1)
                ry_arr = real_ratio(real_arr, r2)
                rx, ry = mask2d(rx_arr, ry_arr)
                ov = overlap2d(rx, ry, neg_x, neg_y, pos_x, pos_y)
                ov.update({"dataset": ds, "pair": f"{plabel} {r1} vs {r2}",
                           "real_source": real_source})
                overlap_rows.append(ov)
                print(f"[{ds}] {plabel} {r1} vs {r2} {real_source}: "
                      f"n_real={ov['n_real']} neg={ov['frac_in_neg']:.2f} "
                      f"pos={ov['frac_in_pos']:.2f} neither={ov['frac_in_neither']:.2f}")

        # --- plots per dataset ---
        plot_source_figure(sim, d, ds, "Xiang", xiang_arr, xiang_ratio_arr,
                           xiang_available_elems, li_fp=None)
        plot_source_figure(sim, d, ds, "Li22", li_arr, li_ratio_arr,
                           li_available_elems, li_fp=li_fp)
        plot_all_figure(sim, d, ds, xiang_arr, xiang_ratio_arr,
                        xiang_available_elems, li_arr, li_ratio_arr, li_available_elems)

    # --- CSVs ---
    with open(os.path.join(OUT_DIR, "availability.csv"), "w", newline="") as f:
        fieldnames = ["dataset", "pair", "real_source", "available", "reason"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in avail_rows:
            w.writerow(r)

    with open(os.path.join(OUT_DIR, "ood_2d_overlap.csv"), "w", newline="") as f:
        fieldnames = ["dataset", "pair", "real_source", "n_real",
                      "frac_in_neg", "frac_in_pos", "frac_in_neither"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in overlap_rows:
            w.writerow({k: r[k] for k in fieldnames})

    write_report(avail_rows, overlap_rows)

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


# ---------------------------------------------------------------- plotting
def _available_pairs(sim, d, ds, real_elems):
    avail = []
    for pidx, (r1, r2) in enumerate(PAIRS, 1):
        sc1 = sim_col(d, r1)
        sc2 = sim_col(d, r2)
        needed = pair_elements((r1, r2))
        if sc1 is not None and sc2 is not None and all(e in d["elems"] for e in needed) \
                and all(e in real_elems for e in needed):
            avail.append((f"P{pidx}", r1, r2))
    return avail


def _grid_dims(n):
    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))
    return rows, cols


def plot_source_figure(sim, d, ds, source, real_arr, real_ratio, real_elems, li_fp=None):
    avail = _available_pairs(sim, d, ds, real_elems)
    if not avail:
        return
    rows, cols = _grid_dims(len(avail))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    sc_mappable = None
    for k, (plabel, r1, r2) in enumerate(avail):
        ax = axes.flat[k]
        sc1 = sim_col(d, r1)
        sc2 = sim_col(d, r2)
        neg_x, neg_y = mask2d(sc1[d["neg"]], sc2[d["neg"]])
        pos_x, pos_y = mask2d(sc1[d["pos"]], sc2[d["pos"]])
        rx_arr = real_ratio(real_arr, r1)
        ry_arr = real_ratio(real_arr, r2)
        rx, ry = mask2d(rx_arr, ry_arr)
        ax.scatter(neg_x, neg_y, s=5, c="tab:gray", alpha=0.3, label=f"f=0 (N={len(neg_x)})")
        ax.scatter(pos_x, pos_y, s=5, c="tab:red", alpha=0.5, label=f"f>=0.5 (N={len(pos_x)})")
        if source == "Li22":
            fp = li_fp[np.isfinite(rx_arr) & np.isfinite(ry_arr)]
            sc = ax.scatter(rx, ry, s=30, c=fp, cmap="viridis", vmin=0, vmax=1,
                            edgecolor="k", linewidth=0.4, label=f"Li22 (N={len(rx)})")
            sc_mappable = sc
        else:
            ax.scatter(rx, ry, s=30, c="tab:green", edgecolor="k", linewidth=0.4,
                       label=f"Xiang (N={len(rx)})")
        ax.set_xlabel(f"{r1} (dex)")
        ax.set_ylabel(f"{r2} (dex)")
        ax.set_title(f"{plabel}: {r1} vs {r2}")
        ax.legend(fontsize=7)
        ax.axhline(0, color="k", ls=":", lw=0.5)
        ax.axvline(0, color="k", ls=":", lw=0.5)
    # hide unused axes
    for ax in axes.flat[len(avail):]:
        ax.set_visible(False)
    fig.suptitle(f"test_v11_1 [{ds}]: sim vs {source} (multi-ratio 2D OOD)", fontsize=12)
    if source == "Li22" and sc_mappable is not None:
        cb = fig.colorbar(sc_mappable, ax=axes.flat[:len(avail)].tolist(),
                          fraction=0.03, pad=0.04)
        cb.set_label("predicted f_pisn (q50)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    tag = "li22" if source == "Li22" else "xiang"
    fig.savefig(_plot_path(f"ood_2d_{tag}_{ds}.png"), dpi=130)
    plt.close(fig)


def plot_all_figure(sim, d, ds, xiang_arr, xiang_ratio, xiang_elems,
                    li_arr, li_ratio, li_elems):
    avail = _available_pairs(sim, d, ds, xiang_elems | li_elems)
    if not avail:
        return
    rows, cols = _grid_dims(len(avail))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    for k, (plabel, r1, r2) in enumerate(avail):
        ax = axes.flat[k]
        sc1 = sim_col(d, r1)
        sc2 = sim_col(d, r2)
        neg_x, neg_y = mask2d(sc1[d["neg"]], sc2[d["neg"]])
        pos_x, pos_y = mask2d(sc1[d["pos"]], sc2[d["pos"]])
        ax.scatter(neg_x, neg_y, s=5, c="tab:gray", alpha=0.3)
        ax.scatter(pos_x, pos_y, s=5, c="tab:red", alpha=0.5)
        # Li22 (blue)
        if all(e in li_elems for e in pair_elements((r1, r2))):
            lx, ly = mask2d(li_ratio(li_arr, r1), li_ratio(li_arr, r2))
            ax.scatter(lx, ly, s=24, c="tab:blue", edgecolor="k", linewidth=0.3,
                       label=f"Li22 (N={len(lx)})")
        # Xiang (green)
        if all(e in xiang_elems for e in pair_elements((r1, r2))):
            xx, xy = mask2d(xiang_ratio(xiang_arr, r1), xiang_ratio(xiang_arr, r2))
            ax.scatter(xx, xy, s=24, c="tab:green", edgecolor="k", linewidth=0.3,
                       label=f"Xiang (N={len(xx)})")
        ax.set_xlabel(f"{r1} (dex)")
        ax.set_ylabel(f"{r2} (dex)")
        ax.set_title(f"{plabel}: {r1} vs {r2}")
        ax.legend(fontsize=7)
    for ax in axes.flat[len(avail):]:
        ax.set_visible(False)
    fig.suptitle(f"test_v11_1 [{ds}]: sim vs Li22 (blue) + Xiang (green)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(_plot_path(f"ood_2d_all_{ds}.png"), dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- report
def write_report(avail_rows, overlap_rows):
    lines = [
        "test_v11_1: corrected multi-ratio 2D OOD overlay",
        "FIX: v11 computed Xiang [Ca/Mg] and [Si/Mg] from DIFFERENT row subsets",
        "(q_[Ca/Fe]==1 vs q_[Si/Fe]==1) then zipped after min-length truncation,",
        "mis-pairing stars.  Here every real point requires ALL components of",
        "BOTH ratios measured for the SAME star; no imputation anywhere.",
        "",
        "== availability (dataset x pair x real source) ==",
    ]
    for r in avail_rows:
        if r["available"]:
            lines.append(f"  {r['dataset']} {r['pair']:<28} {r['real_source']:<6} AVAILABLE")
        else:
            lines.append(f"  {r['dataset']} {r['pair']:<28} {r['real_source']:<6} SKIP ({r['reason']})")
    lines.append("")
    lines.append("== overlap (frac of real stars in sim f=0 / f>=0.5 / neither 5-95% boxes) ==")
    for r in overlap_rows:
        lines.append(f"  {r['dataset']} {r['pair']:<28} {r['real_source']:<6} n={r['n_real']:>4} "
                     f"neg={r['frac_in_neg']:.2f} pos={r['frac_in_pos']:.2f} "
                     f"neither={r['frac_in_neither']:.2f}")
    lines.append("")
    lines.append("Note: test_v11's ood_2d Xiang points were mis-paired; this run is the")
    lines.append("corrected per-star version.")
    with open(os.path.join(OUT_DIR, "test_v11_1_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
