"""Within-class scatter and real-star OOD check (test_v11).

A reviewer flagged that test_v9's AUC ~ 1.0 vs the f_pisn=0 class may be a
ceiling effect: if the simulated CCSN-only (f=0) stars form a near-point in
[Ca/Mg] (or [Si/Mg]), any shifted PISN mean separates perfectly and this says
nothing about real VMP stars.  This test:

  PART A - within-class scatter: distribution of [Ca/Mg], [Si/Mg], [Ca/Fe],
    [Si/Fe], [Mg/Fe] in the f=0 class (and f>=0.5 classes) of the 4 simulated
    datasets, plus the same for the real Li+22 stars.  Flags whether the f=0
    class is a near-point (std << 0.1 dex) for [Ca/Mg] / [Si/Mg].

  PART B - real-star overlay / OOD: overlays the real Li+22 stars (and, as a
    secondary, the Xiang/LAMOST VMP subset) on the simulated f=0 and f>=0.5
    distributions, quantifies how many real stars fall inside each simulated
    envelope (5-95%) vs neither (a gap), and colors Li22 stars by their
    RF-predicted f_pisn from obs_li_predictions.csv.

Simulated features are reconstructed from the cached pairwise ratios via the
test_v9 helper (imported through importlib; test_v9 is untouched).  Class
definitions: neg = (y==0); pos = (y>=0.5), overall and per mass bin.

Outputs (src/randomForest/results/test_v11/):
  within_class_scatter.csv, ood_overlap.csv, test_v11_report.txt
  01_within_class_scatter/within_class_scatter.png
  02_real_overlay_camg/real_overlay_camg_<ds>.png
  03_real_overlay_simg/real_overlay_simg_<ds>.png
  04_ood_2d/ood_2d_<ds>.png
  05_summary/summary_ood.png
"""

import csv
import importlib.util
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer


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

# --- import test_v9 helpers without modifying it ---
V9_PATH = os.path.join(ROOT, "src", "randomForest", "results",
                       "test_v9_signature_separability.py")
_spec = importlib.util.spec_from_file_location("v9sig", V9_PATH)
v9 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v9)

OUT_DIR = os.path.join(ROOT, "src", "randomForest", "results", "test_v11")
os.makedirs(OUT_DIR, exist_ok=True)

_PLOT_SUBDIRS = {
    "within_class_scatter": "01_within_class_scatter",
    "real_overlay_camg_": "02_real_overlay_camg",
    "real_overlay_simg_": "03_real_overlay_simg",
    "ood_2d_": "04_ood_2d",
    "summary_ood": "05_summary",
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

FEATURES = ["[Ca/Mg]", "[Si/Mg]", "[Ca/Fe]", "[Si/Fe]", "[Mg/Fe]"]
RATIO_FEATURES = ["[Ca/Mg]", "[Si/Mg]"]  # Part B focus
NEAR_POINT_THRESH = 0.05  # std below this (dex) is flagged "near-point"

PRED_CSV = os.path.join(ROOT, "src", "randomForest", "results", "obs_li_predictions.csv")
XIANG_CSV = os.path.join(ROOT, "src", "randomForest", "results", "xiang", "xiang_catalog.csv")


def _style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        try:
            plt.style.use("ggplot")
        except Exception:
            pass


# ---------------------------------------------------------------- helpers
def stats(arr):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return None
    q05, q25, q75, q95 = np.percentile(a, [5, 25, 75, 95])
    return {
        "n": int(len(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "iqr": float(q75 - q25),
        "p05": float(q05),
        "p95": float(q95),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def xfe(star, ion):
    """[X/Fe] for a Li+22 star; upper-limit/missing/non-finite -> NaN."""
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


# ---------------------------------------------------------------- data load
def load_sim():
    sim = {}
    for ds, path in DATASETS.items():
        X, y, diag = load_dataset(path)
        y = np.asarray(y, dtype=float)
        pm = np.asarray(diag["pisn_mass"], dtype=float)
        is_pisn = np.isfinite(pm)
        feature_names = list(diag["feature_names"])
        elems, fe, mg, pool_names, pool = v9.build_pool(X, feature_names)
        pool = SimpleImputer(strategy="median").fit_transform(pool)
        idx = {n: pool_names.index(n) for n in FEATURES}
        sim[ds] = {
            "pool": pool,
            "idx": idx,
            "neg": (y == 0.0),
            "pos_all": v9.positive_mask(y, pm, is_pisn, None, None),
            "pos_210": v9.positive_mask(y, pm, is_pisn, 210.0, 230.0),
            "pos_230": v9.positive_mask(y, pm, is_pisn, 230.0, None),
        }
    return sim


def load_real():
    stars = load_obs_li()
    pred = {}
    with open(PRED_CSV) as f:
        for r in csv.DictReader(f):
            pred[r["star"]] = float(r["f_pisn_q50"])

    li = {f: [] for f in FEATURES}
    li_labels = []
    li_fpisn = []
    for s in stars:
        ca = xfe(s, "CaI")
        mg = xfe(s, "MgI")
        si = xfe(s, "SiI")
        camg = ca - mg if (not np.isnan(ca) and not np.isnan(mg)) else np.nan
        simg = si - mg if (not np.isnan(si) and not np.isnan(mg)) else np.nan
        li["[Ca/Fe]"].append(ca)
        li["[Si/Fe]"].append(si)
        li["[Mg/Fe]"].append(mg)
        li["[Ca/Mg]"].append(camg)
        li["[Si/Mg]"].append(simg)
        li_labels.append(s["label"])
        li_fpisn.append(pred.get(s["label"], np.nan))

    # Xiang secondary (LAMOST VMP subset)
    xiang = {"[Ca/Mg]": None, "[Si/Mg]": None}
    xiang_2d = None
    try:
        import pandas as pd
        cols = ["[Fe/H]", "q_[Fe/H]", "[Ca/Fe]", "q_[Ca/Fe]",
                "[Mg/Fe]", "q_[Mg/Fe]", "[Si/Fe]", "q_[Si/Fe]"]
        df = pd.read_csv(XIANG_CSV, usecols=cols)
        df = df[df["q_[Fe/H]"] == 1]
        df = df[pd.to_numeric(df["[Fe/H]"], errors="coerce") < -2.0]
        camg = df[(df["q_[Ca/Fe]"] == 1) & (df["q_[Mg/Fe]"] == 1)]
        simg = df[(df["q_[Si/Fe]"] == 1) & (df["q_[Mg/Fe]"] == 1)]
        xiang["[Ca/Mg]"] = (pd.to_numeric(camg["[Ca/Fe]"], errors="coerce")
                            - pd.to_numeric(camg["[Mg/Fe]"], errors="coerce")).dropna().to_numpy(float)
        xiang["[Si/Mg]"] = (pd.to_numeric(simg["[Si/Fe]"], errors="coerce")
                            - pd.to_numeric(simg["[Mg/Fe]"], errors="coerce")).dropna().to_numpy(float)
    except Exception as e:  # noqa: BLE001
        print(f"  WARN: Xiang secondary skipped: {e}", flush=True)

    return li, li_labels, li_fpisn, xiang


# ---------------------------------------------------------------- Part A
def part_a(sim, li):
    rows = []
    class_map = [("neg", "neg"), ("pos_all", "pos_all"),
                 ("pos_210-230", "pos_210"), ("pos_>=230", "pos_230")]
    for ds, d in sim.items():
        for feat in FEATURES:
            col = d["pool"][:, d["idx"][feat]]
            for cname, ckey in class_map:
                st = stats(col[d[ckey]])
                if st is None:
                    continue
                rows.append({"dataset": ds, "feature": feat, "class_bin": cname, **st})
    # real Li22
    for feat in FEATURES:
        st = stats(li[feat])
        if st is not None:
            rows.append({"dataset": "Li22", "feature": feat, "class_bin": "real", **st})

    with open(os.path.join(OUT_DIR, "within_class_scatter.csv"), "w", newline="") as f:
        fieldnames = ["dataset", "feature", "class_bin", "n", "mean", "std",
                      "iqr", "p05", "p95", "min", "max"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})
    return rows


# ---------------------------------------------------------------- Part B
def overlap(real_vals, sim_neg, sim_pos):
    real_vals = np.asarray(real_vals, dtype=float)
    real_vals = real_vals[np.isfinite(real_vals)]
    n_real = len(real_vals)
    np05, np95 = np.percentile(sim_neg, [5, 95])
    pp05, pp95 = np.percentile(sim_pos, [5, 95])
    in_neg = (real_vals >= np05) & (real_vals <= np95)
    in_pos = (real_vals >= pp05) & (real_vals <= pp95)
    in_neither = ~(in_neg | in_pos)
    return {
        "n_real": n_real,
        "frac_in_neg": float(in_neg.mean()) if n_real else float("nan"),
        "frac_in_pos": float(in_pos.mean()) if n_real else float("nan"),
        "frac_in_neither": float(in_neither.mean()) if n_real else float("nan"),
        "real_mean": float(np.mean(real_vals)) if n_real else float("nan"),
        "real_std": float(np.std(real_vals)) if n_real else float("nan"),
        "sim_neg_mean": float(np.mean(sim_neg)),
        "sim_neg_std": float(np.std(sim_neg)),
        "sim_pos_mean": float(np.mean(sim_pos)),
        "sim_pos_std": float(np.std(sim_pos)),
        "neg_p05": float(np05), "neg_p95": float(np95),
        "pos_p05": float(pp05), "pos_p95": float(pp95),
    }


def part_b(sim, li, li_labels, li_fpisn, xiang):
    rows = []
    overlay = {}
    real_info = {}
    for feat in RATIO_FEATURES:
        all_vals = np.asarray(li[feat], dtype=float)
        valid = np.isfinite(all_vals)
        real_info[feat] = {
            "li_vals": all_vals[valid],
            "li_fp": np.asarray(li_fpisn, dtype=float)[valid],
            "li_labels": np.asarray(li_labels, dtype=object)[valid],
            "xiang_vals": xiang[feat],
        }

    # 2D intersection (Li22 stars with both [Ca/Mg] and [Si/Mg])
    camg = np.asarray(li["[Ca/Mg]"], dtype=float)
    simg = np.asarray(li["[Si/Mg]"], dtype=float)
    both = np.isfinite(camg) & np.isfinite(simg)
    real_info["2d"] = {
        "camg": camg[both],
        "simg": simg[both],
        "fp": np.asarray(li_fpisn, dtype=float)[both],
        "xiang_camg": xiang["[Ca/Mg]"],
        "xiang_simg": xiang["[Si/Mg]"],
    }

    for ds, d in sim.items():
        overlay[ds] = {}
        for feat in RATIO_FEATURES:
            col = d["pool"][:, d["idx"][feat]]
            sim_neg = col[d["neg"]]
            sim_pos = col[d["pos_all"]]
            ov = overlap(real_info[feat]["li_vals"], sim_neg, sim_pos)
            ov.update({"sim_neg": sim_neg, "sim_pos": sim_pos})
            overlay[ds][feat] = ov
            # Li22 row
            rows.append({"dataset": ds, "feature": feat, "real_source": "Li22", **{
                k: ov[k] for k in ["n_real", "frac_in_neg", "frac_in_pos",
                                   "frac_in_neither", "real_mean", "real_std",
                                   "sim_neg_mean", "sim_neg_std",
                                   "sim_pos_mean", "sim_pos_std"]}})
            # Xiang row (secondary)
            if real_info[feat]["xiang_vals"] is not None:
                xov = overlap(real_info[feat]["xiang_vals"], sim_neg, sim_pos)
                rows.append({"dataset": ds, "feature": feat, "real_source": "Xiang", **{
                    k: xov[k] for k in ["n_real", "frac_in_neg", "frac_in_pos",
                                        "frac_in_neither", "real_mean", "real_std",
                                        "sim_neg_mean", "sim_neg_std",
                                        "sim_pos_mean", "sim_pos_std"]}})

    with open(os.path.join(OUT_DIR, "ood_overlap.csv"), "w", newline="") as f:
        fieldnames = ["dataset", "feature", "real_source", "n_real", "frac_in_neg",
                      "frac_in_pos", "frac_in_neither", "real_mean", "real_std",
                      "sim_neg_mean", "sim_neg_std", "sim_pos_mean", "sim_pos_std"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})
    return rows, overlay, real_info


# ---------------------------------------------------------------- plots
def plot_within_scatter(sim, li):
    ds_list = list(DATASETS)
    n_feat = len(FEATURES)
    n_bars = len(ds_list) + 1  # 4 datasets + Li22 real
    x = np.arange(n_feat)
    width = 0.8 / n_bars
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    floor = 1e-4
    for bi, ds in enumerate(ds_list):
        stds = []
        for feat in FEATURES:
            col = sim[ds]["pool"][:, sim[ds]["idx"][feat]]
            s = stats(col[sim[ds]["neg"]])
            stds.append(max(s["std"], floor) if s else floor)
        ax.bar(x + (bi - n_bars / 2 + 0.5) * width, stds, width, color=colors[bi],
               label=f"{ds} (f=0)")
    real_stds = []
    for feat in FEATURES:
        s = stats(li[feat])
        real_stds.append(max(s["std"], floor) if s else floor)
    ax.bar(x + (n_bars - 1 - n_bars / 2 + 0.5) * width, real_stds, width,
           color="tab:red", hatch="//", label="Li22 real")
    ax.axhline(0.1, color="k", ls="--", lw=1, label="0.1 dex (obs. precision)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(FEATURES)
    ax.set_ylabel("within-class std (dex, log scale)")
    ax.set_title("test_v11: f=0 class scatter vs real Li22 scatter")
    ax.legend(fontsize=8)
    for xi in x:
        for bi, ds in enumerate(ds_list):
            col = sim[ds]["pool"][:, sim[ds]["idx"][FEATURES[xi]]]
            s = stats(col[sim[ds]["neg"]])
            v = s["std"] if s else np.nan
            ax.text(xi + (bi - n_bars / 2 + 0.5) * width, max(v, floor) * 1.3,
                    f"{v:.3g}", ha="center", va="bottom", fontsize=6, rotation=90)
        s = stats(li[FEATURES[xi]])
        v = s["std"] if s else np.nan
        ax.text(xi + (n_bars - 1 - n_bars / 2 + 0.5) * width, max(v, floor) * 1.3,
                f"{v:.3g}", ha="center", va="bottom", fontsize=6, rotation=90, color="tab:red")
    fig.tight_layout()
    fig.savefig(_plot_path("within_class_scatter.png"), dpi=130)
    plt.close(fig)


def plot_overlay(overlay, real_info, ds, feat, tag):
    ov = overlay[ds][feat]
    real_vals = real_info[feat]["li_vals"]
    fp = real_info[feat]["li_fp"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.hist(ov["sim_neg"], bins=32, density=True, alpha=0.5, color="tab:gray",
            label=f"sim CCSN-only (f=0, N={len(ov['sim_neg'])})")
    ax.hist(ov["sim_pos"], bins=32, density=True, alpha=0.5, color="tab:red",
            label=f"sim PISN (f>=0.5, N={len(ov['sim_pos'])})")
    ax.axvline(ov["neg_p05"], color="tab:gray", ls="--", lw=1)
    ax.axvline(ov["neg_p95"], color="tab:gray", ls="--", lw=1)
    ax.axvline(ov["pos_p05"], color="tab:red", ls="--", lw=1)
    ax.axvline(ov["pos_p95"], color="tab:red", ls="--", lw=1)
    # real stars as a jittered strip colored by predicted f_pisn
    n = len(real_vals)
    if n:
        ymin, ymax = ax.get_ylim()
        rng = np.random.default_rng(0)
        jy = ymin + rng.uniform(0, 0.10, size=n) * (ymax - ymin)
        sc = ax.scatter(real_vals, jy, c=fp, cmap="viridis", s=18, alpha=0.85,
                        edgecolor="k", linewidth=0.3, label=f"Li22 (N={n})")
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("predicted f_pisn (q50)")
        ax.set_ylim(ymin, ymax)
    ax.set_xlabel(feat)
    ax.set_ylabel("density")
    ax.set_title(f"test_v11 [{ds}]: {feat} — real Li22 vs sim f=0 / f>=0.5")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(_plot_path(f"real_overlay_{tag}_{ds}.png"), dpi=130)
    plt.close(fig)


def plot_2d(overlay, real_info, ds):
    ov_camg = overlay[ds]["[Ca/Mg]"]
    ov_simg = overlay[ds]["[Si/Mg]"]
    # need sim values on the same samples (both use full neg/pos_all)
    camg_col = overlay[ds]["[Ca/Mg]"]
    fig, ax = plt.subplots(figsize=(7, 6))
    # sim neg/pos [Ca/Mg] vs [Si/Mg] (same masks across features)
    # recompute from pools via stored sim arrays is fine; use overlay sim arrays
    ax.scatter(ov_camg["sim_neg"], ov_simg["sim_neg"], s=5, c="tab:gray", alpha=0.35,
               label="sim CCSN-only")
    ax.scatter(ov_camg["sim_pos"], ov_simg["sim_pos"], s=5, c="tab:red", alpha=0.5,
               label="sim PISN")
    d2 = real_info["2d"]
    ax.scatter(d2["camg"], d2["simg"], s=28, c="tab:blue", alpha=0.9,
               edgecolor="k", linewidth=0.4, label=f"Li22 (N={len(d2['camg'])})")
    if d2["xiang_camg"] is not None and d2["xiang_simg"] is not None:
        n = min(len(d2["xiang_camg"]), len(d2["xiang_simg"]))
        ax.scatter(d2["xiang_camg"][:n], d2["xiang_simg"][:n], s=22, c="tab:green",
                   alpha=0.7, edgecolor="k", linewidth=0.3, label=f"Xiang (N={n})")
    ax.set_xlabel("[Ca/Mg] (dex)")
    ax.set_ylabel("[Si/Mg] (dex)")
    ax.axhline(0, color="k", ls=":", lw=0.6)
    ax.axvline(0, color="k", ls=":", lw=0.6)
    ax.set_title(f"test_v11 [{ds}]: [Ca/Mg] vs [Si/Mg] — sim vs real")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(_plot_path(f"ood_2d_{ds}.png"), dpi=130)
    plt.close(fig)


def plot_summary(partB_rows):
    li_rows = [r for r in partB_rows if r["real_source"] == "Li22"]
    ds_list = list(DATASETS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, feat in zip(axes, RATIO_FEATURES):
        fr = [r for r in li_rows if r["feature"] == feat]
        by_ds = {r["dataset"]: r for r in fr}
        x = np.arange(len(ds_list))
        w = 0.28
        ax.bar(x - w, [by_ds[d]["frac_in_neg"] for d in ds_list], w,
               label="in neg [p05,p95]", color="tab:gray")
        ax.bar(x, [by_ds[d]["frac_in_pos"] for d in ds_list], w,
               label="in pos [p05,p95]", color="tab:red")
        ax.bar(x + w, [by_ds[d]["frac_in_neither"] for d in ds_list], w,
               label="in NEITHER (gap/OOD)", color="tab:orange")
        ax.set_xticks(x)
        ax.set_xticklabels(ds_list)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("fraction of real Li22 stars")
        ax.set_title(f"{feat}")
        ax.legend(fontsize=8)
    fig.suptitle("test_v11: where do real Li22 stars fall vs simulated envelopes?",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(_plot_path("summary_ood.png"), dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- report
def write_report(partA_rows, partB_rows):
    lines = [
        "test_v11: within-class scatter + real-star OOD check",
        "Classes: neg = f_pisn==0; pos = f_pisn>=0.5 (overall and per mass bin).",
        f"Near-point flag: f=0 std < {NEAR_POINT_THRESH} dex.",
        "",
        "== PART A: within-class scatter ==",
    ]
    # focused table for ratio features
    for ds in ["4_3", "5", "L", "N", "Li22"]:
        for feat in RATIO_FEATURES:
            for r in partA_rows:
                if r["dataset"] == ds and r["feature"] == feat:
                    tag = "NEAR-POINT" if (r["class_bin"] == "neg" and ds != "Li22"
                                           and r["std"] < NEAR_POINT_THRESH) else ""
                    lines.append(
                        f"  {ds:<5} {feat:<9} {r['class_bin']:<11} n={r['n']:>4} "
                        f"mean={r['mean']:+.3f} std={r['std']:.4f} "
                        f"IQR={r['iqr']:.4f} p05={r['p05']:+.3f} p95={r['p95']:+.3f} "
                        f"[{r['min']:+.3f},{r['max']:+.3f}] {tag}")
    lines.append("")

    # near-point flag summary
    lines.append("== near-point check (f=0 std vs 0.1 dex) ==")
    for ds in ["4_3", "5", "L", "N"]:
        for feat in RATIO_FEATURES:
            for r in partA_rows:
                if r["dataset"] == ds and r["feature"] == feat and r["class_bin"] == "neg":
                    flag = "NEAR-POINT" if r["std"] < NEAR_POINT_THRESH else "resolved"
                    lines.append(f"  {ds} {feat}: f=0 std = {r['std']:.4f} dex -> {flag}")
    lines.append("")

    lines.append("== PART B: real-star overlap (frac inside sim envelopes) ==")
    for ds in ["4_3", "5", "L", "N"]:
        for feat in RATIO_FEATURES:
            for r in partB_rows:
                if r["dataset"] == ds and r["feature"] == feat:
                    lines.append(
                        f"  {ds} {feat} {r['real_source']:<6} n={r['n_real']:>4} "
                        f"neg={r['frac_in_neg']:.2f} pos={r['frac_in_pos']:.2f} "
                        f"neither={r['frac_in_neither']:.2f} "
                        f"real mean/std={r['real_mean']:+.2f}/{r['real_std']:.2f}")
    lines.append("")
    with open(os.path.join(OUT_DIR, "test_v11_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------- main
def main():
    _style()
    sim = load_sim()
    li, li_labels, li_fpisn, xiang = load_real()

    partA_rows = part_a(sim, li)
    partB_rows, overlay, real_info = part_b(sim, li, li_labels, li_fpisn, xiang)

    plot_within_scatter(sim, li)
    for ds in DATASETS:
        plot_overlay(overlay, real_info, ds, "[Ca/Mg]", "camg")
        plot_overlay(overlay, real_info, ds, "[Si/Mg]", "simg")
        plot_2d(overlay, real_info, ds)
    plot_summary(partB_rows)

    write_report(partA_rows, partB_rows)

    # print near-point flags + key overlap numbers to stdout
    print("\nNear-point check (f=0 std):")
    for ds in ["4_3", "5", "L", "N"]:
        for feat in RATIO_FEATURES:
            for r in partA_rows:
                if r["dataset"] == ds and r["feature"] == feat and r["class_bin"] == "neg":
                    flag = "NEAR-POINT" if r["std"] < NEAR_POINT_THRESH else "resolved"
                    print(f"  {ds} {feat}: std={r['std']:.4f} -> {flag}")

    print("\nLi22 overlap (frac in neg / pos / neither):")
    for ds in ["4_3", "5", "L", "N"]:
        for feat in RATIO_FEATURES:
            for r in partB_rows:
                if r["dataset"] == ds and r["feature"] == feat and r["real_source"] == "Li22":
                    print(f"  {ds} {feat}: neg={r['frac_in_neg']:.2f} "
                          f"pos={r['frac_in_pos']:.2f} neither={r['frac_in_neither']:.2f} "
                          f"(n={r['n_real']})")

    files = sorted(os.listdir(OUT_DIR))
    print("\nProduced files (size bytes):")
    for fn in files:
        p = os.path.join(OUT_DIR, fn)
        print(f"  {fn:<40} {os.path.getsize(p):>8}")
    # list subfolder pngs
    sub = []
    for folder in sorted(_PLOT_SUBDIRS.values()):
        fp = os.path.join(OUT_DIR, folder)
        if os.path.isdir(fp):
            pngs = sorted(os.listdir(fp))
            for p in pngs:
                sub.append((folder, p))
    print(f"\nTotal: {len(files)} root files + {len(sub)} pngs in subfolders")


if __name__ == "__main__":
    main()
