"""Scatter analysis of the most important abundance ratios vs f_pisn.

For each dataset (Nomoto13 and Limongi18) rank the 91 [X/Y] ratio features by
permutation importance, then scatter the top ratios against the f_pisn target.
A contrast panel of iron-peak-element ratios is produced alongside to show why
those carry little signal (their [X/Fe] barely moves with f_pisn).

Outputs (per dataset):
    <out>/ratio_scatter_top.png       top-N ratios by permutation importance
    <out>/ratio_scatter_ironpeak.png  iron-peak ratio contrast panel

Run:  .venv/bin/python src/randomForest/diagnostic/ratio_scatter_analysis.py
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

from src.randomForest.train_rf import load_dataset  # noqa: E402

DATA_DIR = os.path.join(ROOT, "src", "randomForest", "data")
DIAG_DIR = os.path.join(ROOT, "src", "randomForest", "diagnostic")

# 14 elements used to build the [X/Y] training features (no Ti, no Sc).
ELEMENTS = ["C", "O", "Na", "Mg", "Al", "Si", "Ca", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"]
ALPHA = {"O", "Mg", "Si", "Ca"}
ODD_Z = {"Na", "Al"}  # odd-Z elements (odd-even effect)
IRON_PEAK = {"Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"}

DATASETS = {
    "fi_nomoto_v5": {
        "cache": os.path.join(DATA_DIR, "dataset.npz"),
        "label": "Nomoto13",
    },
    "fi_limongi_v4_3": {
        "cache": os.path.join(DATA_DIR, "dataset_TakaLimon_realisticNoise.npz"),
        "label": "Limongi18",
    },
}

TOP_N = 12

# Fixed contrast set: iron-peak-element ratios (Cr,Mn,Fe,Co,Ni,Cu,Zn) that should
# be near-flat in f_pisn (these elements are SNII-dominated, not PISN tracers).
IRON_PEAK_RATIOS = [
    "[Cr/Fe]", "[Mn/Fe]", "[Co/Fe]", "[Ni/Fe]", "[Cu/Fe]", "[Zn/Fe]",
    "[Cr/Ni]", "[Mn/Co]", "[Co/Ni]", "[Cr/Mn]", "[Ni/Zn]", "[Cu/Zn]",
]


def _parse_pair(name: str):
    """'[X/Y]' -> (X, Y)."""
    inner = name.strip("[]")
    x, y = inner.split("/")
    return x, y


def _rank_features(X, y):
    """Permutation importance ranking on a held-out split."""
    imp = SimpleImputer(strategy="median")
    Xc = imp.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        Xc, y, test_size=0.25, random_state=12345
    )
    rf = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    r = permutation_importance(
        rf, X_te, y_te, n_repeats=5, random_state=0, n_jobs=-1
    )
    return rf, r.importances_mean, r.importances_std


def _scatter_panel(ax, x, y, title, r_value, imp_value):
    """Scatter ratio (x) vs f_pisn (y) with a rolling-mean trend line."""
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    ax.scatter(x, y, s=5, alpha=0.25, c="#1f77b4", linewidths=0, rasterized=True)

    # rolling mean of y in x
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    win = max(20, len(xs) // 40)
    kernel = np.ones(win) / win
    ym = np.convolve(ys, kernel, mode="same")
    ax.plot(xs, ym, color="crimson", lw=1.6)

    ax.set_title(title, fontsize=9)
    ax.text(
        0.03, 0.97, f"r = {r_value:+.2f}\nimp = {imp_value:.3f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=7,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.9),
    )
    ax.set_xlabel("ratio [dex]")
    ax.set_ylabel("f$_{PISN}$")


def _figure(dataset, key):
    cache = dataset["cache"]
    label = dataset["label"]
    X, y, diag = load_dataset(cache)
    names = list(diag["feature_names"])
    print(f"\n=== {label} ===  X={X.shape}  NaN={int(np.isnan(X).sum())}")

    # drop rows whose target is NaN (shouldn't happen) and impute features
    rf, imp_mean, imp_std = _rank_features(X, y)

    # Pearson r of each feature vs y (on imputed, finite rows)
    imp_med = SimpleImputer(strategy="median")
    Xc = imp_med.fit_transform(X)
    rvals = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        ok = np.isfinite(Xc[:, j])
        if ok.sum() > 2:
            rvals[j] = np.corrcoef(Xc[ok, j], y[ok])[0, 1]

    # rank by permutation importance
    order = np.argsort(imp_mean)[::-1]
    top_idx = order[:TOP_N]

    print(f"top {TOP_N} by permutation importance:")
    for j in top_idx:
        x, yy = _parse_pair(names[j])
        cls = (
            "ALPHA" if (x in ALPHA and yy in ALPHA) else
            "ODD-Z" if (x in ODD_Z or yy in ODD_Z) else
            "iron" if (x in IRON_PEAK and yy in IRON_PEAK) else "mixed"
        )
        print(f"  {names[j]:>8}  imp={imp_mean[j]:.4f}  r={rvals[j]:+.3f}  [{cls}]")

    out_dir = os.path.join(DIAG_DIR, key)
    os.makedirs(out_dir, exist_ok=True)

    # ---- top ratios scatter ----
    ncols = 4
    nrows = int(np.ceil(TOP_N / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3.4 * nrows), squeeze=False
    )
    fig.suptitle(f"Most important abundance ratios vs f$_{{PISN}}$ — {label}",
                 fontsize=13)
    for k, j in enumerate(top_idx):
        ax = axes[k // ncols][k % ncols]
        _scatter_panel(ax, Xc[:, j], y, names[j], rvals[j], imp_mean[j])
    for k in range(TOP_N, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(out_dir, "ratio_scatter_top.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"  saved {p}")

    # ---- iron-peak contrast scatter ----
    iron_idx = [names.index(n) for n in IRON_PEAK_RATIOS if n in names]
    ncols2 = 4
    nrows2 = int(np.ceil(len(iron_idx) / ncols2))
    fig2, axes2 = plt.subplots(
        nrows2, ncols2, figsize=(4 * ncols2, 3.4 * nrows2), squeeze=False
    )
    fig2.suptitle(
        f"Iron-peak ratios (weak f$_{{PISN}}$ tracers) — {label}", fontsize=13
    )
    for k, j in enumerate(iron_idx):
        ax = axes2[k // ncols2][k % ncols2]
        _scatter_panel(ax, Xc[:, j], y, names[j], rvals[j], imp_mean[j])
    for k in range(len(iron_idx), nrows2 * ncols2):
        axes2[k // ncols2][k % ncols2].axis("off")
    fig2.tight_layout(rect=[0, 0, 1, 0.97])
    p2 = os.path.join(out_dir, "ratio_scatter_ironpeak.png")
    fig2.savefig(p2, dpi=130)
    plt.close(fig2)
    print(f"  saved {p2}")

    # element-level importance attribution (for the alpha/odd-even narrative)
    el_imp = {}
    for j in range(X.shape[1]):
        x, yy = _parse_pair(names[j])
        el_imp[x] = el_imp.get(x, 0.0) + imp_mean[j]
        el_imp[yy] = el_imp.get(yy, 0.0) + imp_mean[j]
    el_order = sorted(el_imp.items(), key=lambda kv: kv[1], reverse=True)
    print("element-level importance (sum over ratios):")
    for el, v in el_order:
        tag = "alpha" if el in ALPHA else "odd-Z" if el in ODD_Z else "iron" if el in IRON_PEAK else ""
        print(f"  {el:>3}  {v:.4f}  {tag}")


def main():
    for key, dataset in DATASETS.items():
        _figure(dataset, key)


if __name__ == "__main__":
    main()
