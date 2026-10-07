"""
plot_mgca_li.py: [Mg/Ca] diagnostics for the Li+22 catalogue (test_v4_3).

Reads the existing obs_li_all.csv (f_pisn predictions from the test_v4_3 QRF run)
and the Li+22 stars (load_obs_li), computes the observed [Mg/Ca] = [Mg/Fe] -
[Ca/Fe] wherever BOTH MgI and CaI are reliably measured, and produces:

  1. mgca_hist.png                   - spread of [Mg/Ca] across stars with both
  2. mgca_vs_feh.png                 - [Mg/Ca] vs [Fe/H], colored by f_pisn
  3. mgca_vs_fpisn.png               - [Mg/Ca] vs predicted f_pisn, colored by FeH
  4. mgca_complete_vs_incomplete.png - complete (both Mg&Ca) vs incomplete (missing one)

"complete" here = has BOTH Mg and Ca measured (so [Mg/Ca] is defined);
"incomplete" = missing at least one of Mg/Ca (no [Mg/Ca] value).
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = "/home/ardo/mlem/Vaults/Thesis/Repl"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.load_data import load_obs_li  # noqa: E402

# Test directory from argv (default test_v4_3); e.g. `python plot_mgca_li.py test_v5_0`
TEST_DIR = sys.argv[1] if len(sys.argv) > 1 else "test_v4_3"
CSV = os.path.join(ROOT, "src", "randomForest", "results", TEST_DIR, "obs_li_all.csv")
OUT = os.path.join(ROOT, "src", "randomForest", "results", TEST_DIR)
os.makedirs(OUT, exist_ok=True)
print(f"[plot_mgca_li] test dir = {TEST_DIR}")


def xfe(star, ion):
    """[X/Fe] for a Li+22 star; upper-limit or unmeasured -> NaN."""
    rec = star["abundances"].get(ion)
    if not rec:
        return np.nan
    v = rec.get("XFe")
    l = rec.get("l_XFe")
    if v is None or l == "<":
        return np.nan
    return float(v)


def main():
    stars = load_obs_li()
    label_to_star = {s["label"]: s for s in stars}
    df = pd.read_csv(CSV)
    print(f"Li stars loaded: {len(stars)}; prediction rows: {len(df)}")

    rows = []
    for _, r in df.iterrows():
        s = label_to_star.get(r["star"])
        if s is None:
            print(f"  WARN: {r['star']} not in loader")
            continue
        mg = xfe(s, "MgI")
        ca = xfe(s, "CaI")
        has_both = (not np.isnan(mg)) and (not np.isnan(ca))
        rows.append({
            "star": r["star"], "FeH": r["FeH"],
            "MgFe": mg, "CaFe": ca,
            "MgCa": (mg - ca) if has_both else np.nan,
            "has_both": has_both,
            "is_complete": int(r["is_complete"]),
            "f_std": r["f_pisn_std_q50"], "f_mask": r["f_pisn_mask_q50"],
        })
    d = pd.DataFrame(rows)
    comp = d[d["has_both"]]
    incomp = d[~d["has_both"]]
    print(f"complete (both Mg & Ca): {len(comp)}  incomplete: {len(incomp)}")
    print(f"[Mg/Ca]  mean={comp['MgCa'].mean():+.3f}  median={comp['MgCa'].median():+.3f}  "
          f"std={comp['MgCa'].std():.3f}  min={comp['MgCa'].min():+.3f}  max={comp['MgCa'].max():+.3f}")
    rho_fe, pv_fe = spearmanr(comp["MgCa"], comp["FeH"])
    rho_f, pv_f = spearmanr(comp["MgCa"], comp["f_std"])
    print(f"Spearman([Mg/Ca], FeH) = {rho_fe:+.3f} (p={pv_fe:.2e})")
    print(f"Spearman([Mg/Ca], f_pisn) = {rho_f:+.3f} (p={pv_f:.2e})")

    # is_complete (all 11 elements measured) — the meaningful split
    comp11 = comp[comp["is_complete"] == 1]
    comp_not11 = comp[comp["is_complete"] == 0]
    print(f"among both-Mg&Ca: is_complete(11 el)={len(comp11)}  not={len(comp_not11)}")
    print(f"  [Mg/Ca] complete: mean={comp11['MgCa'].mean():+.3f}  "
          f"median={comp11['MgCa'].median():+.3f}  std={comp11['MgCa'].std():.3f}")
    print(f"  [Mg/Ca] incomplete: mean={comp_not11['MgCa'].mean():+.3f}  "
          f"median={comp_not11['MgCa'].median():+.3f}  std={comp_not11['MgCa'].std():.3f}")

    # ---- stars with mask-aware f_pisn > 0.5 ----
    high = d[d["f_mask"] > 0.5].sort_values("f_mask", ascending=False)
    print(f"\nstars with f_pisn(mask q50) > 0.5: {len(high)}")
    print(f"  {'star':<14}{'FeH':>8}{'Mg/Ca':>9}{'f_mask':>9}{'f_std':>9}")
    high_lines = [f"stars with f_pisn(mask q50) > 0.5: {len(high)}",
                  f"{'star':<14}{'FeH':>8}{'Mg/Ca':>9}{'f_mask':>9}{'f_std':>9}"]
    for _, r in high.iterrows():
        mgca = r["MgCa"]
        mgca_s = "   --" if np.isnan(mgca) else f"{mgca:+.3f}"
        line = f"{r['star']:<14}{r['FeH']:>8.3f}{mgca_s:>9}{r['f_mask']:>9.3f}{r['f_std']:>9.3f}"
        high_lines.append(line)
        print("  " + line)
    out_txt = os.path.join(OUT, "mgca_high_fpisn_mask.txt")
    with open(out_txt, "w") as f:
        f.write("\n".join(high_lines) + "\n")
    print(f"saved {out_txt}")

    # ---- 1. histogram of [Mg/Ca] (stars with both) ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(comp["MgCa"], bins=20, color="tab:blue", alpha=0.8, edgecolor="white")
    ax.axvline(0.0, color="k", ls="--", lw=1)
    ax.axvline(comp["MgCa"].median(), color="tab:red", ls="-", lw=1.2,
               label=f"median={comp['MgCa'].median():+.2f}")
    ax.set_xlabel("[Mg/Ca] (dex)")
    ax.set_ylabel("N stars")
    ax.set_title(f"Li+22 [Mg/Ca] — stars with both Mg & Ca (N={len(comp)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mgca_hist.png"), dpi=130)
    plt.close(fig)
    print("saved mgca_hist.png")

    # ---- 2. [Mg/Ca] vs [Fe/H], colored by f_pisn ----
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sc = ax.scatter(comp["FeH"], comp["MgCa"], c=comp["f_std"], cmap="viridis",
                    s=40, alpha=0.85, edgecolor="k", linewidth=0.4)
    ax.axhline(0.0, color="k", ls="--", lw=1)
    ax.set_xlabel("[Fe/H]")
    ax.set_ylabel("[Mg/Ca] (dex)")
    ax.set_title(f"Li+22 [Mg/Ca] vs [Fe/H] (N={len(comp)}, color = f_pisn q50)")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("predicted f_pisn (std, q50)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mgca_vs_feh.png"), dpi=130)
    plt.close(fig)
    print("saved mgca_vs_feh.png")

    # ---- 3. [Mg/Ca] vs f_pisn, colored by FeH ----
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sc = ax.scatter(comp["f_std"], comp["MgCa"], c=comp["FeH"], cmap="plasma",
                    s=40, alpha=0.85, edgecolor="k", linewidth=0.4)
    ax.axhline(0.0, color="k", ls="--", lw=1)
    ax.set_xlabel("predicted f_pisn (std, q50)")
    ax.set_ylabel("[Mg/Ca] (dex)")
    ax.set_title(f"Li+22 [Mg/Ca] vs f_pisn (N={len(comp)}, color = [Fe/H])")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("[Fe/H]")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mgca_vs_fpisn.png"), dpi=130)
    plt.close(fig)
    print("saved mgca_vs_fpisn.png")

    # ---- 4. complete vs incomplete ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.hist(comp["MgCa"], bins=20, color="tab:blue", alpha=0.8, edgecolor="white",
            label=f"complete (both, N={len(comp)})")
    ax.axvline(0.0, color="k", ls="--", lw=1)
    ax.set_xlabel("[Mg/Ca] (dex)")
    ax.set_ylabel("N stars")
    ax.set_title("[Mg/Ca] — complete only (incomplete has no [Mg/Ca])")
    ax.legend()

    ax = axes[1]
    ax.boxplot([comp["f_std"], incomp["f_std"]],
               tick_labels=[f"complete\n(N={len(comp)})", f"incomplete\n(N={len(incomp)})"])
    ax.set_ylabel("predicted f_pisn (std, q50)")
    ax.set_title("f_pisn of complete vs incomplete")
    ax.grid(axis="y", alpha=0.3)

    # also compare [Fe/H]
    ax2 = ax.twinx()
    ax2.boxplot([comp["FeH"], incomp["FeH"]], positions=[1.35, 2.35], widths=0.2,
                boxprops=dict(color="tab:orange"), whiskerprops=dict(color="tab:orange"),
                capprops=dict(color="tab:orange"), medianprops=dict(color="tab:orange"),
                flierprops=dict(markeredgecolor="tab:orange"))
    ax2.set_ylabel("[Fe/H] (orange)")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mgca_complete_vs_incomplete.png"), dpi=130)
    plt.close(fig)
    print("saved mgca_complete_vs_incomplete.png")

    # ---- 5. [Mg/Ca] split by is_complete (all 11 elements measured) ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    lo = min(comp["MgCa"].min(), -1.2)
    hi = max(comp["MgCa"].max(), 1.2)
    bins = np.linspace(lo, hi, 24)
    ax.hist(comp_not11["MgCa"], bins=bins, color="tab:gray", alpha=0.6,
            edgecolor="white", label=f"incomplete (N={len(comp_not11)})")
    ax.hist(comp11["MgCa"], bins=bins, color="tab:blue", alpha=0.7,
            edgecolor="white", label=f"complete 11 el (N={len(comp11)})")
    ax.axvline(0.0, color="k", ls="--", lw=1)
    ax.set_xlabel("[Mg/Ca] (dex)")
    ax.set_ylabel("N stars")
    ax.set_title("[Mg/Ca] split by is_complete (all 11 elements)")
    ax.legend()

    ax = axes[1]
    ax.boxplot([comp11["MgCa"], comp_not11["MgCa"]],
               tick_labels=[f"complete\n(N={len(comp11)})",
                            f"incomplete\n(N={len(comp_not11)})"])
    ax.axhline(0.0, color="k", ls="--", lw=1)
    ax.set_ylabel("[Mg/Ca] (dex)")
    ax.set_title("[Mg/Ca] complete vs incomplete")
    ax.grid(axis="y", alpha=0.3)

    # f_pisn comparison on twin axis
    ax2 = ax.twinx()
    ax2.boxplot([comp11["f_std"], comp_not11["f_std"]],
                positions=[1.35, 2.35], widths=0.2,
                boxprops=dict(color="tab:orange"), whiskerprops=dict(color="tab:orange"),
                capprops=dict(color="tab:orange"), medianprops=dict(color="tab:orange"),
                flierprops=dict(markeredgecolor="tab:orange"))
    ax2.set_ylabel("predicted f_pisn (std, q50) [orange]")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mgca_by_iscomplete.png"), dpi=130)
    plt.close(fig)
    print("saved mgca_by_iscomplete.png")

    print("done.")


if __name__ == "__main__":
    main()
