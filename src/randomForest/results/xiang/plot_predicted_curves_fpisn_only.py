"""Forward-model abundance curves vs. observed — f_pisn-UNCERTAINTY-ONLY band.

Same as plot_predicted_curves.py, EXCEPT the nuisance parameters (PISN mass,
f_ratio, tpop2) are FIXED at their geometric-median values (mass=210 Msun,
f_ratio=3.16e-3, tpop2=7.5e6) and ONLY f_pisn is Monte-Carlo sampled (from the
QRF predictive quantiles).  This makes the blue band reflect purely "how sure
is the model about this star's f_pisn" rather than the (shared, huge) PISN-mass
prior, so robust stars get a thin band and artifact stars get a wide smear.

The existing xiang_curve_<Name>.png plots are left untouched; this writes new
files xiang_curve_fpisn_<Name>.png.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --- repo-root / path setup -------------------------------------------------
def _find_root(start: str) -> str:
    d = os.path.abspath(start)
    for _ in range(12):
        if os.path.exists(os.path.join(d, "params.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    raise RuntimeError("Could not locate repo root")


ROOT = _find_root(os.getcwd())
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.load_data import load_hw2002, load_limongi18
from src.salvadori_funcs import salvadori_yields, salvadori_combined_abundratio
from src.utils import build_pisn_interpolator
from src.yield_sources import get_source

XIANG_DIR = os.path.join(ROOT, "src", "randomForest", "results", "xiang")
PRED_CSV = os.path.join(XIANG_DIR, "xiang_predictions.csv")
CAT_CSV = os.path.join(XIANG_DIR, "xiang_catalog.csv")

# Model elements (same order as dataset._abundance_elements).
FULL_ELEMENTS = ["C", "O", "Na", "Mg", "Al", "Si", "Ca", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"]
X_AXIS = [e for e in FULL_ELEMENTS if e != "Fe"]  # 13 elements (Fe is reference)

STARS = ["S0345958+580852", "S0300611+601311", "S1410926+262321", "S3531920+260443"]

N_MC = 200  # f_pisn Monte-Carlo samples per star (nuisance params fixed)

# Fixed (geometric-median) nuisance params — same values the reference grid uses.
FIXED_MASS = 210.0
FIXED_F_RATIO = 3.16e-3
FIXED_TPOP2 = 7.5e6


def load_yields():
    hw = load_hw2002()
    hw = hw[1:]
    sal_pisn = salvadori_yields(hw)
    pisn_interp = build_pisn_interpolator(sal_pisn)

    sn_entries = [e for e in load_limongi18() if e["params"]["velocity"] == 0]
    salv_sn = salvadori_yields(sn_entries)
    sn_src = get_source("Limongi18")
    return pisn_interp, sn_entries, salv_sn, sn_src


def forward_xfe(pisn_interp, sn_entries, salv_sn, sn_src,
                f_pisn, pisn_mass, f_ratio, tpop2):
    """[X/Fe] for every model element, using the exact training-time formula."""
    pisn_entry = pisn_interp(pisn_mass)
    out = {}
    for elem in FULL_ELEMENTS:
        if elem == "Fe":
            out[elem] = 0.0
            continue
        out[elem] = salvadori_combined_abundratio(
            elem, elem, "Fe", "Fe",
            pisn_data=pisn_entry,
            sn_data=sn_entries,
            salv_sn_data=salv_sn,
            auto_sn=False,
            single_sn=False,
            sn_input=sn_src,
            f_pisn=f_pisn,
            f_ratio=f_ratio,
            tpop2=tpop2,
        )
    return out


def sample_fpisn(f_qs, rng):
    """Draw f_pisn from the QRF predictive quantiles (piecewise-linear CDF)."""
    prob = np.array([0.05, 0.16, 0.5, 0.84, 0.95])
    val = np.array([f_qs[p] for p in prob])
    u = rng.uniform(0.0, 1.0)
    return float(np.clip(np.interp(u, prob, val), 0.0, 1.0))


def observed_for_star(cat, name):
    """dict elem -> (xfe, err) for the measured model elements; NaN if absent."""
    mask = cat["Name"].astype(str) == name
    if not mask.any():
        return None
    row = cat[mask].iloc[0]
    obs = {}
    for elem in FULL_ELEMENTS:
        if elem == "Fe":
            obs[elem] = (0.0, 0.0)
            continue
        col, ecol, qcol = f"[{elem}/Fe]", f"e_[{elem}/Fe]", f"q_[{elem}/Fe]"
        if col not in cat.columns:
            obs[elem] = (np.nan, np.nan)
            continue
        val = pd.to_numeric(row[col], errors="coerce")
        e = pd.to_numeric(row[ecol], errors="coerce") if ecol in cat.columns else np.nan
        q = pd.to_numeric(row[qcol], errors="coerce") if qcol in cat.columns else np.nan
        if pd.isna(val) or (not pd.isna(q) and q != 1.0):
            obs[elem] = (np.nan, np.nan)
        else:
            obs[elem] = (float(val), float(e) if not pd.isna(e) else 0.0)
    return obs


def main():
    pisn_interp, sn_entries, salv_sn, sn_src = load_yields()
    pred = pd.read_csv(PRED_CSV, dtype={"star": str})
    cat = pd.read_csv(CAT_CSV, dtype={"Name": str})

    for name in STARS:
        p = pred[pred["star"] == name]
        if p.empty:
            print(f"!! {name}: not found in predictions CSV")
            continue
        f_q50 = float(p.iloc[0]["f_pisn_q50"])
        f_q16 = float(p.iloc[0]["f_pisn_q16"])
        f_q84 = float(p.iloc[0]["f_pisn_q84"])
        f_q05 = float(p.iloc[0]["f_pisn_q05"])
        f_q95 = float(p.iloc[0]["f_pisn_q95"])
        f_qs = {0.05: f_q05, 0.16: f_q16, 0.5: f_q50, 0.84: f_q84, 0.95: f_q95}

        obs = observed_for_star(cat, name)
        if obs is None:
            print(f"!! {name}: not found in catalogue CSV")
            continue
        feh = pd.to_numeric(cat[cat["Name"] == name].iloc[0]["[Fe/H]"], errors="coerce")
        teff = pd.to_numeric(cat[cat["Name"] == name].iloc[0]["Teff"], errors="coerce")

        # Monte-Carlo over f_pisn ONLY; nuisance params fixed at median so the
        # band isolates the model's f_pisn uncertainty.
        rng = np.random.default_rng(42)
        curves = np.full((N_MC, len(FULL_ELEMENTS)), np.nan)
        for k in range(N_MC):
            f_k = sample_fpisn(f_qs, rng)
            fx = forward_xfe(pisn_interp, sn_entries, salv_sn, sn_src,
                             f_k, FIXED_MASS, FIXED_F_RATIO, FIXED_TPOP2)
            for j, e in enumerate(FULL_ELEMENTS):
                curves[k, j] = fx[e]

        med = np.nanmedian(curves, axis=0)
        lo = np.nanpercentile(curves, 16, axis=0)
        hi = np.nanpercentile(curves, 84, axis=0)

        xaxis_idx = [FULL_ELEMENTS.index(e) for e in X_AXIS]
        med_p, lo_p, hi_p = med[xaxis_idx], lo[xaxis_idx], hi[xaxis_idx]

        obs_x = np.array([obs[e][0] for e in X_AXIS])
        obs_e = np.array([obs[e][1] for e in X_AXIS])
        finite = np.isfinite(obs_x)

        # --- plot ---
        xpos = np.arange(len(X_AXIS))
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.fill_between(xpos, lo_p, hi_p, color="tab:blue", alpha=0.2,
                        label="predicted band (f_pisn only, 16-84%)")
        ax.plot(xpos, med_p, "-o", ms=4, color="tab:blue", lw=1.4,
                label="predicted [X/Fe] (median)")
        ax.errorbar(xpos[finite], obs_x[finite], yerr=obs_e[finite],
                    fmt="o", ms=6, color="tab:red", capsize=3, lw=1.0,
                    label="observed (Xiang)")
        if (~finite).any():
            ax.scatter(xpos[~finite], np.zeros((~finite).sum()),
                       marker="x", color="grey", s=30, zorder=3, alpha=0.6)
            ax.scatter(xpos[~finite], [lo_p[~finite]], marker="x", color="grey", s=30, zorder=3)

        ax.axhline(0, ls="--", color="k", lw=0.8, alpha=0.7)
        ax.set_xticks(xpos)
        ax.set_xticklabels(X_AXIS)
        ax.set_xlabel("Element")
        ax.set_ylabel("[X/Fe]")
        ax.set_ylim(min(-2.5, np.nanmin(lo_p) - 0.3), max(2.5, np.nanmax(hi_p) + 0.3))
        ax.set_title(
            f"{name}   f_pisn = {f_q50:.2f} [{f_q16:.2f}, {f_q84:.2f}]   "
            f"[Fe/H] = {feh:.2f}   Teff = {teff:.0f} K   (f_pisn-uncertainty band)"
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        out = os.path.join(XIANG_DIR, f"xiang_curve_fpisn_{name}.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"saved {out}")

        print(f"  {name}: f_pisn={f_q50:.3f} ({f_q16:.3f}-{f_q84:.3f})")
        for j, e in enumerate(X_AXIS):
            o = f"{obs_x[j]:+.3f}" if finite[j] else "  nan"
            print(f"    {e:>3}  pred={med_p[j]:+.3f} (lo={lo_p[j]:+.3f} hi={hi_p[j]:+.3f})  obs={o}")


if __name__ == "__main__":
    main()
