"""Yield plots for Li+22 stars with mask-aware f_pisn > 0.5 (test_v5_0).

For each high-f_pisn candidate this reproduces the GENERATIVE model used to
build the training data (Takahashi PISN + Nomoto13 SNII, the exact
`salvadori_combined_abundratio` path in dataset.py) at the star's PREDICTED
f_pisn, drawing the nuisance parameters (PISN mass, f_ratio, tpop2) from the
same priors the training set used.  The resulting [X/Fe] pattern (median +
16-84% band) is overlaid with the star's ACTUAL measured [X/Fe] points (with
sigma_Total error bars).  Non-detected elements are shown as crossed-out (x)
markers; upper limits are crossed out with a downward arrow.

Same as the test_v4_3 yield plots, but driven by the test_v5_0 predictions
(Takahashi PISN + Nomoto13 SNII, clean noiseless dataset), with a threshold
of 0.5 on the mask-aware f_pisn q50.  Nomoto13's grid tops out at 40 Msun,
so the tpop2 prior is shifted to (5.5e6, 17.4e6) to match the training set
(dataset.py TPOP2_RANGE_BY_SOURCE) and keep the turnoff mass inside the grid.

Output: one PNG per candidate in src/randomForest/results/test_v5_0/.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import loguniform


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

from params import filter_elements
from src.load_data import load_takahashi, load_nomoto13, load_obs_li
from src.salvadori_funcs import salvadori_yields, salvadori_combined_abundratio
from src.utils import build_pisn_interpolator, _ion_to_element
from src.yield_sources import get_source

RESULTS_DIR = os.path.join(ROOT, "src", "randomForest", "results", "test_v5_0")
PRED_CSV = os.path.join(RESULTS_DIR, "obs_li_all.csv")

ELEMENTS = list(filter_elements)          # 25 elements (Fe included; Fe = 0)
X_AXIS = [e for e in ELEMENTS if e != "Fe"]

PISN_MASS_RANGE = (150.0, 270.0)
F_RATIO_RANGE = (1e-4, 1e-1)
TPOP2_RANGE = (5.5e6, 17.4e6)   # Nomoto13 prior (grid tops out at 40 Msun)
N_MC = 120
FPISN_THRESH = 0.5


def load_yields():
    taka = load_takahashi()["NR"]
    sal_pisn = salvadori_yields(taka)
    pisn_interp = build_pisn_interpolator(sal_pisn)
    sn_entries = [e for e in load_nomoto13() if e["params"]["energy"] == 1.0]
    salv_sn = salvadori_yields(sn_entries)
    sn_src = get_source("Nomoto13")
    return pisn_interp, sn_entries, salv_sn, sn_src


def forward_xfe(pisn_interp, sn_entries, salv_sn, sn_src,
                f_pisn, pisn_mass, f_ratio, tpop2, elements):
    """[X/Fe] for every requested element via the training-time formula."""
    pisn_entry = pisn_interp(pisn_mass)
    out = {}
    for elem in elements:
        if elem == "Fe":
            out[elem] = 0.0
            continue
        try:
            v = salvadori_combined_abundratio(
                elem, elem, "Fe", "Fe",
                pisn_data=pisn_entry,
                sn_data=sn_entries,
                salv_sn_data=salv_sn,
                auto_sn=False, single_sn=False,
                sn_input=sn_src,
                f_pisn=f_pisn, f_ratio=f_ratio, tpop2=tpop2,
            )
            out[elem] = float(v) if np.isfinite(v) else np.nan
        except (ValueError, KeyError, IndexError):
            out[elem] = np.nan
    return out


def sample_fpisn(f_qs, rng):
    prob = np.array([0.05, 0.16, 0.5, 0.84, 0.95])
    val = np.array([f_qs[p] for p in prob])
    u = rng.uniform(0.0, 1.0)
    return float(np.clip(np.interp(u, prob, val), 0.0, 1.0))


def observed_for_star(star):
    """dict elem -> (value, error, status) over filter_elements."""
    ion_by_elem = {}
    for ion, rec in star["abundances"].items():
        elem = _ion_to_element(ion)
        if elem in filter_elements:
            ion_by_elem.setdefault(elem, (ion, rec))

    obs = {}
    for elem in ELEMENTS:
        if elem == "Fe":
            obs[elem] = (0.0, 0.0, "detected")
            continue
        if elem not in ion_by_elem:
            obs[elem] = (np.nan, np.nan, "missing")
            continue
        ion, rec = ion_by_elem[elem]
        xfe = rec.get("XFe")
        l_xfe = rec.get("l_XFe")
        err = rec.get("sigma_Total")
        if xfe is None or (isinstance(xfe, float) and np.isnan(xfe)):
            obs[elem] = (np.nan, np.nan, "missing")
        elif l_xfe == "<":
            obs[elem] = (float(xfe),
                         (float(err) if err is not None and np.isfinite(err) else np.nan),
                         "upper_limit")
        else:
            obs[elem] = (float(xfe),
                         (float(err) if err is not None and np.isfinite(err) else 0.0),
                         "detected")
    return obs


def main():
    pisn_interp, sn_entries, salv_sn, sn_src = load_yields()
    pred = pd.read_csv(PRED_CSV, dtype={"star": str})
    stars = {s["label"]: s for s in load_obs_li()}

    cand = pred[pred["f_pisn_mask_q50"] > FPISN_THRESH].sort_values(
        "f_pisn_mask_q50", ascending=False)
    print(f"candidates (mask q50 > {FPISN_THRESH}): {len(cand)}")

    for _, row in cand.iterrows():
        name = row["star"]
        star = stars.get(name)
        if star is None:
            print(f"!! {name}: not found in load_obs_li")
            continue

        f_qs = {
            0.05: float(row["f_pisn_mask_q05"]),
            0.16: float(row["f_pisn_mask_q16"]),
            0.50: float(row["f_pisn_mask_q50"]),
            0.84: float(row["f_pisn_mask_q84"]),
            0.95: float(row["f_pisn_mask_q95"]),
        }
        std_q50 = float(row["f_pisn_std_q50"])
        feh = star["params"].get("FeH")
        teff = star["params"].get("Teff")
        logg = star["params"].get("logg")

        obs = observed_for_star(star)

        # Monte-Carlo over f_pisn (mask quantiles) + nuisance params
        rng = np.random.default_rng(42)
        curves = np.full((N_MC, len(ELEMENTS)), np.nan)
        mass_draw = np.empty(N_MC)
        for k in range(N_MC):
            f_k = sample_fpisn(f_qs, rng)
            pm = rng.uniform(*PISN_MASS_RANGE)
            fr = loguniform.rvs(*F_RATIO_RANGE, random_state=rng)
            tp = loguniform.rvs(*TPOP2_RANGE, random_state=rng)
            mass_draw[k] = pm
            fx = forward_xfe(pisn_interp, sn_entries, salv_sn, sn_src,
                             f_k, pm, fr, tp, ELEMENTS)
            for j, e in enumerate(ELEMENTS):
                curves[k, j] = fx[e]

        med = np.nanmedian(curves, axis=0)
        lo = np.nanpercentile(curves, 16, axis=0)
        hi = np.nanpercentile(curves, 84, axis=0)
        med_mass = float(np.median(mass_draw))

        # --- observed arrays aligned to ELEMENTS ---
        obs_val = np.array([obs[e][0] for e in ELEMENTS])
        obs_err = np.array([obs[e][1] for e in ELEMENTS])
        obs_stat = [obs[e][2] for e in ELEMENTS]
        det = [i for i in range(len(ELEMENTS)) if obs_stat[i] == "detected"]
        ul = [i for i in range(len(ELEMENTS)) if obs_stat[i] == "upper_limit"]
        miss = [i for i in range(len(ELEMENTS)) if obs_stat[i] == "missing"]

        xpos = np.arange(len(ELEMENTS))

        # --- plot ---
        fig, ax = plt.subplots(figsize=(12, 5.8))

        finite_mock = np.isfinite(med)
        lo_f = np.where(np.isfinite(lo), lo, np.nan)
        hi_f = np.where(np.isfinite(hi), hi, np.nan)
        ax.fill_between(xpos, lo_f, hi_f, color="tab:blue", alpha=0.18,
                        label="model [X/Fe] (16-84%)")
        ax.plot(xpos[finite_mock], med[finite_mock], "-o", ms=4.5, lw=1.4,
                color="tab:blue", label="model [X/Fe] (median)")

        if det:
            ax.errorbar([xpos[i] for i in det], [obs_val[i] for i in det],
                        yerr=[obs_err[i] if np.isfinite(obs_err[i]) else 0.0
                              for i in det],
                        fmt="o", ms=6.5, color="tab:red", capsize=3, lw=1.2,
                        label="observed (detected)")

        for i in ul:
            ax.plot(xpos[i], obs_val[i], marker="x", ls="none", color="grey",
                    ms=8, mew=1.6)
            ax.annotate("", xy=(xpos[i], obs_val[i] - 0.28),
                        xytext=(xpos[i], obs_val[i]),
                        arrowprops=dict(arrowstyle="->", color="grey", lw=0.9))

        if miss:
            ybot = min(-3.0, np.nanmin(lo_f) - 0.6)
            ax.plot([xpos[i] for i in miss], [ybot] * len(miss),
                    marker="x", ls="none", color="lightgrey", ms=7, mew=1.2,
                    label="not detected")
        else:
            ybot = min(-3.0, np.nanmin(lo_f) - 0.6)

        ax.axhline(0, ls="--", color="k", lw=0.8, alpha=0.7)

        # y limits
        ymin = min(-3.0, np.nanmin(lo_f) - 0.5, (np.nanmin(obs_val[det]) if det else -3.0) - 0.5)
        ymax = max(3.0, np.nanmax(hi_f) + 0.5, (np.nanmax(obs_val[det]) if det else 3.0) + 0.5)
        ax.set_ylim(ymin, ymax)

        ax.set_xticks(xpos)
        ax.set_xticklabels([f"${e}$" for e in ELEMENTS], rotation=45, ha="right")
        ax.set_xlabel("Element")
        ax.set_ylabel("[X/Fe]")

        title = (
            f"{name}   f_pisn(mask) = {f_qs[0.50]:.2f} "
            f"[{f_qs[0.16]:.2f}, {f_qs[0.84]:.2f}]   "
            f"M_PISN(mock) ~ {med_mass:.0f} $M_\\odot$"
        )
        ax.set_title(title, pad=18)
        ax.text(0.02, 0.98,
                f"[Fe/H] = {feh:.2f}   Teff = {teff:.0f} K   logg = {logg:.2f}\n"
                f"std q50 = {std_q50:.2f}   "
                f"mask q05 = {f_qs[0.05]:.2f} / q95 = {f_qs[0.95]:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.8", alpha=0.9))
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.2)

        fig.tight_layout()
        out = os.path.join(RESULTS_DIR, f"{name}_yield.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}")

        # numerical verification (model cannot view images)
        print(f"  {name}: mask f_pisn={f_qs[0.50]:.3f} "
              f"[{f_qs[0.16]:.3f},{f_qs[0.84]:.3f}]  std={std_q50:.3f}  "
              f"M_PISN~{med_mass:.0f}")
        for j, e in enumerate(ELEMENTS):
            o = f"{obs_val[j]:+.3f}" if obs_stat[j] == "detected" else \
                (f"{obs_val[j]:+.3f}<" if obs_stat[j] == "upper_limit" else "  ---")
            m = f"{med[j]:+.3f}" if np.isfinite(med[j]) else "   nan"
            print(f"    {e:>3}  model={m}  obs={o}")


if __name__ == "__main__":
    main()
