"""Plot [X/Y] vs [A/B] with scatter colored by metallicity + density contours.

Generalizes fig7's abundance plot to *arbitrary* element pairs (not just
[X/Fe] vs [Fe/H]). Uses the same physics: HW2002 PISN yields + Limongi18
(LC18, velocity=0) SNII yields mixed by f_pisn, with the SN turnoff mass set
by the Limongi lifetime table.

Two metallicity treatments are available:
    * "snap"  -- snap log10(Z_star/Z_sun) to the nearest LC18 [Fe/H] grid
                 point {0, -1, -2, -3} (this is what salvadori_combined_abundratio
                 does internally, via salvadori_select_limongi_feh).
    * "cont"  -- keep [Fe/H] continuous, interpolating the SN yields between the
                 4 LC18 grid points (smooth, no discrete banding).

The element pairs are fully configurable. Run from the repo root, e.g.:

    uv run python -m src.salvadori.fig_ratio_ratio \
        --ratio1 Zn Fe --ratio2 Cu Fe --mode both

or edit the DEFAULT_* constants below and run without arguments.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, loguniform

# --- Memoize the CSV-reading lifetime lookup (it re-reads the whole CSV every
# --- call otherwise, which is the dominant cost of the continuous mode). Only
# --- ~36 unique (velocity, feh, phase, mass) inputs exist, so this is a big win.
import functools as _functools
import src.utils as _src_utils
_src_utils.limongi_lifetime = _functools.lru_cache(maxsize=None)(_src_utils.limongi_lifetime)

from src.load_data import load_hw2002, load_limongi18
from src.salvadori_funcs import (
    salvadori_combined_abundratio,
    salvadori_Y_X_II,
    salvadori_Y_Z_II,
    salvadori_yields,
)
from src.utils import (
    _combine_elements,
    build_pisn_interpolator,
    limongi_mass_from_lifetime,
    salvadori_select_limongi_feh,
)
from src.yield_sources import get_source
from params import Z_SUN, asplund, atomic_mass

# --------------------------------------------------------------------------
# Editable defaults (overridable via CLI)
# --------------------------------------------------------------------------
DEFAULT_RATIO1 = ("Zn", "Fe")   # x-axis ratio  ->  [X/Y]
DEFAULT_RATIO2 = ("Cu", "Fe")   # y-axis ratio  ->  [A/B]
DEFAULT_MODE = "both"           # "snap" | "cont" | "both"
F_PISN = 0.5                    # PISN mass fraction of the enrichment
N_DRAWS = 5000
SEED = 42
PISN_MASS_RANGE = (150.0, 270.0)
F_RATIO_RANGE = (1e-4, 1e-1)
TPOP2_RANGE = (3.2e6, 17.4e6)   # population-II age (yr)
OUT_DIR = "plots/abundance_ratio_ratio"

# LC18 metallicity grid (ascending, for np.interp) + plotting colors
FEH_X = [-3.0, -2.0, -1.0, 0.0]
FEH_COLORS = {0: "#d62728", -1: "#ff7f0e", -2: "#2ca02c", -3: "#1f77b4"}

with open(asplund) as _f:
    SOLAR = json.load(_f)


def A(elem: str) -> float:
    """Asplund+09 solar abundance A(X) = log10(N_X/N_H) + 12."""
    return SOLAR[elem]["val"]


def _norm_element(s: str) -> str:
    s = s.strip()
    return s[0].upper() + s[1:].lower()


def _combined_ratio_cont(elem1, elem2, py, f_pisn, f_ratio, tpop2, sn_data, sn_source):
    """Continuous-[Fe/H] analogue of salvadori_combined_abundratio.

    Mirrors salvadori_combined_abundratio exactly, but instead of snapping
    [Fe/H] to a grid point it evaluates the SN yields at each of the 4 LC18
    grid points and np.interp's them at the continuous value
    feh_cont = clip(log10(Z_star/Z_sun), -3, 0).

    Returns (ratio, feh_cont).  Returns np.nan if the ratio is undefined.
    """
    Yz_pisn = sum(v for k, v in py.items() if k not in ("H", "He"))
    Z_star = f_ratio * Yz_pisn
    feh_cont = float(np.clip(np.log10(Z_star / Z_SUN), -3.0, 0.0))

    ye1, ye2, yz = [], [], []
    for fv in FEH_X:
        m = limongi_mass_from_lifetime(tpop2, int(fv), velocity=0)
        if m is None:            # tpop2 outside LC18 tabulated range at this feh
            return np.nan, feh_cont
        ye1.append(salvadori_Y_X_II(sn_data, elem1, m, sn_input=sn_source, feh=int(fv)))
        ye2.append(salvadori_Y_X_II(sn_data, elem2, m, sn_input=sn_source, feh=int(fv)))
        yz.append(salvadori_Y_Z_II(sn_data, m, sn_input=sn_source, feh=int(fv)))

    yx1 = float(np.interp(feh_cont, FEH_X, ye1))
    yx2 = float(np.interp(feh_cont, FEH_X, ye2))
    yz_sn = float(np.interp(feh_cont, FEH_X, yz))

    beta = (1.0 - f_pisn) / f_pisn
    Yx1_p = py[elem1]
    Yx2_p = py[elem2]

    if yz_sn <= 0.0:
        num, den = Yx1_p, Yx2_p   # pure-PISN fallback
    else:
        sn_term_1 = (Yz_pisn / yz_sn) * yx1
        sn_term_2 = (Yz_pisn / yz_sn) * yx2
        num = Yx1_p + beta * sn_term_1
        den = Yx2_p + beta * sn_term_2

    if num <= 0.0 or den <= 0.0:
        return np.nan, feh_cont

    ratio = np.log10(num / den) - (A(elem1) - A(elem2)) - np.log10(atomic_mass[elem1] / atomic_mass[elem2])
    return ratio, feh_cont


def _add_density_contours(ax, x, y, levels=7):
    """Overlay gaussian_kde density contour lines (neutral, so they don't
    compete with the metallicity color encoding)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 20:
        return
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)
    xmin, xmax = np.percentile(x, [0.5, 99.5])
    ymin, ymax = np.percentile(y, [0.5, 99.5])
    xpad = 0.05 * (xmax - xmin)
    ypad = 0.05 * (ymax - ymin)
    Xg, Yg = np.mgrid[xmin - xpad:xmax + xpad:150j, ymin - ypad:ymax + ypad:150j]
    Z = np.reshape(kde(np.vstack([Xg.ravel(), Yg.ravel()])).T, Xg.shape)
    ax.contour(Xg, Yg, Z, levels=levels, colors="0.25", linewidths=0.6, alpha=0.55)


def _plot_panel(ax, r1, r2, met, xlabel, ylabel, title, continuous):
    _add_density_contours(ax, r1, r2)
    if continuous:
        sc = ax.scatter(r1, r2, c=met, s=6, alpha=0.5, cmap="viridis")
        plt.colorbar(sc, ax=ax, label="[Fe/H] continuous")
    else:
        for fh in [0, -1, -2, -3]:
            m = met == fh
            if m.sum() == 0:
                continue
            ax.scatter(r1[m], r2[m], s=6, alpha=0.5, color=FEH_COLORS[fh], label=f"[Fe/H]={fh}")
        ax.legend(fontsize=7, loc="best")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)


def _corr(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 3 else np.nan


def run(ratio1, ratio2, mode, n=N_DRAWS, f_pisn=F_PISN, seed=SEED):
    # ---- load + prepare data (once) ----
    sn_data = [e for e in load_limongi18() if e["params"]["velocity"] == 0]
    salv_sn_data = salvadori_yields(sn_data)          # single_sn path (unused, kept for API parity)
    sn_source = get_source("Limongi18")

    hw_yields = load_hw2002()[1:]
    pisn_interp = build_pisn_interpolator(salvadori_yields(hw_yields))

    # ---- validate requested elements against all three tables ----
    pisn_elements = set(pisn_interp(PISN_MASS_RANGE[0])["yields"].keys())
    sn_elements = set(sn_data[0]["yields"].keys())
    for e in (ratio1[0], ratio1[1], ratio2[0], ratio2[1]):
        if e not in atomic_mass:
            raise ValueError(f"'{e}' not in params.atomic_mass")
        if e not in SOLAR:
            raise ValueError(f"'{e}' not in data/asplund.json")
        if e not in pisn_elements:
            raise ValueError(f"'{e}' not in PISN (HW2002) yields; available: {sorted(pisn_elements)}")
        if e not in sn_elements:
            raise ValueError(f"'{e}' not in SN (Limongi18) yields; available: {sorted(sn_elements)}")

    rng = np.random.RandomState(seed)
    snap_r1, snap_r2, snap_met = [], [], []
    cont_r1, cont_r2, cont_met = [], [], []
    n_skip_snap = n_skip_cont = 0

    for _ in range(n):
        pm = rng.uniform(*PISN_MASS_RANGE)
        f_ratio = loguniform.rvs(*F_RATIO_RANGE, random_state=rng)
        tpop2 = loguniform.rvs(*TPOP2_RANGE, random_state=rng)
        pisn_entry = pisn_interp(pm)
        py = _combine_elements(pisn_entry["yields"])

        # ---- snapped metallicity (repo's own path) ----
        r1 = salvadori_combined_abundratio(
            ratio1[0], ratio1[0], ratio1[1], ratio1[1],
            pisn_data=pisn_entry, sn_data=sn_data, salv_sn_data=salv_sn_data,
            auto_sn=False, single_sn=False, sn_input=sn_source,
            f_pisn=f_pisn, f_ratio=f_ratio, tpop2=tpop2)
        r2 = salvadori_combined_abundratio(
            ratio2[0], ratio2[0], ratio2[1], ratio2[1],
            pisn_data=pisn_entry, sn_data=sn_data, salv_sn_data=salv_sn_data,
            auto_sn=False, single_sn=False, sn_input=sn_source,
            f_pisn=f_pisn, f_ratio=f_ratio, tpop2=tpop2)
        Yz_pisn = sum(v for k, v in py.items() if k not in ("H", "He"))
        feh_snap = salvadori_select_limongi_feh(f_ratio * Yz_pisn / Z_SUN)
        if np.isfinite(r1) and np.isfinite(r2):
            snap_r1.append(r1); snap_r2.append(r2); snap_met.append(feh_snap)
        else:
            n_skip_snap += 1

        # ---- continuous metallicity ----
        cr1, fh_cont = _combined_ratio_cont(ratio1[0], ratio1[1], py, f_pisn, f_ratio, tpop2, sn_data, sn_source)
        cr2, _ = _combined_ratio_cont(ratio2[0], ratio2[1], py, f_pisn, f_ratio, tpop2, sn_data, sn_source)
        if np.isfinite(cr1) and np.isfinite(cr2):
            cont_r1.append(cr1); cont_r2.append(cr2); cont_met.append(fh_cont)
        else:
            n_skip_cont += 1

    snap = (np.array(snap_r1), np.array(snap_r2), np.array(snap_met), n_skip_snap)
    cont = (np.array(cont_r1), np.array(cont_r2), np.array(cont_met), n_skip_cont)
    return snap, cont


def plot(ratio1, ratio2, mode, snap, cont):
    xlab = f"[{ratio1[0]}/{ratio1[1]}]"
    ylab = f"[{ratio2[0]}/{ratio2[1]}]"

    if mode == "both":
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        _plot_panel(axes[0], snap[0], snap[1], snap[2], xlab, ylab, "SNAPPED [Fe/H]", continuous=False)
        _plot_panel(axes[1], cont[0], cont[1], cont[2], xlab, ylab, "CONTINUOUS [Fe/H]", continuous=True)
    elif mode == "snap":
        fig, ax = plt.subplots(figsize=(6.5, 6))
        _plot_panel(ax, snap[0], snap[1], snap[2], xlab, ylab, "SNAPPED [Fe/H]", continuous=False)
    else:  # "cont"
        fig, ax = plt.subplots(figsize=(6.5, 6))
        _plot_panel(ax, cont[0], cont[1], cont[2], xlab, ylab, "CONTINUOUS [Fe/H]", continuous=True)

    fig.suptitle(f"f_pisn = {F_PISN:.2f}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f"{ratio1[0]}{ratio1[1]}_vs_{ratio2[0]}{ratio2[1]}_{mode}"
    path = os.path.join(OUT_DIR, f"{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description="Plot [X/Y] vs [A/B] colored by metallicity.")
    p.add_argument("--ratio1", nargs=2, default=DEFAULT_RATIO1, metavar=("X", "Y"))
    p.add_argument("--ratio2", nargs=2, default=DEFAULT_RATIO2, metavar=("A", "B"))
    p.add_argument("--mode", choices=["snap", "cont", "both"], default=DEFAULT_MODE)
    p.add_argument("--f-pisn", type=float, default=F_PISN)
    p.add_argument("--n", type=int, default=N_DRAWS)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args(argv)

    ratio1 = (_norm_element(args.ratio1[0]), _norm_element(args.ratio1[1]))
    ratio2 = (_norm_element(args.ratio2[0]), _norm_element(args.ratio2[1]))

    snap, cont = run(ratio1, ratio2, args.mode, n=args.n, f_pisn=args.f_pisn, seed=args.seed)
    path = plot(ratio1, ratio2, args.mode, snap, cont)

    print(f"[{ratio1[0]}/{ratio1[1]}] vs [{ratio2[0]}/{ratio2[1]}]  (mode={args.mode}, n={args.n})")
    if args.mode in ("snap", "both"):
        print(f"  SNAPPED:  n={len(snap[0])} (skipped {snap[3]})  "
              f"[{ratio1[0]}/{ratio1[1]}] range [{snap[0].min():+.2f}, {snap[0].max():+.2f}]  "
              f"[{ratio2[0]}/{ratio2[1]}] range [{snap[1].min():+.2f}, {snap[1].max():+.2f}]")
        print(f"            corr(x, met)={_corr(snap[0], snap[2]):+.3f}  corr(y, met)={_corr(snap[1], snap[2]):+.3f}")
    if args.mode in ("cont", "both"):
        print(f"  CONT:     n={len(cont[0])} (skipped {cont[3]})  "
              f"[{ratio1[0]}/{ratio1[1]}] range [{cont[0].min():+.2f}, {cont[0].max():+.2f}]  "
              f"[{ratio2[0]}/{ratio2[1]}] range [{cont[1].min():+.2f}, {cont[1].max():+.2f}]")
        print(f"            corr(x, met)={_corr(cont[0], cont[2]):+.3f}  corr(y, met)={_corr(cont[1], cont[2]):+.3f}")
    print(f"  Saved {path}")


if __name__ == "__main__":
    main()
