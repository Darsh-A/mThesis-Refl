"""Does tracer availability change the ranking?

Computes each element's [X/Fe] tracer strength (signal/nuisance) from the
forward model, then cross-tabulates against which elements the Li+22 and
Xiang+2019 catalogues actually observe.  The per-element strength is intrinsic
physics (independent of the observed set); what changes with availability is
*which* tracers you can access -- and notably which odd-Z tracers are missing.

Outputs (src/sbi/results/sensitivity/):
    availability.txt
    availability.png
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.sbi.forward import make_forward_fast  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results", "sensitivity")
os.makedirs(OUT_DIR, exist_ok=True)

CATALOGS = {
    "Li+22": ["C", "Na", "Mg", "Si", "Ca", "Cr", "Mn", "Fe", "Co", "Ni", "Zn"],
    "Xiang": ["C", "N", "O", "Na", "Mg", "Al", "Si", "Ca", "Ti", "Cr", "Mn",
              "Fe", "Co", "Ni", "Cu"],
}
ALL_TRACERS = ["C", "N", "O", "Na", "Mg", "Al", "Si", "Ca", "Ti", "Cr", "Mn",
               "Co", "Ni", "Cu", "Zn"]

F_PISN_GRID = np.linspace(0.05, 0.9, 25)
N_NUIS = 200

SN_SOURCE = sys.argv[1] if len(sys.argv) > 1 else "Limongi18"


def main():
    forward, pairs = make_forward_fast(sn_source=SN_SOURCE, elements=ALL_TRACERS + ["Fe"])
    idx = {e: pairs.index((e, "Fe")) for e in ALL_TRACERS}
    rng = np.random.default_rng(0)

    strengths = {}
    for e in ALL_TRACERS:
        traj = []
        for f in F_PISN_GRID:
            theta = np.array([f, -3.0, np.log10(1e7), 200.0])
            traj.append(forward(theta)[idx[e]])
        sig = float(np.ptp(traj))

        vals = []
        for _ in range(N_NUIS):
            theta = np.array([
                0.5,
                rng.uniform(-4.0, -1.0),
                rng.uniform(np.log10(3.2e6), np.log10(17.4e6)),
                rng.uniform(150.0, 270.0),
            ])
            v = forward(theta)[idx[e]]
            if np.isfinite(v):
                vals.append(v)
        nui = float(np.ptp(vals)) if vals else np.nan
        strengths[e] = sig / nui if nui and nui > 0 else np.nan

    order = sorted(ALL_TRACERS, key=lambda e: -strengths[e])

    lines = ["Element [X/Fe] tracer strength (signal / nuisance), by catalogue availability.", ""]
    lines.append(f"{'element':<7}{'strength':>9}  Li+22  Xiang")
    for e in order:
        l22 = "yes" if e in CATALOGS["Li+22"] else " - "
        xg = "yes" if e in CATALOGS["Xiang"] else " - "
        lines.append(f"{e:<7}{strengths[e]:>9.2f}  {l22:<5} {xg}")
    with open(os.path.join(OUT_DIR, f"availability_{SN_SOURCE}.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"{'element':<7}{'strength':>9}  Li+22  Xiang")
    for e in order:
        l22 = "yes" if e in CATALOGS["Li+22"] else " - "
        xg = "yes" if e in CATALOGS["Xiang"] else " - "
        print(f"{e:<7}{strengths[e]:>9.2f}  {l22:<5} {xg}")

    for k, els in CATALOGS.items():
        avail = [e for e in els if e != "Fe"]
        top = sorted(avail, key=lambda e: -strengths.get(e, 0.0))[:5]
        print(f"\n{k} top-5 available tracers: "
              + ", ".join(f"{e}({strengths[e]:.2f})" for e in top))

    # ---- plot: strength, colored by availability ----
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, e in enumerate(order):
        in_l22 = e in CATALOGS["Li+22"]
        in_xg = e in CATALOGS["Xiang"]
        color = "tab:blue" if (in_l22 and in_xg) else (
            "tab:green" if in_l22 else "tab:red")
        ax.bar(i, strengths[e], color=color)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel("tracer strength (signal / nuisance)")
    ax.set_title("[X/Fe] tracer strength by element\n(blue=both, green=Li+22 only, red=Xiang only)")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="signal = nuisance")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"availability_{SN_SOURCE}.png"), dpi=130)
    plt.close(fig)

    print(f"\nsaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
