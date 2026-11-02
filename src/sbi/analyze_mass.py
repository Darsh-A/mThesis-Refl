"""How does tracer strength depend on the PISN progenitor mass (M_pisn)?

For a grid of M_pisn values, compute each element's [X/Fe] *signal* (range over
f_pisn at fixed dilution/timing) and *nuisance* (range over f_ratio/tpop2 at
fixed f_pisn), then strength = signal / nuisance.  This shows whether N, Cu, Zn
(and the benchmark Ca, Si) become stronger or weaker tracers at different PISN
masses.

Outputs (src/sbi/results/sensitivity/):
    mass_dependence.txt
    mass_dependence.png
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

TRACERS = ["N", "Cu", "Zn", "Ca", "Si"]
M_GRID = np.array([150.0, 170.0, 190.0, 210.0, 230.0, 250.0, 270.0])
F_PISN_GRID = np.linspace(0.05, 0.9, 25)
N_NUIS = 100


def main():
    forward, pairs = make_forward_fast(elements=TRACERS + ["Fe"])
    idx = {e: pairs.index((e, "Fe")) for e in TRACERS}

    rng = np.random.default_rng(0)

    signal = {e: np.empty(len(M_GRID)) for e in TRACERS}
    nuisance = {e: np.empty(len(M_GRID)) for e in TRACERS}
    strength = {e: np.empty(len(M_GRID)) for e in TRACERS}

    for mi, m_pisn in enumerate(M_GRID):
        # signal: [X/Fe] vs f_pisn at fixed nuisance
        traj = {e: np.empty(len(F_PISN_GRID)) for e in TRACERS}
        for j, f in enumerate(F_PISN_GRID):
            theta = np.array([f, -3.0, np.log10(1e7), m_pisn])
            x = forward(theta)
            for e in TRACERS:
                traj[e][j] = x[idx[e]]

        # nuisance: [X/Fe] over f_ratio/tpop2 at fixed f_pisn, M_pisn
        nui_vals = {e: [] for e in TRACERS}
        for _ in range(N_NUIS):
            theta = np.array([
                0.5,
                rng.uniform(-4.0, -1.0),
                rng.uniform(np.log10(3.2e6), np.log10(17.4e6)),
                m_pisn,
            ])
            x = forward(theta)
            for e in TRACERS:
                v = x[idx[e]]
                if np.isfinite(v):
                    nui_vals[e].append(v)

        for e in TRACERS:
            sig = float(np.ptp(traj[e]))
            nui = float(np.ptp(nui_vals[e])) if nui_vals[e] else np.nan
            signal[e][mi] = sig
            nuisance[e][mi] = nui
            strength[e][mi] = sig / nui if nui and nui > 0 else np.nan

    # ---- text ----
    lines = ["Tracer strength (signal / nuisance) vs PISN progenitor mass M_pisn.", ""]
    header = f"{'M_pisn':>7}" + "".join(f"{e:>10}" for e in TRACERS)
    lines.append(header)
    for mi, m in enumerate(M_GRID):
        lines.append(f"{m:>7.0f}" + "".join(f"{strength[e][mi]:>10.2f}" for e in TRACERS))
    with open(os.path.join(OUT_DIR, "mass_dependence.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(header)
    for mi, m in enumerate(M_GRID):
        print(f"{m:>7.0f}" + "".join(f"{strength[e][mi]:>10.2f}" for e in TRACERS))

    # ---- plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"N": "tab:purple", "Cu": "tab:red", "Zn": "tab:green",
              "Ca": "tab:blue", "Si": "tab:orange"}

    ax = axes[0]
    for e in TRACERS:
        ax.plot(M_GRID, strength[e], marker="o", lw=1.6, color=colors[e], label=e)
    ax.axhline(1.0, color="k", ls="--", lw=1, label="signal = nuisance")
    ax.set_xlabel("M_pisn (M_sun)")
    ax.set_ylabel("tracer strength (signal / nuisance)")
    ax.set_title("Tracer strength vs PISN mass")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for e in TRACERS:
        ax.plot(M_GRID, signal[e], marker="o", lw=1.6, color=colors[e], label=e)
    ax.set_xlabel("M_pisn (M_sun)")
    ax.set_ylabel("signal (dex of [X/Fe] over f_pisn)")
    ax.set_title("Absolute signal vs PISN mass")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "mass_dependence.png"), dpi=130)
    plt.close(fig)

    print(f"\nsaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
