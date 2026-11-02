"""Focused tracer-strength analysis for N, Cu, Zn (vs Ca/Si as benchmarks).

For each element X, compute [X/Fe] with the Salvadori forward model and
separate its *signal* (how much [X/Fe] moves over f_pisn) from its *nuisance*
(how much [X/Fe] moves when f_ratio / tpop2 / M_pisn vary).  A strong tracer
has signal >> nuisance.

Outputs (src/sbi/results/sensitivity/):
    tracers_ncu_zn.txt
    tracers_ncu_zn.png
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
F_PISN_GRID = np.linspace(0.05, 0.9, 25)
REF_NUIS = {"f_ratio": 1e-3, "tpop2": 1e7, "M_pisn": 200.0}
N_NUIS = 200


def main():
    forward, pairs = make_forward_fast(elements=TRACERS + ["Fe"])
    idx = {e: pairs.index((e, "Fe")) for e in TRACERS}

    # ---- signal: [X/Fe] vs f_pisn at fixed nuisance ----
    traj = {e: np.empty(len(F_PISN_GRID)) for e in TRACERS}
    for j, f in enumerate(F_PISN_GRID):
        theta = np.array([f, np.log10(REF_NUIS["f_ratio"]),
                          np.log10(REF_NUIS["tpop2"]), REF_NUIS["M_pisn"]])
        x = forward(theta)
        for e in TRACERS:
            traj[e][j] = x[idx[e]]

    # ---- nuisance: [X/Fe] spread at f_pisn=0.5 over nuisance samples ----
    rng = np.random.default_rng(0)
    nui_vals = {e: [] for e in TRACERS}
    for _ in range(N_NUIS):
        theta = np.array([
            0.5,
            rng.uniform(-4.0, -1.0),
            rng.uniform(np.log10(3.2e6), np.log10(17.4e6)),
            rng.uniform(150.0, 270.0),
        ])
        x = forward(theta)
        for e in TRACERS:
            v = x[idx[e]]
            if np.isfinite(v):
                nui_vals[e].append(v)

    lines = ["Element tracer strength (signal = [X/Fe] range over f_pisn;",
             "nuisance = [X/Fe] range over f_ratio/tpop2/M_pisn at f_pisn=0.5).", ""]
    lines.append(f"{'element':<8}{'signal(dex)':>12}{'nuisance(dex)':>14}{'strength':>10}")
    strengths = {}
    for e in TRACERS:
        sig = float(np.ptp(traj[e]))
        nui = float(np.ptp(nui_vals[e])) if nui_vals[e] else np.nan
        strengths[e] = sig / nui if nui and nui > 0 else np.nan
        lines.append(f"{e:<8}{sig:>12.2f}{nui:>14.2f}{strengths[e]:>10.2f}")
        print(f"{e:<6} signal={sig:5.2f} dex  nuisance={nui:5.2f} dex  "
              f"strength={strengths[e]:5.2f}")

    with open(os.path.join(OUT_DIR, "tracers_ncu_zn.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- plot ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    colors = {"N": "tab:purple", "Cu": "tab:red", "Zn": "tab:green",
              "Ca": "tab:blue", "Si": "tab:orange"}
    for e in TRACERS:
        ax.plot(F_PISN_GRID, traj[e], marker="o", ms=3, lw=1.6,
                color=colors[e], label=f"{e} (strength {strengths[e]:.1f})")
    ax.set_xlabel("f_pisn")
    ax.set_ylabel("[X/Fe] (dex)")
    ax.set_title("[X/Fe] vs f_pisn (signal)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i, e in enumerate(TRACERS):
        ax.bar(i, strengths[e], color=colors[e])
    ax.set_xticks(np.arange(len(TRACERS)))
    ax.set_xticklabels(TRACERS)
    ax.axhline(1.0, color="k", ls="--", lw=1, label="signal = nuisance")
    ax.set_ylabel("tracer strength (signal / nuisance)")
    ax.set_title("Tracer strength: N, Cu, Zn vs Ca, Si")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "tracers_ncu_zn.png"), dpi=130)
    plt.close(fig)

    print(f"\nsaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
