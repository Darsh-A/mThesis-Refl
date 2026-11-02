"""How does the PISN-tracer ranking change with f_pisn?

Sweeps f_pisn from ~0.01 to ~0.9 and, at each value, computes each ratio's
*log-log* sensitivity (dex of [X/Y] per dex of parameter) to f_pisn vs. the
nuisance parameters (f_ratio, tpop2, M_pisn).  All sensitivities are
dimensionless (dex per dex), so they are directly comparable.

This answers: which elements still discriminate PISN from pure SNII when the
PISN fraction is tiny (the detection regime)?

Outputs (src/sbi/results/sensitivity/):
    sweep_fpisn.txt
    sweep_fpisn.png
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
from src.randomForest.dataset import _feature_pairs  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results", "sensitivity")
os.makedirs(OUT_DIR, exist_ok=True)

F_PISN_GRID = np.array([0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.9])
N_NUIS = 15
DELTA = 0.02  # dex step in log10 space


def main():
    forward, pairs = make_forward_fast()
    ratio_names = [f"[{x}/{y}]" for x, y in pairs]
    elements = sorted({e for p in pairs for e in p})

    def forward_log10(l):
        # l = [log10 f_pisn, log10 f_ratio, log10 tpop2, log10 M_pisn]
        return forward(np.array([10.0 ** l[0], l[1], l[2], 10.0 ** l[3]]))

    rng = np.random.default_rng(0)

    elem_score = {e: np.empty(len(F_PISN_GRID)) for e in elements}
    elem_signal = {e: np.empty(len(F_PISN_GRID)) for e in elements}

    for gi, f_pisn in enumerate(F_PISN_GRID):
        log10_fpisn = float(np.log10(f_pisn))
        S = np.zeros((len(pairs), 4))
        for _ in range(N_NUIS):
            base = np.array([
                log10_fpisn,
                rng.uniform(-3.5, -1.5),
                rng.uniform(6.6, 7.1),
                rng.uniform(np.log10(160.0), np.log10(260.0)),
            ])
            if not np.all(np.isfinite(forward_log10(base))):
                continue
            for k in range(4):
                hi = base.copy()
                lo = base.copy()
                hi[k] += DELTA
                lo[k] -= DELTA
                xh = forward_log10(hi)
                xl = forward_log10(lo)
                if not (np.all(np.isfinite(xh)) and np.all(np.isfinite(xl))):
                    continue
                S[:, k] += np.abs(xh - xl) / (2.0 * DELTA)
        S /= N_NUIS

        pisn = S[:, 0]
        nuisance = np.sqrt(S[:, 1] ** 2 + S[:, 2] ** 2 + S[:, 3] ** 2)
        for e in elements:
            idx = [j for j, (x, y) in enumerate(pairs) if e in (x, y)]
            sig = float(pisn[idx].sum())
            nui = float(nuisance[idx].sum())
            elem_signal[e][gi] = sig
            elem_score[e][gi] = sig / (nui + 1e-6)

    # ---- text: ranking at lowest and highest f_pisn ----
    lines = ["Element tracer score (signal/nuisance) vs f_pisn", ""]
    for gi, f in enumerate(F_PISN_GRID):
        rank = sorted(elements, key=lambda e: elem_score[e][gi], reverse=True)
        lines.append(f"f_pisn={f:.2f}: " + "  ".join(
            f"{e}:{elem_score[e][gi]:.2f}" for e in rank))
    with open(os.path.join(OUT_DIR, "sweep_fpisn.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print("=== element tracer score vs f_pisn ===")
    for e in sorted(elements, key=lambda e: elem_score[e][-1], reverse=True):
        vals = " ".join(f"{v:.2f}" for v in elem_score[e])
        print(f"  {e:<4} {vals}")

    # ---- plot ----
    top5 = sorted(elements, key=lambda e: elem_score[e][-1], reverse=True)[:5]
    poor = ["Fe", "Ni", "N"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for e in elements:
        if e in top5:
            ax.plot(F_PISN_GRID, elem_score[e], marker="o", lw=1.6, label=e)
        elif e in poor:
            ax.plot(F_PISN_GRID, elem_score[e], marker="s", lw=1.2, ls="--", label=e)
        else:
            ax.plot(F_PISN_GRID, elem_score[e], lw=0.9, alpha=0.35, color="gray")
    ax.set_xscale("log")
    ax.set_xlabel("f_pisn")
    ax.set_ylabel("tracer score (signal / nuisance)")
    ax.set_title("PISN tracer score vs f_pisn (log-log sensitivity)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "sweep_fpisn.png"), dpi=130)
    plt.close(fig)

    print(f"\nsaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
