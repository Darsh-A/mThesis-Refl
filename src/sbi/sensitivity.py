"""Forward-model sensitivity: which ratios/elements constrain f_pisn?

For each abundance ratio [X/Y], measure how much it moves when f_pisn changes
vs. how much it moves when the nuisance parameters (f_ratio, tpop2, M_pisn)
change -- using the Salvadori forward model directly (no ML, no emulator).

A *clean* PISN tracer is a ratio that responds strongly to f_pisn but weakly to
the nuisance parameters.  Sensitivity is computed by central finite differences
at many prior samples, then scaled by each parameter's prior width so the
"signal" and "nuisance" terms are directly comparable in dex.

Outputs (src/sbi/results/sensitivity/):
    sensitivity_ratios.txt   ranked ratios + element aggregation
    element_tracers.png      element-level tracer scores
    ratio_sensitivity.png     PISN vs nuisance sensitivity per ratio
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

PARAM_NAMES = ["f_pisn", "log10_f_ratio", "log10_tpop2", "M_pisn"]
# prior ranges (from src.sbi.data.PRIOR_BOUNDS) -> width for scaling
PRIOR_RANGE = {
    "f_pisn": (0.0, 1.0),
    "log10_f_ratio": (-4.0, -1.0),
    "log10_tpop2": (6.5, 7.25),
    "M_pisn": (150.0, 270.0),
}
FD_STEP = {
    "f_pisn": 0.01,
    "log10_f_ratio": 0.05,
    "log10_tpop2": 0.05,
    "M_pisn": 5.0,
}
N_SAMPLES = 30


def main():
    forward, pairs = make_forward_fast()
    ratio_names = [f"[{x}/{y}]" for x, y in pairs]
    elements = sorted({e for p in pairs for e in p})

    # representative theta samples (interior of the prior)
    rng = np.random.default_rng(0)
    thetas = np.column_stack([
        rng.uniform(0.1, 0.9, N_SAMPLES),
        rng.uniform(-3.5, -1.5, N_SAMPLES),
        rng.uniform(6.6, 7.1, N_SAMPLES),
        rng.uniform(160.0, 260.0, N_SAMPLES),
    ])

    widths = np.array([
        PRIOR_RANGE[n][1] - PRIOR_RANGE[n][0] for n in PARAM_NAMES
    ])

    # range-scaled sensitivity: mean |dx/dtheta| * prior_width (dex over range)
    S = np.zeros((len(pairs), len(PARAM_NAMES)))
    for th in thetas:
        for k, name in enumerate(PARAM_NAMES):
            th_hi = th.copy()
            th_lo = th.copy()
            th_hi[k] += FD_STEP[name]
            th_lo[k] -= FD_STEP[name]
            x_hi = forward(th_hi)
            x_lo = forward(th_lo)
            if not (np.all(np.isfinite(x_hi)) and np.all(np.isfinite(x_lo))):
                continue
            deriv = (x_hi - x_lo) / (2.0 * FD_STEP[name])
            S[:, k] += np.abs(deriv) * widths[k]
    S /= N_SAMPLES

    pisn = S[:, 0]
    nuisance = np.sqrt(S[:, 1] ** 2 + S[:, 2] ** 2 + S[:, 3] ** 2)
    score = pisn / (nuisance + 1e-6)

    # ---- ranked ratios ----
    order = np.argsort(score)[::-1]
    lines = ["Ratio sensitivity: signal = |d[X/Y]/d f_pisn| over prior width,",
             "nuisance = sqrt(sum of |d/d f_ratio, tpop2, M_pisn|^2 over prior widths).",
             "score = signal / nuisance (clean tracer if large).", ""]
    lines.append(f"{'rank':<5}{'ratio':<10}{'signal':>9}{'nuisance':>10}{'score':>8}")
    for rank, j in enumerate(order[:20], 1):
        lines.append(f"{rank:<5}{ratio_names[j]:<10}{pisn[j]:>9.2f}"
                     f"{nuisance[j]:>10.2f}{score[j]:>8.2f}")

    # ---- element aggregation ----
    elem_pisn = {e: 0.0 for e in elements}
    elem_nuis = {e: 0.0 for e in elements}
    for j, (x, y) in enumerate(pairs):
        elem_pisn[x] += pisn[j]
        elem_pisn[y] += pisn[j]
        elem_nuis[x] += nuisance[j]
        elem_nuis[y] += nuisance[j]
    elem_score = {e: elem_pisn[e] / (elem_nuis[e] + 1e-6) for e in elements}
    elem_order = sorted(elements, key=lambda e: elem_score[e], reverse=True)

    lines.append("")
    lines.append(f"{'element':<10}{'signal':>9}{'nuisance':>10}{'score':>8}")
    for e in elem_order:
        lines.append(f"{e:<10}{elem_pisn[e]:>9.2f}{elem_nuis[e]:>10.2f}"
                     f"{elem_score[e]:>8.2f}")

    with open(os.path.join(OUT_DIR, "sensitivity_ratios.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # print the top of the table to stdout too
    print("\n".join(lines[:25]))
    print("\n=== element tracer score ===")
    for e in elem_order:
        print(f"  {e:<4} score={elem_score[e]:.2f}  "
              f"(signal={elem_pisn[e]:.2f}, nuisance={elem_nuis[e]:.2f})")

    # ---- plots ----
    # element tracer scores
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(np.arange(len(elem_order)), [elem_score[e] for e in elem_order],
           color="tab:blue")
    ax.set_xticks(np.arange(len(elem_order)))
    ax.set_xticklabels(elem_order)
    ax.set_xlabel("element")
    ax.set_ylabel("tracer score (signal / nuisance)")
    ax.set_title("PISN tracer score by element")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "element_tracers.png"), dpi=130)
    plt.close(fig)

    # ratio scatter: PISN sensitivity vs nuisance sensitivity
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(nuisance, pisn, s=18, alpha=0.6, color="tab:green")
    lim = max(nuisance.max(), pisn.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="signal = nuisance")
    top = order[:6]
    for j in top:
        ax.annotate(ratio_names[j], (nuisance[j], pisn[j]),
                    fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("nuisance sensitivity (dex)")
    ax.set_ylabel("f_pisn sensitivity (dex)")
    ax.set_title("Abundance ratios: f_pisn vs nuisance sensitivity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ratio_sensitivity.png"), dpi=130)
    plt.close(fig)

    print(f"\nsaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
