"""Quantify the odd-even effect as a PISN tracer.

Classifies each pairwise abundance ratio by the parity of its two elements'
atomic numbers (odd-odd, odd-even, even-even) and compares their tracer scores
(signal / nuisance, from the forward-model sensitivity).  The odd-even effect
predicts that odd-Z/even-Z ratios (e.g. [Na/Mg], [Al/Ca]) should be the
strongest PISN discriminators.

Outputs (src/sbi/results/sensitivity/):
    odd_even.txt
    odd_even.png
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

Z = {"C": 6, "N": 7, "O": 8, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "Ca": 20,
     "Ti": 22, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30}

PARAM_NAMES = ["f_pisn", "log10_f_ratio", "log10_tpop2", "M_pisn"]
PRIOR_RANGE = {"f_pisn": (0.0, 1.0), "log10_f_ratio": (-4.0, -1.0),
               "log10_tpop2": (6.5, 7.25), "M_pisn": (150.0, 270.0)}
FD_STEP = {"f_pisn": 0.01, "log10_f_ratio": 0.05, "log10_tpop2": 0.05, "M_pisn": 5.0}
N_SAMPLES = 30

SN_SOURCE = sys.argv[1] if len(sys.argv) > 1 else "Limongi18"


def main():
    forward, pairs = make_forward_fast(sn_source=SN_SOURCE)
    ratio_names = [f"[{x}/{y}]" for x, y in pairs]

    widths = np.array([PRIOR_RANGE[n][1] - PRIOR_RANGE[n][0] for n in PARAM_NAMES])

    rng = np.random.default_rng(0)
    thetas = np.column_stack([
        rng.uniform(0.1, 0.9, N_SAMPLES),
        rng.uniform(-3.5, -1.5, N_SAMPLES),
        rng.uniform(6.6, 7.1, N_SAMPLES),
        rng.uniform(160.0, 260.0, N_SAMPLES),
    ])

    S = np.zeros((len(pairs), 4))
    for th in thetas:
        for k, name in enumerate(PARAM_NAMES):
            hi = th.copy()
            lo = th.copy()
            hi[k] += FD_STEP[name]
            lo[k] -= FD_STEP[name]
            xh = forward(hi)
            xl = forward(lo)
            if not (np.all(np.isfinite(xh)) and np.all(np.isfinite(xl))):
                continue
            S[:, k] += np.abs(xh - xl) / (2.0 * FD_STEP[name]) * widths[k]
    S /= N_SAMPLES

    pisn = S[:, 0]
    nuisance = np.sqrt(S[:, 1] ** 2 + S[:, 2] ** 2 + S[:, 3] ** 2)
    score = pisn / (nuisance + 1e-6)

    cls = []
    for x, y in pairs:
        px = Z[x] % 2
        py = Z[y] % 2
        if px != py:
            cls.append("odd-even")
        elif px == 1:
            cls.append("odd-odd")
        else:
            cls.append("even-even")
    cls = np.array(cls)

    lines = ["Odd-even effect: tracer score by ratio parity.", ""]
    lines.append(f"{'class':<10}{'n_ratios':>9}{'mean score':>12}{'median score':>12}")
    summary = {}
    for c in ["odd-even", "even-even", "odd-odd"]:
        m = cls == c
        summary[c] = (int(m.sum()), float(score[m].mean()), float(np.median(score[m])))
        lines.append(f"{c:<10}{summary[c][0]:>9}{summary[c][1]:>12.2f}{summary[c][2]:>12.2f}")

    # top 10 odd-even ratios
    oe_idx = np.where(cls == "odd-even")[0]
    oe_order = oe_idx[np.argsort(score[oe_idx])[::-1]]
    lines.append("")
    lines.append("Top odd-even ratios by tracer score:")
    for j in oe_order[:10]:
        lines.append(f"  {ratio_names[j]:<10} {score[j]:.2f}")

    with open(os.path.join(OUT_DIR, f"odd_even_{SN_SOURCE}.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines[:4]))
    print("")
    print("Top odd-even ratios:")
    for j in oe_order[:10]:
        print(f"  {ratio_names[j]:<10} {score[j]:.2f}")

    # ---- plot: score distribution by class ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    labels = ["odd-even", "even-even", "odd-odd"]
    means = [summary[c][1] for c in labels]
    ax.bar(labels, means, color=["tab:red", "tab:blue", "tab:green"])
    ax.set_ylabel("mean tracer score")
    ax.set_title("Tracer score by ratio parity")

    ax = axes[1]
    for c, color in zip(labels, ["tab:red", "tab:blue", "tab:green"]):
        m = cls == c
        ax.hist(score[m], bins=20, alpha=0.5, color=color, label=c, density=True)
    ax.set_xlabel("tracer score")
    ax.set_ylabel("density")
    ax.set_title("Score distribution by ratio parity")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"odd_even_{SN_SOURCE}.png"), dpi=130)
    plt.close(fig)

    print(f"\nsaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
