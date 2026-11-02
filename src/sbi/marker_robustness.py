"""Rank PISN markers by robustness (test v1.9).

A "robust marker" is an abundance ratio/element that is a strong f_pisn tracer
in its WORST condition -- the minimum tracer score across both SN yield models
(Limongi18, Nomoto13) and across the PISN progenitor mass grid (150-270 Msun).
This surfaces the markers that don't depend on a lucky yield/mass choice.

Uses the forward-model sensitivity (signal = |d[X/Y]/d f_pisn| scaled by the
prior width; nuisance = sqrt of the same for f_ratio, tpop2, M_pisn).

Outputs (src/sbi/results/testv1.9/):
    marker_robustness.txt
    marker_robustness.png
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
OUT_DIR = os.path.join(HERE, "results", "testv1.9")
os.makedirs(OUT_DIR, exist_ok=True)

YIELD_MODELS = ["Limongi18", "Nomoto13"]
MASSES = [150.0, 190.0, 230.0, 270.0]

PARAM_NAMES = ["f_pisn", "log10_f_ratio", "log10_tpop2", "M_pisn"]
WIDTHS = np.array([1.0, 3.0, 1.12, 120.0])  # prior widths
FD_STEP = {"f_pisn": 0.01, "log10_f_ratio": 0.05, "log10_tpop2": 0.05, "M_pisn": 5.0}
N_SAMPLES = 20


def ratio_scores(sn_source: str, m_pisn: float):
    forward, pairs = make_forward_fast(sn_source=sn_source)
    rng = np.random.default_rng(0)
    thetas = np.column_stack([
        rng.uniform(0.1, 0.9, N_SAMPLES),
        rng.uniform(-3.5, -1.5, N_SAMPLES),
        rng.uniform(6.6, 7.4, N_SAMPLES),
        np.full(N_SAMPLES, m_pisn),
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
            S[:, k] += np.abs(xh - xl) / (2.0 * FD_STEP[name]) * WIDTHS[k]
    S /= N_SAMPLES
    signal = S[:, 0]
    nuisance = np.sqrt(S[:, 1] ** 2 + S[:, 2] ** 2 + S[:, 3] ** 2)
    return signal, nuisance, pairs


def main():
    ratio_names = [f"[{x}/{y}]" for x, y in _feature_pairs()]
    elements = sorted({e for p in _feature_pairs() for e in p})

    signals = {}
    nuisances = {}
    for ym in YIELD_MODELS:
        for m in MASSES:
            sig, nui, _ = ratio_scores(ym, m)
            signals[(ym, m)] = sig
            nuisances[(ym, m)] = nui

    combos = list(signals.keys())

    # ratio robust score = min score over all (yield, mass) combos
    ratio_robust = np.min([
        signals[c] / (nuisances[c] + 1e-6) for c in combos
    ], axis=0)

    # element robust score = min over combos of (sum_signal / sum_nuisance)
    elem_robust = {}
    for e in elements:
        idx = [j for j, (x, y) in enumerate(_feature_pairs()) if e in (x, y)]
        best = np.min([
            signals[c][idx].sum() / (nuisances[c][idx].sum() + 1e-6) for c in combos
        ])
        elem_robust[e] = float(best)

    # ---- text ----
    lines = ["PISN marker robustness (min tracer score across yield models x masses).",
             "signal = |d[X/Y]/d f_pisn| over prior width; nuisance = f_ratio/tpop2/M_pisn.",
             "Robust score > 1 means signal dominates nuisance in the WORST case.", ""]

    order = np.argsort(ratio_robust)[::-1]
    lines.append("Top robust ratios:")
    for j in order[:15]:
        lines.append(f"  {ratio_names[j]:<10} robust={ratio_robust[j]:.2f}")

    eorder = sorted(elements, key=lambda e: -elem_robust[e])
    lines.append("")
    lines.append("Element robust score:")
    for e in eorder:
        lines.append(f"  {e:<4} {elem_robust[e]:.2f}")

    with open(os.path.join(OUT_DIR, "marker_robustness.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))

    # ---- plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.bar(np.arange(len(eorder)), [elem_robust[e] for e in eorder], color="tab:blue")
    ax.set_xticks(np.arange(len(eorder)))
    ax.set_xticklabels(eorder)
    ax.axhline(1.0, color="k", ls="--", lw=1, label="signal = nuisance")
    ax.set_ylabel("robust tracer score (min across yield x mass)")
    ax.set_title("Element-level robust PISN markers")
    ax.legend(fontsize=8)

    ax = axes[1]
    top = order[:12]
    ax.barh(np.arange(len(top))[::-1], ratio_robust[top][::-1], color="tab:green")
    ax.set_yticks(np.arange(len(top))[::-1])
    ax.set_yticklabels([ratio_names[j] for j in top][::-1], fontsize=9)
    ax.set_xlabel("robust tracer score")
    ax.set_title("Top robust abundance ratios")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "marker_robustness.png"), dpi=130)
    plt.close(fig)

    print(f"\nsaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
