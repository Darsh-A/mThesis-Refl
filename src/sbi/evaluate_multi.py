"""Evaluate the multi-parameter NPE posterior on held-out data (test v1.1).

Loads the posterior + artifacts saved by train_multi.py and reports, per
parameter (f_pisn, log10 f_ratio, log10 tpop2, M_pisn):

    1. posterior median vs true value  (R2 / RMSE / Spearman rho)
    2. marginal credible-interval coverage (calibration)
    3. recovery scatter (4-panel)
    4. joint posterior of (f_pisn, log10 f_ratio) for an uncertain test point,
       revealing the f_pisn -- f_ratio degeneracy

Run:
    uv run python src/sbi/evaluate_multi.py

Plots -> src/sbi/results/testv1.1/
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.sbi.data import PARAM_NAMES  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results", "testv1.1")

N_POSTERIOR_SAMPLES = 4000
LEVELS = [0.5, 0.68, 0.9]


def main():
    import joblib

    posterior = joblib.load(os.path.join(OUT_DIR, "npe_posterior.pkl"))
    art = np.load(os.path.join(OUT_DIR, "train_artifacts.npz"), allow_pickle=True)
    x_te = art["x_te"]
    th_te = art["th_te"]  # (n_test, 4)
    n_test, d = th_te.shape

    import torch

    x_te_t = torch.from_numpy(x_te)
    samples = posterior.sample_batched((N_POSTERIOR_SAMPLES,), x=x_te_t).numpy()
    samples = samples.reshape(N_POSTERIOR_SAMPLES, n_test, d)  # (N, n_test, d)

    from scipy.stats import spearmanr
    from sklearn.metrics import r2_score, mean_squared_error

    med = np.median(samples, axis=0)  # (n_test, d)

    print("=== per-parameter posterior median vs truth ===")
    for j, name in enumerate(PARAM_NAMES):
        y = th_te[:, j]
        p = med[:, j]
        r2 = r2_score(y, p)
        rmse = float(np.sqrt(mean_squared_error(y, p)))
        rho, _ = spearmanr(y, p)
        print(
            f"  {name:<14} R2={r2:.4f}  RMSE={rmse:.4f}  rho={rho:+.4f}  "
            f"bias={float(np.mean(p - y)):+.4f}"
        )

    print("\n=== marginal credible-interval coverage (held-out) ===")
    for j, name in enumerate(PARAM_NAMES):
        y = th_te[:, j]
        cov_str = []
        for level in LEVELS:
            lo_q = (1.0 - level) / 2.0
            hi_q = (1.0 + level) / 2.0
            lo, hi = np.quantile(samples[:, :, j], [lo_q, hi_q], axis=0)
            in_int = (y >= lo) & (y <= hi)
            cov_str.append(f"{level*100:>3.0f}%:{float(in_int.mean())*100:5.1f}%")
        print(f"  {name:<14} " + "  ".join(cov_str))

    # ---- 4-panel recovery scatter ----
    fig, axes = plt.subplots(1, d, figsize=(4 * d, 3.5))
    for j, (ax, name) in enumerate(zip(axes, PARAM_NAMES)):
        y = th_te[:, j]
        p = med[:, j]
        r2 = r2_score(y, p)
        ax.scatter(y, p, s=8, alpha=0.5, color="tab:blue")
        ax.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=1)
        ax.set_xlabel(f"true {name}")
        ax.set_ylabel(f"posterior median {name}")
        ax.set_title(f"{name}  R2={r2:.3f}", fontsize=10)
    fig.suptitle("NPE v1.1: parameter recovery (held-out, with noise)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "npe_multi_recovery.png"), dpi=130)
    plt.close(fig)

    # ---- degeneracy: joint (f_pisn, log10 f_ratio) posterior ----
    # pick the test point whose f_pisn posterior is most uncertain
    f_pisn_width = np.ptp(np.quantile(samples[:, :, 0], [0.05, 0.95], axis=0), axis=0)
    idx_wide = int(np.argmax(f_pisn_width))
    idx_narrow = int(np.argmin(f_pisn_width))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, idx, tag in [(axes[0], idx_wide, "wide"), (axes[1], idx_narrow, "narrow")]:
        fp = samples[:, idx, 0]
        fr = samples[:, idx, 1]
        hb = ax.hexbin(fr, fp, gridsize=40, cmap="viridis", mincnt=1)
        ax.axvline(th_te[idx, 1], color="crimson", lw=1.5, ls="--", label="true")
        ax.axhline(th_te[idx, 0], color="crimson", lw=1.5, ls="--")
        ax.set_xlabel("log10 f_ratio")
        ax.set_ylabel("f_pisn")
        ax.set_title(f"{tag} f_pisn posterior (star {idx})", fontsize=10)
        fig.colorbar(hb, ax=ax, label="posterior density")
    fig.suptitle("NPE v1.1: joint posterior P(f_pisn, log10 f_ratio)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "npe_degeneracy.png"), dpi=130)
    plt.close(fig)

    print(f"\nsaved plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
