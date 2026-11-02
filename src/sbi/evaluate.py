"""Evaluate the trained NPE posterior on held-out data (test v1.0).

Loads the posterior + artifacts saved by train.py and reports:

    1. posterior median vs true f_pisn  (R2 / RMSE / Spearman rho)
    2. empirical coverage of central credible intervals (calibration)
    3. a reliability diagram (nominal vs empirical coverage)
    4. example 1-D posteriors for a few held-out stars

Run:
    uv run python src/sbi/evaluate.py

Plots -> src/sbi/results/testv1.0/
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results", "testv1.0")

N_POSTERIOR_SAMPLES = 2000
NOMINAL = np.arange(0.1, 1.0, 0.05)


def _central_interval(samples: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """samples: (n_samples, n_test) -> (lo, hi) each (n_test,)."""
    lo = (1.0 - level) / 2.0
    hi = (1.0 + level) / 2.0
    q = np.quantile(samples, [lo, hi], axis=0)
    return q[0], q[1]


def main():
    import joblib

    posterior = joblib.load(os.path.join(OUT_DIR, "npe_posterior.pkl"))
    art = np.load(os.path.join(OUT_DIR, "train_artifacts.npz"), allow_pickle=True)
    x_te = art["x_te"]
    th_te = art["th_te"].reshape(-1)
    n_test = len(th_te)

    import torch

    x_te_t = torch.from_numpy(x_te)
    samples = posterior.sample_batched((N_POSTERIOR_SAMPLES,), x=x_te_t).numpy()
    # (n_samples, n_test, d=1) -> (n_samples, n_test)
    samples = samples.reshape(N_POSTERIOR_SAMPLES, n_test)

    med = np.median(samples, axis=0)
    lo90, hi90 = _central_interval(samples, 0.90)

    from scipy.stats import spearmanr
    from sklearn.metrics import r2_score, mean_squared_error

    r2 = float(r2_score(th_te, med))
    rmse = float(np.sqrt(mean_squared_error(th_te, med)))
    rho, _ = spearmanr(th_te, med)

    print("=== NPE posterior (median) vs truth ===")
    print(f"  R2    = {r2:.4f}")
    print(f"  RMSE  = {rmse:.4f}")
    print(f"  rho   = {rho:+.4f}")
    print(f"  bias  = {float(np.mean(med - th_te)):+.4f}")

    print("\n=== Central credible interval coverage (held-out) ===")
    coverage_rows = []
    for level in NOMINAL:
        lo, hi = _central_interval(samples, level)
        in_int = (th_te >= lo) & (th_te <= hi)
        cov = float(in_int.mean())
        width = float(np.mean(hi - lo))
        coverage_rows.append((level, cov, width))
        print(f"  {level*100:>5.0f}% CI: coverage={cov*100:6.2f}%  mean width={width:.4f}")

    # ---- scatter: true vs posterior median with 90% CI ----
    fig, ax = plt.subplots(figsize=(6, 6))
    order = np.argsort(th_te)
    ax.errorbar(
        th_te[order], med[order],
        yerr=[med[order] - lo90[order], hi90[order] - med[order]],
        fmt="o", ms=4, alpha=0.6, color="tab:blue", capsize=0, lw=0.8,
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("true f_pisn")
    ax.set_ylabel("posterior median f_pisn")
    ax.set_title(f"NPE v1.0: held-out  (R2={r2:.3f}, rho={rho:+.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "npe_posterior_scatter.png"), dpi=130)
    plt.close(fig)

    # ---- reliability diagram ----
    nominal = np.array([r[0] for r in coverage_rows])
    empirical = np.array([r[1] for r in coverage_rows])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.plot(nominal, empirical, "o-", color="tab:blue", label="NPE")
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title("NPE v1.0: reliability diagram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "npe_reliability.png"), dpi=130)
    plt.close(fig)

    # ---- example posteriors ----
    n_ex = min(4, n_test)
    fig, axes = plt.subplots(1, n_ex, figsize=(4 * n_ex, 3.5), squeeze=False)
    for k, ax in enumerate(axes[0]):
        ax.hist(samples[:, k], bins=40, range=(0, 1), color="tab:green",
                alpha=0.8, density=True)
        ax.axvline(th_te[k], color="crimson", lw=2, label="true")
        ax.set_xlabel("f_pisn")
        ax.set_title(f"star {k}  (true={th_te[k]:.2f})", fontsize=9)
        if k == 0:
            ax.set_ylabel("posterior density")
            ax.legend(fontsize=8)
    fig.suptitle("NPE v1.0: example 1-D posteriors (held-out)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "npe_example_posteriors.png"), dpi=130)
    plt.close(fig)

    print(f"\nsaved plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
