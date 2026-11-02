"""Non-ML (classical) inference of f_pisn, compared against the NPE.

Two classical methods, both using the *fast* forward model directly (no neural
net, no emulator):

    1. 4-D MAP optimization (scipy L-BFGS-B) over
       (f_pisn, log10 f_ratio, log10 tpop2, M_pisn) for a few held-out stars.
    2. A short emcee MCMC over the same 4 parameters for one held-out star,
       giving an actual non-ML posterior.

The MCMC f_pisn marginal is overlaid on the NPE (test v1.1) f_pisn marginal for
the same star.  Uses the same noisy data as test v1.1.

Run:
    uv run python src/sbi/nonml.py

Outputs -> src/sbi/results/testv1.3/
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scipy.optimize import minimize  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from src.sbi.data import load_theta_x_multi  # noqa: E402
from src.sbi.forward import make_forward_fast  # noqa: E402
from src.randomForest.dataset import _feature_pairs, li_plus_element_sigmas  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results", "testv1.3")
os.makedirs(OUT_DIR, exist_ok=True)

N_POINTS = 3
SEED = 0
BOUNDS = [(0.001, 0.999), (-4.0, -1.0), (6.5, 7.25), (150.0, 270.0)]
PRIOR_CENTER = np.array([0.5, -2.5, 6.875, 210.0])
MCMC_NWALKERS = 16
MCMC_NSTEPS = 400
MCMC_BURNIN = 80
N_NPE_SAMPLES = 2000


def neg_log_like(theta, forward, x_obs, sigma):
    x_model = forward(theta)
    if not np.all(np.isfinite(x_model)):
        return 1e6  # large penalty for numerically invalid regions
    resid = (x_obs - x_model) / sigma
    return 0.5 * np.sum(resid ** 2)


def log_posterior(theta, forward, x_obs, sigma):
    f, lf, lt, m = theta
    if not (0.0 < f < 1.0 and -4.0 < lf < -1.0 and 6.5 < lt < 7.25 and 150.0 < m < 270.0):
        return -np.inf
    return -neg_log_like(theta, forward, x_obs, sigma)


def main():
    theta, x, _ = load_theta_x_multi(add_noise=True, noise_seed=42)
    forward, pairs = make_forward_fast()

    el_sig = li_plus_element_sigmas()
    sigma = np.array([
        np.sqrt(el_sig.get(a, 0.05) ** 2 + el_sig.get(b, 0.05) ** 2)
        for a, b in pairs
    ])

    _, x_te, _, th_te = train_test_split(
        x, theta, test_size=0.2, random_state=SEED
    )

    print("=== 4-D MAP fit (non-ML) ===")
    print(f"{'pt':<4}{'true f_pisn':>12}{'MAP f_pisn':>12}{'true log10 fr':>14}"
          f"{'MAP log10 fr':>14}{'true M':>9}{'MAP M':>9}{'evals':>7}")
    for k in range(N_POINTS):
        x_obs, th_true = x_te[k], th_te[k]
        res = minimize(
            neg_log_like, PRIOR_CENTER, args=(forward, x_obs, sigma),
            method="L-BFGS-B", bounds=BOUNDS,
        )
        print(f"{k:<4}{th_true[0]:>12.3f}{res.x[0]:>12.3f}{th_true[1]:>14.2f}"
              f"{res.x[1]:>14.2f}{th_true[3]:>9.0f}{res.x[3]:>9.0f}{res.nfev:>7}")

    # ---- MCMC for the first held-out point ----
    import emcee

    x_obs, th_true = x_te[0], th_te[0]
    res0 = minimize(
        neg_log_like, PRIOR_CENTER, args=(forward, x_obs, sigma),
        method="L-BFGS-B", bounds=BOUNDS,
    )
    rng = np.random.default_rng(0)
    p0 = res0.x + 0.02 * rng.normal(size=(MCMC_NWALKERS, 4))
    sampler = emcee.EnsembleSampler(
        MCMC_NWALKERS, 4, log_posterior, args=(forward, x_obs, sigma)
    )
    sampler.run_mcmc(p0, MCMC_NSTEPS, progress=False)
    chain = sampler.get_chain(discard=MCMC_BURNIN, flat=True)  # (n, 4)

    f_pisn_mcmc = chain[:, 0]
    print(f"\n=== MCMC f_pisn posterior (pt 0, true={th_true[0]:.3f}) ===")
    print(f"  median = {np.median(f_pisn_mcmc):.3f}  "
          f"90% CI = [{np.percentile(f_pisn_mcmc, 5):.3f}, "
          f"{np.percentile(f_pisn_mcmc, 95):.3f}]")

    # ---- NPE marginal for the same star ----
    import joblib
    import torch

    post = joblib.load(os.path.join(HERE, "results", "testv1.1", "npe_posterior.pkl"))
    art = np.load(os.path.join(HERE, "results", "testv1.1", "train_artifacts.npz"),
                  allow_pickle=True)
    x_obs_z = (x_obs - art["scaler_mean"]) / art["scaler_scale"]
    npe_s = post.sample_batched(
        (N_NPE_SAMPLES,),
        x=torch.tensor(x_obs_z[None].astype(np.float32)),
        reject_outside_prior=False,
    ).detach().numpy()
    f_pisn_npe = np.clip(npe_s[..., 0].ravel(), 0.0, 1.0)
    print(f"  NPE  median = {np.median(f_pisn_npe):.3f}  "
          f"90% CI = [{np.percentile(f_pisn_npe, 5):.3f}, "
          f"{np.percentile(f_pisn_npe, 95):.3f}]")

    # ---- overlay plot ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.hist(f_pisn_mcmc, bins=40, range=(0, 1), color="tab:blue",
            alpha=0.5, density=True, label="MCMC (non-ML)")
    ax.hist(f_pisn_npe, bins=40, range=(0, 1), color="tab:orange",
            alpha=0.5, density=True, label="NPE")
    ax.axvline(th_true[0], color="crimson", lw=2, label="true")
    ax.set_xlabel("f_pisn")
    ax.set_ylabel("posterior density")
    ax.set_title("f_pisn: MCMC vs NPE (held-out star 0)")
    ax.legend(fontsize=8)

    ax = axes[1]
    hb = ax.hexbin(chain[:, 1], chain[:, 0], gridsize=40, cmap="viridis", mincnt=1)
    ax.axvline(th_true[1], color="crimson", lw=1.5, ls="--")
    ax.axhline(th_true[0], color="crimson", lw=1.5, ls="--")
    ax.set_xlabel("log10 f_ratio")
    ax.set_ylabel("f_pisn")
    ax.set_title("MCMC joint posterior P(f_pisn, log10 f_ratio)")
    fig.colorbar(hb, ax=ax, label="density")
    fig.suptitle("Non-ML vs ML inference (test v1.3)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "nonml_vs_npe.png"), dpi=130)
    plt.close(fig)

    print(f"\nsaved plot to {OUT_DIR}")


if __name__ == "__main__":
    main()
