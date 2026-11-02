"""Diagnostic + result plots for the observed Li+22 application (test v1.8).

Trains ONE 4-D NPE (Limongi18, tpop42) and produces:
    result_fpisn.png      P(f_pisn>0.5) + f_pisn median vs [Fe/H]
    diagnostic_leak.png   per-dimension posterior leak + tpop2 posterior
    corner_example.png     f_pisn vs log10 f_ratio for the highest-P star

Run:
    uv run python src/sbi/diagnostics_plots.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.sbi.apply_obs_ensemble import (  # noqa: E402
    CHUNK,
    N_SAMPLES,
    PARAM_NAMES,
    PRIOR_BOUNDS,
    build_obs_features,
    load_training,
    train_posterior,
)
from src.load_data import load_obs_li  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results", "testv1.8")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    import torch

    sn_source = "Limongi18"
    theta, Xs, scaler, imp = load_training(sn_source)
    posterior = train_posterior(theta, Xs, seed=0)

    stars = load_obs_li()
    X_obs, feh = build_obs_features(stars)
    complete = np.all(np.isfinite(X_obs), axis=1)
    stars = [s for s, c in zip(stars, complete) if c]
    feh = feh[complete]
    X_obs_s = scaler.transform(imp.transform(X_obs[complete])).astype(np.float32)

    # sample the full 4-D posterior for all stars (chunked)
    chunks = []
    for start in range(0, len(stars), CHUNK):
        end = min(start + CHUNK, len(stars))
        s = posterior.sample_batched(
            (N_SAMPLES,), x=torch.from_numpy(X_obs_s[start:end]),
            reject_outside_prior=False,
        ).detach().numpy()  # (N, chunk, 4)
        chunks.append(s)
    samples = np.concatenate(chunks, axis=1)  # (N, n_stars, 4)

    f_pisn = samples[..., 0]
    tpop2 = 10.0 ** samples[..., 2] / 1e6  # Myr
    p_gt05 = np.mean(f_pisn > 0.5, axis=0)
    f_pisn_med = np.median(f_pisn, axis=0)
    tpop2_med = np.median(tpop2, axis=0)

    # ---- Figure 1: result ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    ax.hist(p_gt05, bins=25, range=(0, 1), color="tab:blue", alpha=0.85)
    ax.axvline(0.5, color="k", ls="--", lw=1, label="P = 0.5")
    ax.set_xlabel("P(f_pisn > 0.5)")
    ax.set_ylabel("N stars")
    ax.set_title(f"PISN-dominated probability (N={len(stars)})")
    ax.legend()

    ax = axes[1]
    ax.hist(f_pisn_med, bins=25, range=(0, 1), color="tab:green", alpha=0.85)
    ax.axvline(0.5, color="k", ls="--", lw=1, label="f_pisn = 0.5")
    ax.set_xlabel("f_pisn (posterior median)")
    ax.set_ylabel("N stars")
    ax.set_title("f_pisn posterior median")
    ax.legend()

    ax = axes[2]
    ax.scatter(feh, f_pisn_med, s=15, alpha=0.6, color="tab:red")
    ax.set_xlabel("[Fe/H]")
    ax.set_ylabel("f_pisn (posterior median)")
    ax.set_title("f_pisn vs [Fe/H]")
    fig.suptitle(f"Li+22 observed stars, {sn_source} (tpop2 ∈ [3.2, 42] Myr)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "result_fpisn.png"), dpi=130)
    plt.close(fig)

    # ---- Figure 2: diagnostics (leak + tpop2) ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    leaks = []
    for k, name in enumerate(PARAM_NAMES):
        lo, hi = PRIOR_BOUNDS[name]
        pct = 100.0 * ((samples[..., k] < lo) | (samples[..., k] > hi)).mean()
        leaks.append(pct)
    ax.bar(PARAM_NAMES, leaks, color="tab:orange")
    ax.set_ylabel("% samples outside prior")
    ax.set_title("Posterior leak by dimension")

    ax = axes[1]
    ax.hist(tpop2_med, bins=25, color="tab:purple", alpha=0.85)
    ax.axvline(3.2, color="k", ls="--", lw=1)
    ax.axvline(42, color="k", ls="--", lw=1, label="prior [3.2, 42] Myr")
    ax.set_xlabel("tpop2 (posterior median, Myr)")
    ax.set_ylabel("N stars")
    ax.set_title("tpop2 posterior")
    ax.legend()

    ax = axes[2]
    ax.hist(tpop2.flatten(), bins=50, color="tab:cyan", alpha=0.6, density=True)
    ax.axvline(3.2, color="k", ls="--", lw=1)
    ax.axvline(42, color="k", ls="--", lw=1, label="prior [3.2, 42] Myr")
    ax.set_xlabel("tpop2 (all samples, Myr)")
    ax.set_ylabel("density")
    ax.set_title("tpop2 posterior (all stars)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "diagnostic_leak.png"), dpi=130)
    plt.close(fig)

    # ---- Figure 3: corner for the highest-P star ----
    i = int(np.argmax(p_gt05))
    fp = f_pisn[:, i]
    fr = samples[:, i, 1]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].hist(fp, bins=40, range=(0, 1), color="tab:blue", alpha=0.8, density=True)
    axes[0].set_xlabel("f_pisn")
    axes[0].set_title(f"highest-P star {stars[i]['label']}")
    hb = axes[1].hexbin(fr, fp, gridsize=40, cmap="viridis", mincnt=1)
    axes[1].set_xlabel("log10 f_ratio")
    axes[1].set_ylabel("f_pisn")
    axes[1].set_title("joint P(f_pisn, log10 f_ratio)")
    fig.colorbar(hb, ax=axes[1])
    fig.suptitle(f"Posterior for {stars[i]['label']} ([Fe/H]={feh[i]:.2f}, "
                 f"P(f_pisn>0.5)={p_gt05[i]:.2f})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "corner_example.png"), dpi=130)
    plt.close(fig)

    print(f"saved plots to {OUT_DIR}")
    print(f"n_stars={len(stars)}  P(f_pisn>0.5)>0.9: {(p_gt05>0.9).sum()}  "
          f"max P={p_gt05.max():.3f}")


if __name__ == "__main__":
    main()
