"""Locate the posterior leak (test v1.7).

Trains ONE 4-D NPE and, for the complete observed stars, reports the fraction
of posterior samples outside the prior support *per theta dimension*.  If the
nuisance parameters leak, the f_pisn marginal is contaminated too (the joint
density is wrong), so the result is untrustworthy regardless of how tight the
ensemble spread is.

Run:
    uv run python src/sbi/verify_fpisn_leak.py [Limongi18|Nomoto13]
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

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
OUT_DIR = os.path.join(HERE, "results", "testv1.7")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    sn_source = sys.argv[1] if len(sys.argv) > 1 else "Limongi18"

    theta, Xs, scaler, imp = load_training(sn_source)
    posterior = train_posterior(theta, Xs, seed=0)

    stars = load_obs_li()
    X_obs, feh = build_obs_features(stars)
    complete = np.all(np.isfinite(X_obs), axis=1)
    stars = [s for s, c in zip(stars, complete) if c]
    X_obs = imp.transform(X_obs[complete])
    X_obs_s = scaler.transform(X_obs).astype(np.float32)

    import torch

    leak = np.zeros(4)
    total = 0
    f_pisn_lo, f_pisn_hi = np.inf, -np.inf
    for start in range(0, len(stars), CHUNK):
        end = min(start + CHUNK, len(stars))
        x_chunk = torch.from_numpy(X_obs_s[start:end])
        s = posterior.sample_batched(
            (N_SAMPLES,), x=x_chunk, reject_outside_prior=False
        ).detach().numpy()  # (N, chunk, 4)
        for k, name in enumerate(PARAM_NAMES):
            lo, hi = PRIOR_BOUNDS[name]
            leak[k] += int((s[..., k] < lo).sum() + (s[..., k] > hi).sum())
        total += s.shape[0] * s.shape[1]
        f_pisn_lo = min(f_pisn_lo, float(s[..., 0].min()))
        f_pisn_hi = max(f_pisn_hi, float(s[..., 0].max()))

    print(f"\n=== {sn_source}: posterior leak by dimension "
          f"({len(stars)} complete stars) ===")
    for k, name in enumerate(PARAM_NAMES):
        pct = 100.0 * leak[k] / total
        print(f"  {name:<14} {pct:5.1f}% outside prior")
    print(f"  f_pisn sample range: [{f_pisn_lo:.3f}, {f_pisn_hi:.3f}] "
          f"(prior [0, 1])")

    # how many stars have NO f_pisn leakage (f_pisn samples all in [0,1])?
    clean = 0
    for start in range(0, len(stars), CHUNK):
        end = min(start + CHUNK, len(stars))
        x_chunk = torch.from_numpy(X_obs_s[start:end])
        s = posterior.sample_batched(
            (N_SAMPLES,), x=x_chunk, reject_outside_prior=False
        ).detach().numpy()
        fp = s[..., 0]
        in_bounds = (fp >= 0.0) & (fp <= 1.0)
        clean += int(np.all(in_bounds, axis=0).sum())
    print(f"  stars with ALL f_pisn samples in [0,1]: {clean}/{len(stars)}")


if __name__ == "__main__":
    main()
