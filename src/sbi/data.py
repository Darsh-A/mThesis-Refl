"""Prepare (theta, x) pairs for simulation-based inference of f_pisn.

The forward model is the Salvadori+2019 PISN+SNII mixture, already wrapped by
``src.randomForest.dataset.generate`` and cached as an .npz (see
``src.randomForest.train_rf``).  Here we load that cache and repackage it into
the (theta, x) convention used by ``sbi``:

    theta : (n, d)  inferred parameters
    x     : (n, m)  observed features

For test v1.0 only f_pisn is inferred (d = 1); the observations are the m=105
pairwise [X/Y] abundance ratios.  The prior on f_pisn is Uniform(0, 1).

To move to multi-parameter inference, add the other simulator diagnostics
(f_ratio, tpop2, pisn_mass) to ``theta`` below -- they are already stored in
the cache.
"""

from __future__ import annotations

import os

import numpy as np
from sklearn.impute import SimpleImputer

SBI_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SBI_DIR)
DEFAULT_CACHE = os.path.join(SRC_DIR, "randomForest", "data", "dataset.npz")


def load_theta_x(cache_path: str = DEFAULT_CACHE, drop_sn_only: bool = True):
    """Load cached simulator output as (theta, x, feature_names).

    Args:
        cache_path: path to the .npz written by ``dataset.generate``.
        drop_sn_only: if True, discard the f_pisn == 0 (pure-SNII) samples so
            the target distribution is continuous on (0, 1].  (The pure-SN
            branch uses a different forward path and puts a point mass at 0,
            which is awkward for a continuous density estimator.)

    Returns:
        theta: (n, 1) float array of f_pisn.
        x: (n, m) float array of imputed abundance ratios.
        feature_names: list[str] of length m.
    """
    data = np.load(cache_path, allow_pickle=True)
    X = np.asarray(data["X"], dtype=float)
    f_pisn = np.asarray(data["y"], dtype=float)
    diag = data["diagnostics"].item()

    mask = np.isfinite(f_pisn)
    if drop_sn_only:
        mask &= f_pisn > 0.0

    X = X[mask]
    f_pisn = f_pisn[mask]

    # Impute NaN ratio cells with the column median (SN-safe, deterministic),
    # matching the existing randomForest training pipeline.
    X = SimpleImputer(strategy="median").fit_transform(X)

    theta = f_pisn.reshape(-1, 1)
    return theta, X, list(diag["feature_names"])


# --- Multi-parameter theta (test v1.1+) -------------------------------------

PARAM_NAMES = ["f_pisn", "log10_f_ratio", "log10_tpop2", "M_pisn"]

# Prior bounds in the same order as PARAM_NAMES.  These match the sampling
# ranges in src.randomForest.dataset (log10 for the two log-uniform priors).
PRIOR_BOUNDS = {
    "f_pisn": (0.0, 1.0),
    "log10_f_ratio": (-4.0, -1.0),
    "log10_tpop2": (6.5, 7.62),
    "M_pisn": (150.0, 270.0),
}


def load_theta_x_multi(
    cache_path: str = DEFAULT_CACHE,
    drop_sn_only: bool = True,
    add_noise: bool = False,
    noise_seed: int = 42,
):
    """Load cached simulator output as multi-parameter (theta, x).

    theta columns (see PARAM_NAMES):
        f_pisn, log10(f_ratio), log10(tpop2), M_pisn.

    The pure-SNII (f_pisn == 0) branch uses a different forward path and leaves
    M_pisn undefined, so those rows are dropped when ``drop_sn_only`` is True.

    Args:
        cache_path: path to the .npz written by ``dataset.generate``.
        drop_sn_only: drop the f_pisn == 0 rows (recommended).
        add_noise: if True, add realistic per-element Gaussian observational
            noise to the ratios via ``dataset.add_dex_noise`` + the Li+22 median
            sigmas (ratios sharing an element stay correlated).
        noise_seed: RNG seed for the noise draws.

    Returns:
        theta: (n, 4) float array.
        x: (n, m) float array of imputed (optionally noised) abundance ratios.
        feature_names: list[str] of length m.
    """
    data = np.load(cache_path, allow_pickle=True)
    X = np.asarray(data["X"], dtype=float)
    diag = data["diagnostics"].item()

    f_pisn = np.asarray(data["y"], dtype=float)
    f_ratio = np.asarray(diag["f_ratio"], dtype=float)
    tpop2 = np.asarray(diag["tpop2"], dtype=float)
    pisn_mass = np.asarray(diag["pisn_mass"], dtype=float)

    mask = (
        np.isfinite(f_pisn)
        & np.isfinite(f_ratio)
        & np.isfinite(tpop2)
        & np.isfinite(pisn_mass)
    )
    if drop_sn_only:
        mask &= f_pisn > 0.0

    X = X[mask]
    f_pisn = f_pisn[mask]
    f_ratio = f_ratio[mask]
    tpop2 = tpop2[mask]
    pisn_mass = pisn_mass[mask]

    X = SimpleImputer(strategy="median").fit_transform(X)

    if add_noise:
        from src.randomForest.dataset import (
            _feature_pairs,
            add_dex_noise,
            li_plus_element_sigmas,
        )

        sigmas = li_plus_element_sigmas()
        rng = np.random.default_rng(noise_seed)
        X = add_dex_noise(X.copy(), _feature_pairs(), sigmas, rng)

    theta = np.column_stack([f_pisn, np.log10(f_ratio), np.log10(tpop2), pisn_mass])
    return theta, X, list(diag["feature_names"])
