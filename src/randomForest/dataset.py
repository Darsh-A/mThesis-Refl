"""Generate training datasets for the f_pisn regression. 

Each sample picks a random PISN entry (via the continuous-mass interpolator)
and a fixed SN source (IMF-integrated; single_sn=False, auto_sn=False), then
computes the abundance ratios used as model features.

Features:
    - [X/Y] for every unordered pair (X, Y) of elements in filter_elements.
      This is C(n, 2) ratios where n = len(filter_elements); H/He are excluded.

Target:
    - f_pisn (drawn ~ [0.1, 1])

Diagnostics (not features, stored for later inspection):
    - f_ratio  (~ loguniform 1e-4 .. 1e-1)
    - tpop2    (~ loguniform 3.2e6 .. 17.4e6)
    - pisn_mass (drawn ~ [150, 270]) -- optionally added as a feature later
      to test whether the ratios alone encode PISN mass.

Per-source routing:
    - Limongi18 / Nomoto13: element-level yields (pass entries directly).
    - WW95: isotope-level raw yields (extract applies decay + combine).
"""

import os
from itertools import combinations

import numpy as np
from scipy.stats import loguniform

from params import filter_elements
from src.load_data import load_hw2002, load_takahashi, load_ww95, load_limongi18, load_nomoto13
from src.salvadori_funcs import (
    salvadori_yields,
    salvadori_combined_abundratio,
)
from src.utils import build_pisn_interpolator
from src.yield_sources import get_source

# --- Parameter priors ---
F_PISN_RANGE = (0.1, 1.0)
F_RATIO_RANGE = (1e-4, 1e-1)
TPOP2_RANGE = (3.2e6, 17.4e6)
PISN_MASS_RANGE = (150.0, 270.0)

# Clamp floor for abundance ratios
RATIO_FLOOR = -5.0

# Map source name -> element-level vs isotope-level handling of its yield
# entries. Element-level sources are passed to salvadori_Y_X_II directly;
# isotope-level sources (WW95) rely on source.extract to apply radioactive
# decay + combine isotopes.
ELEMENT_LEVEL_SOURCES = {"Limongi18", "Nomoto13"}
ISOTOPE_LEVEL_SOURCES = {"WW95"}


def _load_pisn(source: str) -> tuple[list[dict], object]:
    """Load PISN entries and return (entries, interpolator).

    The interpolator maps a total mass -> a yield entry dict (see
    build_pisn_interpolator).
    """
    if source == "HW2002":
        entries = load_hw2002()
        entries = entries[1:]
    elif source == "Takahashi":
        entries = load_takahashi()["NR"]
    else:
        raise ValueError(f"Unrecognized pisn_source: {source!r}")

    sal_pisn = salvadori_yields(entries)
    interp = build_pisn_interpolator(sal_pisn)
    return entries, interp


def _load_sn(source: str) -> tuple[list[dict], object]:
    """Load SN entries and the matching YieldSource instance."""
    get_source(source)
    if source == "Limongi18":
        entries = load_limongi18()
        entries = [e for e in entries if e["params"]["velocity"] == 0]
    elif source == "WW95":
        entries = load_ww95()
    elif source == "Nomoto13":
        entries = load_nomoto13()
    else:
        raise ValueError(f"Unrecognized sn_source: {source!r}")

    return entries, get_source(source)


def _abundance_elements() -> list[str]:
    """Elements used to build pairwise abundance-ratio features.
    """
    # return [e for e in filter_elements if e not in ("H", "He")]
    return ["C", "O", "Na", "Mg", "Al", "Si", "Ca", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"]


def _feature_pairs() -> list[tuple[str, str]]:
    """All unordered (X, Y) element pairs -> [X/Y] features."""
    return list(combinations(_abundance_elements(), 2))


def _feature_names() -> list[str]:
    """Feature column names: [X/Y] for each element pair."""
    return [f"[{x}/{y}]" for x, y in _feature_pairs()]


def generate(
    pisn_source: str,
    sn_source: str,
    n_samples: int,
    seed: int,
    cache_path: str = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build (X, y, diagnostics) for f_pisn regression.

    Args:
        pisn_source: "HW2002" or "Takahashi".
        sn_source: "Limongi18", "WW95", or "Nomoto13".
        n_samples: number of samples to draw.
        seed: RNG seed for reproducibility.
        cache_path: if set, load from / save to this .npz instead of recomputing.

    Returns:
        X: (n_samples, n_features) float array of abundance ratios.
        y: (n_samples,) float array of f_pisn targets.
        diagnostics: dict with keys f_ratio, tpop2, pisn_mass (each length n_samples)
                     and feature_names.
    """
    rng = np.random.default_rng(seed)

    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        return data["X"], data["y"], dict(data["diagnostics"].item())

    pisn_entries, pisn_interp = _load_pisn(pisn_source)
    sn_entries, sn_source_obj = _load_sn(sn_source)

    # Precomputed transforms (done once, outside the sample loop).
    salv_sn_yields = salvadori_yields(sn_entries)

    pairs = _feature_pairs()
    n_features = len(pairs)

    X = np.full((n_samples, n_features), np.nan)
    y = np.empty(n_samples)
    f_ratio_diag = np.empty(n_samples)
    tpop2_diag = np.empty(n_samples)
    pisn_mass_diag = np.empty(n_samples)

    for i in range(n_samples):
        f_pisn = rng.uniform(*F_PISN_RANGE)
        f_ratio = loguniform.rvs(*F_RATIO_RANGE, random_state=rng)
        tpop2 = loguniform.rvs(*TPOP2_RANGE, random_state=rng)
        pisn_mass = rng.uniform(*PISN_MASS_RANGE)

        pisn_entry = pisn_interp(pisn_mass)

        features = []
        for x_elem, y_elem in pairs:
            ratio = salvadori_combined_abundratio(
                x_elem, x_elem, y_elem, y_elem,
                pisn_data=pisn_entry,
                sn_data=sn_entries,
                salv_sn_data=salv_sn_yields,
                auto_sn=False, single_sn=False,
                sn_input=sn_source_obj,
                f_pisn=f_pisn, f_ratio=f_ratio, tpop2=tpop2,
            )
            features.append(ratio)

        X[i] = np.asarray(features, dtype=float)
        y[i] = f_pisn
        f_ratio_diag[i] = f_ratio
        tpop2_diag[i] = tpop2
        pisn_mass_diag[i] = pisn_mass

    # Hygiene: clamp extreme low ratios and replace non-finite with NaN
    # (trainer will handle NaNs via mask/drop).
    valid = np.isfinite(X)
    X = np.where(valid & (X < RATIO_FLOOR), RATIO_FLOOR, X)
    X = np.where(np.isfinite(X), X, np.nan)

    diagnostics = {
        "f_ratio": f_ratio_diag,
        "tpop2": tpop2_diag,
        "pisn_mass": pisn_mass_diag,
        "feature_names": _feature_names(),
        "pisn_source": pisn_source,
        "sn_source": sn_source,
        "n_samples": n_samples,
        "seed": seed,
    }

    if cache_path:
        np.savez_compressed(
            cache_path,
            X=X,
            y=y,
            diagnostics=diagnostics,
        )

    return X, y, diagnostics


__all__ = ["generate"]
