"""Regenerate training datasets with the extended tpop2 range.

The tpop2 prior is now [3.2, 42] Myr (down to the ~8 Msun CCSN lower mass
limit) instead of [3.2, 17.4] Myr.  This regenerates the Limongi18 and Nomoto13
datasets so the 8-13 Msun CCSNe are included (constant-extrapolated below the
13 Msun lightest grid model).

Run (slow: ~1-2 h for both):
    uv run python src/sbi/regenerate_data.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.randomForest.dataset import (  # noqa: E402
    _feature_pairs,
    add_dex_noise,
    generate,
    li_plus_element_sigmas,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "randomForest", "data")

SOURCES = [
    ("Limongi18", "dataset_Limongi_tpop42.npz"),
    ("Nomoto13", "dataset_Nomoto_tpop42.npz"),
]


def main():
    sigmas = li_plus_element_sigmas()
    pairs = _feature_pairs()
    for sn_source, fname in SOURCES:
        cache = os.path.join(DATA_DIR, fname)
        print(f"generating {sn_source} -> {fname} ...", flush=True)
        X, y, diag = generate(
            pisn_source="Takahashi",
            sn_source=sn_source,
            n_samples=2000,
            seed=12345,
            cache_path=None,  # regenerate fresh; we save the noisy version below
            p_sn_only=0.5,
        )
        rng = np.random.default_rng(42)
        X_noisy = add_dex_noise(X.copy(), pairs, sigmas, rng)
        np.savez_compressed(cache, X=X_noisy, y=y, diagnostics=diag)
        print(f"  saved {cache} (X={X.shape}, realistic Li+22 noise added)", flush=True)


if __name__ == "__main__":
    main()
