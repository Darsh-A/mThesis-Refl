"""Diagnostic checks for src.randomForest.dataset.generate().

Validates the (X, y, diagnostics) triple produced by the f_pisn dataset
builder. NOT a pytest collection -- a lightweight standalone checker::

    python src/randomForest/test_data_generate.py

Run this whenever dataset.py (feature set, priors, hygiene) changes to catch
silent regressions in the generated training data.

Use a small n_samples (3) so the expensive per-sample abundance-ratio loop is
cheap. Each check returns a bool; failures raise AssertionError.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.randomForest.dataset import generate, _feature_pairs, _abundance_elements

# Expected geometry -- keep in sync with dataset.py design.
N_SAMPLES = 3
N_FEATURES = len(_feature_pairs())  # C(n_elem, 2)
SEED = 12345

PISN_SOURCE = "HW2002"
SN_SOURCE = "Limongi18"
# NOTE: "Nomoto13" currently fails inside salvadori_combined_abundratio because
# its YieldSource.mass_from_lifetime is None, so it cannot map tpop2 -> SN mass.
# Use "Limongi18" or "WW95" as the default SN source until that is fixed.

TOL = 1e-12


def check_shapes(X, y, diag):
    assert X.shape == (N_SAMPLES, N_FEATURES), f"X shape {X.shape}"
    assert y.shape == (N_SAMPLES,), f"y shape {y.shape}"
    for key in ("f_ratio", "tpop2", "pisn_mass"):
        assert diag[key].shape == (N_SAMPLES,), f"diagnostic {key} shape {diag[key].shape}"
    return True


def check_dtypes(X, y):
    assert X.dtype == np.float64, f"X dtype {X.dtype}"
    assert y.dtype == np.float64, f"y dtype {y.dtype}"
    return True


def check_feature_names(diag):
    names = diag["feature_names"]
    assert len(names) == N_FEATURES, f"feature_names length {len(names)}"
    # Names match the pairwise [X/Y] scheme and are unique.
    assert len(set(names)) == N_FEATURES, "duplicate feature names"
    elems = set(_abundance_elements())
    for nm in names:
        assert nm.startswith("[") and nm.endswith("]") and "/" in nm, f"bad name {nm}"
        body = nm[1:-1].split("/")
        assert len(body) == 2 and set(body) <= elems, f"name elems outside set: {nm}"
    return True


def check_target_range(y):
    assert np.all(np.isfinite(y)), "non-finite target"
    assert np.all((y >= 0.1) & (y <= 1.0)), f"target out of [0.1, 1]: {y}"
    return True


def check_features(X):
    finite = np.isfinite(X)
    assert np.all(finite), f"NaNs in X: {np.isnan(X).sum()} cells"
    # floor clamp: no value below RATIO_FLOOR (-5)
    assert np.all(X >= -5.0 - TOL), f"value below floor: {X.min()}"
    # sanity: ratios should be in a plausible dex range
    assert X.max() < 5.0, f"implausible ratio value {X.max()}"
    return True


def check_diagnostics_ranges(diag):
    f_ratio = diag["f_ratio"]
    tpop2 = diag["tpop2"]
    mass = diag["pisn_mass"]
    assert np.all(np.isfinite(f_ratio)) and np.all(np.isfinite(tpop2)) and np.all(np.isfinite(mass))
    assert np.all((f_ratio >= 1e-4) & (f_ratio <= 1e-1)), f"f_ratio out of range: {f_ratio}"
    assert np.all((tpop2 >= 3.2e6) & (tpop2 <= 17.4e6)), f"tpop2 out of range: {tpop2}"
    assert np.all((mass >= 150.0) & (mass <= 270.0)), f"pisn_mass out of range: {mass}"
    return True


def check_metadata(diag):
    assert diag["pisn_source"] == PISN_SOURCE, diag["pisn_source"]
    assert diag["sn_source"] == SN_SOURCE, diag["sn_source"]
    assert diag["n_samples"] == N_SAMPLES, diag["n_samples"]
    assert diag["seed"] == SEED, diag["seed"]
    return True


def check_determinism():
    X1, y1, d1 = generate(PISN_SOURCE, SN_SOURCE, N_SAMPLES, SEED)
    X2, y2, d2 = generate(PISN_SOURCE, SN_SOURCE, N_SAMPLES, SEED)
    assert np.array_equal(X1, X2, equal_nan=True), "X not reproducible"
    assert np.array_equal(y1, y2), "y not reproducible"
    assert np.array_equal(d1["f_ratio"], d2["f_ratio"]), "f_ratio not reproducible"
    return True


def check_distinct_samples(X, y):
    # y drawn from a continuous prior -> samples should differ (not collapse).
    # (n_samples=3 from U[0.1,1]; collisions are measure-zero, but guard softly.)
    assert len(set(np.round(y, 6))) == N_SAMPLES, "targets collapsed"
    # features should vary across samples
    assert np.ptp(X, axis=0).max() > 0, "features constant across samples"
    return True


def run_all():
    X, y, diag = generate(PISN_SOURCE, SN_SOURCE, N_SAMPLES, SEED)
    checks = [
        ("shapes", check_shapes, (X, y, diag)),
        ("dtypes", check_dtypes, (X, y)),
        ("feature_names", check_feature_names, (diag,)),
        ("target_range", check_target_range, (y,)),
        ("features", check_features, (X,)),
        ("diagnostics_ranges", check_diagnostics_ranges, (diag,)),
        ("metadata", check_metadata, (diag,)),
        ("determinism", check_determinism, ()),
        ("distinct_samples", check_distinct_samples, (X, y)),
    ]
    failed = 0
    for name, fn, args in checks:
        try:
            fn(*args)
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
    print("-" * 40)
    print(f"{len(checks) - failed}/{len(checks)} checks passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if run_all() else 0)
