"""Tests for the counterfactual detectability label."""

import numpy as np

from src.potluck.config import ModelConfig, PriorConfig, RuntimeConfig, YieldConfig
from src.potluck.detect import DETECT_REASONS, detect_labels
from src.potluck.ejecta import build_ejecta
from src.potluck.events import draw_events
from src.potluck.features import build_features
from src.potluck.yields.pisn import load_pisn_table
from src.potluck.yields.popii import PopIITable
from src.potluck.yields.popiii import PopIIITable

CONFIG = ModelConfig(
    prior=PriorConfig(
        n_pisn_probs=(0.5, 0.3, 0.2),
        lambda_iii_range=(0.1, 5.0),
        delta_range_myr=(2.0, 50.0),
        t_range_myr=(1.0, 50.0),
        mdil_range=(2.0e5, 1.0e7),
        mstar_ii_range=(1.0e3, 1.0e5),
        mstar_total_max_ratio=0.1,
    ),
    yields=YieldConfig(popii_z_grid_size=4, popii_t_grid_size=4),
    runtime=RuntimeConfig(n_samples=120, chunk_size=40, seed=8),
)


def _build():
    rng = np.random.default_rng(9)
    pisn = load_pisn_table("HW2002")
    popiii = PopIIITable.from_raw()
    popii = PopIITable(z_grid_size=4, t_grid_size=4, cache_dir=None)
    events = draw_events(120, CONFIG, rng, pisn, popiii)
    ejecta = build_ejecta(events, CONFIG, pisn, popiii, popii, rng=rng)
    features = build_features(ejecta, CONFIG)
    return ejecta, features, detect_labels(features, ejecta, CONFIG)


def test_detection_shapes_and_codes() -> None:
    ejecta, _, detection = _build()
    n = len(ejecta)
    assert detection.D.shape == (n,)
    assert detection.D2.shape == (n,)
    assert detection.y.shape == (n,)
    assert detection.reason.shape == (n,)
    assert detection.n_used.shape == (n,)
    assert np.all(detection.reason >= 0)
    assert np.all(detection.reason < len(DETECT_REASONS))
    assert np.all(~detection.y[~ejecta.accepted])


def test_no_pisn_is_not_detected() -> None:
    ejecta, _, detection = _build()
    quiet = ejecta.accepted & (ejecta.events.n_pisn == 0)
    assert quiet.any()
    assert np.all(~detection.y[quiet])
    assert np.all(detection.D[quiet] == 0.0)
    assert np.all(
        detection.reason[quiet] == DETECT_REASONS.index("no_pisn")
    )


def test_normal_cases_follow_threshold() -> None:
    ejecta, _, detection = _build()
    normal = ejecta.accepted & (
        detection.reason == DETECT_REASONS.index("normal")
    )
    assert normal.any()
    assert np.all(detection.n_used[normal] >= CONFIG.detect.min_detected)
    assert np.all(
        detection.y[normal] == (detection.D[normal] > CONFIG.detect.k)
    )
    assert np.all(detection.D[normal] >= 0.0)


def test_baseline_zero_is_detected() -> None:
    ejecta, _, detection = _build()
    zero = detection.reason == DETECT_REASONS.index("baseline_zero")
    if zero.any():
        assert np.all(detection.y[zero])
        assert np.all(np.isinf(detection.D[zero]))
