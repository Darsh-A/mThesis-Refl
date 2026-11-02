"""Tests for noiseless features and PISN labels."""

import numpy as np

from src.potluck.config import ModelConfig, PriorConfig, RuntimeConfig, YieldConfig
from src.potluck.constants import ELEMENT_INDEX, METALS
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
    runtime=RuntimeConfig(n_samples=120, chunk_size=40, seed=6),
)


def _build():
    rng = np.random.default_rng(7)
    pisn = load_pisn_table("HW2002")
    popiii = PopIIITable.from_raw()
    popii = PopIITable(z_grid_size=4, t_grid_size=4, cache_dir=None)
    events = draw_events(120, CONFIG, rng, pisn, popiii)
    ejecta = build_ejecta(events, CONFIG, pisn, popiii, popii, rng=rng)
    return ejecta, build_features(ejecta, CONFIG)


def test_feature_shapes_and_ranges() -> None:
    ejecta, features = _build()
    n = len(ejecta)
    assert features.log_abundance.shape == (n, 32)
    assert features.ratios.shape == (n, len(CONFIG.feature_elements) - 1)
    assert features.ratio_mask.shape == (n, len(CONFIG.feature_elements) - 1)
    assert features.ratio_names[0] == "Al/Fe"
    assert len(features.ratio_names) == len(CONFIG.feature_elements) - 1
    assert features.independent_ratios.shape == (n, 4)
    assert features.independent_mask.shape == (n, 4)
    assert features.independent_names == ("Na/Mg", "Co/Ni", "Mn/Cr", "Zn/Mg")
    assert features.f_xp.shape == (n, 32)

    assert np.all(np.isfinite(features.ratios[features.ratio_mask]))
    assert np.all(np.isnan(features.ratios[~features.ratio_mask]))
    assert np.all(
        np.isfinite(
            features.independent_ratios[features.independent_mask]
        )
    )
    assert np.all(
        np.isnan(features.independent_ratios[~features.independent_mask])
    )

    accepted = ejecta.accepted
    assert np.all((features.f_zp[accepted] >= 0.0) & (features.f_zp[accepted] <= 1.0))
    assert np.all((features.f_fep[accepted] >= 0.0) & (features.f_fep[accepted] <= 1.0))
    assert np.all(np.isfinite(features.f_zp[accepted]))
    assert np.all(np.isfinite(features.f_fep[accepted]))


def test_no_pisn_has_zero_labels() -> None:
    ejecta, features = _build()
    quiet = ejecta.accepted & (ejecta.events.n_pisn == 0)
    assert quiet.any()
    assert np.allclose(features.f_zp[quiet], 0.0)
    assert np.allclose(features.f_fep[quiet], 0.0)


def test_f_zp_matches_matrix() -> None:
    ejecta, features = _build()
    accepted = ejecta.accepted
    metal_indices = [ELEMENT_INDEX[element] for element in METALS]
    expected = (
        ejecta.matrix[accepted, 0, :][:, metal_indices].sum(axis=1)
        / ejecta.M_z_tot[accepted]
    )
    assert np.allclose(features.f_zp[accepted], expected)
