"""Tests for the ejecta matrix and selection cuts."""

import numpy as np

from src.potluck.config import ModelConfig, PriorConfig, RuntimeConfig, YieldConfig
from src.potluck.constants import ELEMENT_INDEX, METALS
from src.potluck.ejecta import build_ejecta
from src.potluck.events import draw_events
from src.potluck.yields.pisn import load_pisn_table
from src.potluck.yields.popii import PopIITable
from src.potluck.yields.popiii import PopIIITable

CONFIG = ModelConfig(
    prior=PriorConfig(
        n_pisn_probs=(0.5, 0.3, 0.2),
        lambda_iii_range=(0.1, 5.0),
        delta_range_myr=(2.0, 50.0),
        t_range_myr=(1.0, 50.0),
        mdil_range=(2.0e5, 1.0e9),
        mstar_ii_range=(1.0e3, 1.0e5),
        mstar_total_max_ratio=0.1,
    ),
    yields=YieldConfig(popii_z_grid_size=4, popii_t_grid_size=4),
    runtime=RuntimeConfig(n_samples=150, chunk_size=50, seed=4),
)


def _tables() -> tuple:
    return (
        load_pisn_table("HW2002"),
        PopIIITable.from_raw(),
        PopIITable(
            z_grid_size=4,
            t_grid_size=4,
            cache_dir=None,
        ),
    )


def test_ejecta_shapes_and_invariants() -> None:
    rng = np.random.default_rng(2)
    pisn, popiii, popii = _tables()
    events = draw_events(150, CONFIG, rng, pisn, popiii)
    ejecta = build_ejecta(events, CONFIG, pisn, popiii, popii, rng=rng)

    assert len(ejecta) == 150
    assert ejecta.matrix.shape == (150, 4, 32)
    assert np.allclose(ejecta.M_tot, ejecta.matrix.sum(axis=1))
    assert np.all(ejecta.M_pre <= ejecta.M_tot + 1.0e-12)
    assert np.all(ejecta.M_z_tot >= ejecta.M_z_pre - 1.0e-12)
    assert np.allclose(
        ejecta.source_metal.sum(axis=1), ejecta.M_z_tot
    )

    positive = ejecta.M_z_pre > 0.0
    assert np.allclose(
        ejecta.z_star[positive],
        ejecta.M_z_pre[positive] / events.mdil[positive],
    )
    assert np.all(ejecta.z_final[ejecta.accepted] >= ejecta.z_star[ejecta.accepted])

    assert np.all(~ejecta.r2 | ejecta.r1)
    assert np.all(~ejecta.r4 | ejecta.r2)
    assert np.all(~ejecta.r3 | ejecta.r4)
    assert np.array_equal(ejecta.accepted, ejecta.r3)
    assert np.all(np.isfinite(ejecta.feh[ejecta.accepted]))

    assert ejecta.valid_features.shape == (150, len(CONFIG.feature_elements))
    assert np.all(
        ejecta.n_valid_features
        == ejecta.valid_features.sum(axis=1)
    )
    assert np.all(
        ejecta.stage[ejecta.accepted] == 0
    )
    assert np.all(ejecta.stage[~ejecta.r1] == 1)


def test_ejecta_subset() -> None:
    rng = np.random.default_rng(3)
    pisn, popiii, popii = _tables()
    events = draw_events(40, CONFIG, rng, pisn, popiii)
    ejecta = build_ejecta(events, CONFIG, pisn, popiii, popii, rng=rng)
    mask = np.arange(len(ejecta)) % 2 == 0
    subset = ejecta.subset(mask)
    assert len(subset) == 20
    assert np.all(subset.M_z_tot == ejecta.M_z_tot[mask])
    assert np.all(subset.events.mdil == events.mdil[mask])


def test_metal_mass_matches_matrix() -> None:
    rng = np.random.default_rng(5)
    pisn, popiii, popii = _tables()
    events = draw_events(30, CONFIG, rng, pisn, popiii)
    ejecta = build_ejecta(events, CONFIG, pisn, popiii, popii, rng=rng)
    metal_indices = [ELEMENT_INDEX[element] for element in METALS]
    assert np.allclose(
        ejecta.M_tot[:, metal_indices].sum(axis=1), ejecta.M_z_tot
    )
