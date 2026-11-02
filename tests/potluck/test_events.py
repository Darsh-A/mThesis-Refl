"""Tests for the event prior."""

import numpy as np

from src.potluck.config import (
    DetectConfig,
    ModelConfig,
    PriorConfig,
    RuntimeConfig,
    YieldConfig,
)
from src.potluck.events import draw_events
from src.potluck.yields.pisn import load_pisn_table
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
    detect=DetectConfig(),
    runtime=RuntimeConfig(n_samples=200, chunk_size=100, seed=3),
)


def test_draw_events_ranges_and_priors() -> None:
    rng = np.random.default_rng(0)
    events = draw_events(
        200,
        CONFIG,
        rng,
        load_pisn_table("HW2002"),
        PopIIITable.from_raw(),
    )

    assert len(events) == 200
    assert events.pisn_masses.shape == (200, 2)
    assert events.iii_group_counts.shape[1] == 10
    assert events.iii_group_offsets.shape == (11,)
    assert events.iii_group_offsets[-1] == events.n_iii.sum()
    assert events.iii_sample_index.size == events.n_iii.sum()
    assert events.iii_row_index.size == events.n_iii.sum()
    assert np.all(np.diff(events.iii_group_offsets) >= 0)
    assert np.all((events.n_pisn >= 0) & (events.n_pisn <= 2))
    assert np.all((events.n_iii >= 0) & (events.n_iii <= 4096))
    assert np.all(events.mdil >= 2.0e5)
    assert np.all(events.mdil <= 1.0e9)
    assert np.all(
        events.mstar_iii + events.mstar_ii <= 0.1 * events.mdil * (1.0 + 1e-12)
    )
    assert np.all(events.delta_myr >= 2.0)
    assert np.all(events.t_myr >= 1.0)
    assert np.all(np.isfinite(events.log_prior))

    padding = np.arange(events.pisn_masses.shape[1])[None, :]
    assert np.all(events.pisn_masses[padding >= events.n_pisn[:, None]] == 0.0)
    assert np.all(events.n_ia == 0)


def test_events_subset() -> None:
    rng = np.random.default_rng(1)
    events = draw_events(
        20,
        CONFIG,
        rng,
        load_pisn_table("HW2002"),
        PopIIITable.from_raw(),
    )
    mask = np.arange(len(events)) % 2 == 0
    subset = events.subset(mask)
    assert len(subset) == 10
    assert np.all(subset.mdil == events.mdil[mask])
    assert np.all(subset.n_pisn == events.n_pisn[mask])
    assert subset.iii_group_offsets[-1] == subset.n_iii.sum()
    assert subset.iii_sample_index.size == subset.n_iii.sum()
    assert np.all((subset.iii_sample_index >= 0) & (subset.iii_sample_index < 10))
