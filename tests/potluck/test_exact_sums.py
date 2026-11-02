"""Checks that the vectorised Pop III sums equal a naive event loop."""

import numpy as np
from scipy.stats import chisquare, ks_2samp, kstest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.potluck.config import ModelConfig, PriorConfig
from src.potluck.constants import ELEMENT_INDEX, ELEMENTS
from src.potluck.ejecta import _popiii_sums
from src.potluck.events import Events, draw_events
from src.potluck.yields.pisn import load_pisn_table
from src.potluck.yields.popiii import PopIIITable


def _single_group_events(
    table: PopIIITable,
    group: int,
    count: int,
    rng: np.random.Generator,
) -> Events:
    """Build one sample containing ``count`` events from one group."""
    n_groups = table.n_combos
    counts = np.zeros((1, n_groups), dtype=np.int64)
    counts[0, group] = count
    offsets = np.concatenate([[0], np.cumsum(counts.sum(axis=0))])
    rows = rng.choice(table.combo_indices[group], size=count, replace=True)
    return Events(
        n_pisn=np.zeros(1, dtype=np.int64),
        pisn_masses=np.zeros((1, 1), dtype=float),
        lambda_iii=np.zeros(1, dtype=float),
        n_iii=np.asarray([count], dtype=np.int64),
        iii_group_counts=counts,
        iii_group_offsets=offsets.astype(np.int64),
        iii_sample_index=np.zeros(count, dtype=np.int32),
        iii_group_index=np.full(count, group, dtype=np.int8),
        iii_row_index=rows.astype(np.int32),
        mstar_iii=np.zeros(1, dtype=float),
        delta_myr=np.zeros(1, dtype=float),
        t_myr=np.ones(1, dtype=float),
        mdil=np.full(1, 1.0e6),
        mstar_ii=np.full(1, 1.0e3),
        n_ia=np.zeros(1, dtype=np.int64),
        log_prior=np.zeros(1, dtype=float),
    )


def test_vectorised_sums_match_naive_loop() -> None:
    rng = np.random.default_rng(11)
    table = PopIIITable.from_raw()
    for group in range(table.n_combos):
        for count in (10, 30, 31, 100, 300, 1000):
            events = _single_group_events(table, group, count, rng)
            final, pre = _popiii_sums(
                events,
                table,
                np.zeros(1),
                np.full(1, 1.0e9),
            )
            rows = events.iii_row_index
            naive = table.vectors[rows].sum(axis=0)
            assert np.allclose(final[0], naive, rtol=0.0, atol=1e-12)
            assert np.all(pre[0] == 0.0)
            assert final.shape == (1, len(ELEMENTS))


def test_explicit_sampler_seam() -> None:
    """Two independent explicit draws must be indistinguishable."""
    table = PopIIITable.from_raw()
    rng = np.random.default_rng(8)
    n_events = 31
    n_sample = 5000
    for group, rows in enumerate(table.combo_indices):
        left = table.vectors[
            rng.choice(rows, size=(n_sample, n_events), replace=True)
        ].sum(axis=1) / n_events
        right = table.vectors[
            rng.choice(rows, size=(n_sample, n_events), replace=True)
        ].sum(axis=1) / n_events
        features = np.vstack([left, right])
        labels = np.r_[np.zeros(n_sample), np.ones(n_sample)]
        splits = StratifiedKFold(3, shuffle=True, random_state=1)
        probability = cross_val_predict(
            HistGradientBoostingClassifier(max_iter=100, max_depth=3),
            features,
            labels,
            cv=splits,
            method="predict_proba",
        )[:, 1]
        auc = roc_auc_score(labels, probability)
        assert abs(auc - 0.5) <= 0.02, (group, auc)


def _event_config() -> ModelConfig:
    return ModelConfig(
        prior=PriorConfig(
            n_pisn_probs=(1.0,),
            lambda_iii_range=(499.9, 500.1),
            mdil_range=(2.0e5, 1.0e9),
            mstar_ii_range=(1.0e3, 1.0e5),
            mstar_total_max_ratio=None,
        )
    )


def test_group_counts_match_multinomial() -> None:
    table = PopIIITable.from_raw()
    events = draw_events(
        2000,
        _event_config(),
        np.random.default_rng(21),
        load_pisn_table("HW2002"),
        table,
    )
    observed = events.iii_group_counts.sum(axis=0).astype(float)
    expected = np.full(table.n_combos, observed.sum() / table.n_combos)
    assert chisquare(observed, expected).pvalue > 0.01


def test_rows_are_uniform_within_a_group() -> None:
    table = PopIIITable.from_raw()
    rows = table.sample_indices(
        np.random.default_rng(22), 100_000, mode="combo"
    )
    group_rows = table.combo_indices[0]
    selected = rows[np.isin(rows, group_rows)].astype(float)
    selected = (selected - group_rows.min()) / (group_rows.max() - group_rows.min())
    assert kstest(selected, "uniform").pvalue > 0.01


def test_events_match_naive_per_event_draw() -> None:
    table = PopIIITable.from_raw()
    config = _event_config()
    config = ModelConfig(
        prior=PriorConfig(
            n_pisn_probs=(1.0,),
            lambda_iii_range=(0.999, 1.001),
            mdil_range=config.prior.mdil_range,
            mstar_ii_range=config.prior.mstar_ii_range,
            mstar_total_max_ratio=None,
        )
    )
    events = draw_events(
        5000,
        config,
        np.random.default_rng(23),
        load_pisn_table("HW2002"),
        table,
    )
    index = ELEMENT_INDEX["Fe"]
    sampled = table.vectors[events.iii_row_index, index]
    rng = np.random.default_rng(24)
    naive = np.array(
        [
            table.vectors[
                rng.choice(table.combo_indices[rng.integers(table.n_combos)]),
                index,
            ]
            for _ in range(sampled.size)
        ]
    )
    assert ks_2samp(sampled, naive).pvalue > 0.01
