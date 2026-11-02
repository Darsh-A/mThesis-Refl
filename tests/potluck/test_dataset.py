"""Integration tests for the rejection-sampled dataset builder."""

from __future__ import annotations

import numpy as np

from src.potluck.dataset import build_tables, generate_dataset
from src.potluck.validation import validate_dataset
from src.potluck.yields.popii import PopIITable
from src.potluck.yields.popiii import PopIIITable
from src.potluck.yields.base import PISNTable


def test_dataset_shapes(small_config, small_dataset):
    arrays, meta = small_dataset
    n = small_config.runtime.n_samples

    assert len(arrays["y"]) == n
    assert arrays["features"].shape == (n, len(small_config.feature_elements) - 1)
    assert arrays["independent_features"].shape == (n, 4)
    assert arrays["independent_mask"].shape == (n, 4)
    assert arrays["matrix"].shape == (n, 4, 32)
    assert arrays["valid_features"].shape == (n, len(small_config.feature_elements))
    assert arrays["f_xp"].shape == (n, 32)
    assert arrays["z_pisn"].shape == (n,)
    assert arrays["y_f_05"].shape == (n,)
    assert arrays["y_f_08"].shape == (n,)
    assert meta["n_accepted"] == n
    assert meta["n_drawn"] == meta["ledger"]["drawn"]


def test_dataset_is_internally_consistent(small_dataset):
    arrays, meta = small_dataset
    result = validate_dataset(arrays, meta)
    assert result["ok"], {
        name: check for name, check in result["checks"].items() if not check["ok"]
    }


def test_labels_match_features(small_dataset):
    arrays, _ = small_dataset
    np.testing.assert_array_equal(arrays["z_pisn"], arrays["n_pisn"] > 0)
    np.testing.assert_array_equal(arrays["y_f_05"], arrays["f_zp"] >= 0.5)
    np.testing.assert_array_equal(arrays["y_f_08"], arrays["f_zp"] >= 0.8)


def test_ledger_accounts_for_every_draw(small_dataset):
    _, meta = small_dataset
    ledger = meta["ledger"]
    assert ledger["accepted"] == meta["n_accepted"]
    assert (
        ledger["r1"] + ledger["r2"] + ledger["r3"] + ledger["r4"]
        + ledger["accepted"]
        <= ledger["drawn"]
    )
    assert meta["n_accepted_drawn"] == ledger["accepted"] + ledger["discarded"]


def test_tables_have_expected_types(small_config):
    pisn, popiii, popii = build_tables(small_config)
    assert isinstance(pisn, PISNTable)
    assert isinstance(popiii, PopIIITable)
    assert isinstance(popii, PopIITable)


def test_same_seed_reproduces_dataset(small_config):
    first, _ = generate_dataset(small_config, n_samples=10, seed=7)
    second, _ = generate_dataset(small_config, n_samples=10, seed=7)
    np.testing.assert_allclose(first["features"], second["features"], equal_nan=True)
    np.testing.assert_array_equal(first["y"], second["y"])
    np.testing.assert_allclose(first["matrix"], second["matrix"])
