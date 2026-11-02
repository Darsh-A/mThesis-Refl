"""Tests for Li+ importance reweighting."""

from __future__ import annotations

import json

import numpy as np

from src.potluck.diagnostics.reweighting import (
    _active_ratio_elements,
    _mock_ratios,
    _null_ess,
    _weighted_quantile,
    run_reweighting,
)


def test_active_ratio_elements():
    elements = _active_ratio_elements(("Ti",))
    assert "Ti" not in elements
    assert "Fe" not in elements
    assert elements[0] == "C"
    assert elements[-1] == "Zn"


def test_mock_ratios_match_features(small_dataset):
    arrays, meta = small_dataset
    elements = _active_ratio_elements(())
    mock = _mock_ratios(arrays, elements, meta["feature_names"])
    names = list(meta["feature_names"])
    for index, element in enumerate(elements):
        column = names.index(f"{element}/Fe")
        assert np.array_equal(
            mock[:, index],
            arrays["features"][:, column],
            equal_nan=True,
        )


def test_weighted_quantile():
    values = np.asarray([0.0, 10.0])
    weight = np.asarray([3.0, 1.0])
    quantiles = _weighted_quantile(values, weight)
    assert quantiles[1] == 0.0
    assert quantiles[2] == 10.0
    equal = _weighted_quantile(
        np.asarray([0.0, 1.0, 2.0, 3.0]),
        np.ones(4),
    )
    assert equal[1] == 1.0


def test_null_ess(small_dataset):
    arrays, meta = small_dataset
    elements = _active_ratio_elements(("Ti",))
    mock_ratios = _mock_ratios(arrays, elements, meta["feature_names"])
    result = _null_ess(
        arrays,
        elements,
        min_elements=2,
        mock_ratios=mock_ratios,
        n_stars=10,
        n_draws=2,
    )
    assert result["n_draws"] == 2
    assert len(result["values"]) == 2
    assert result["mean"] >= 0.0


def test_run_reweighting_smoke(small_dataset, tmp_path):
    summary = run_reweighting(
        small_dataset,
        outdir=tmp_path,
        min_elements=3,
        core_elements=("Na", "Mg", "Ca", "Cr", "Ni"),
    )
    assert summary["n_mocks"] == 30
    assert "all" in summary["schemes"]
    assert "core" in summary["schemes"]
    for result in summary["schemes"].values():
        assert 0.0 <= result["ess"] <= summary["n_mocks"]
    assert (tmp_path / "reweighting.json").is_file()
    assert (tmp_path / "reweighting.png").stat().st_size > 0
    with (tmp_path / "reweighting.json").open() as handle:
        loaded = json.load(handle)
    assert loaded["n_mocks"] == 30
