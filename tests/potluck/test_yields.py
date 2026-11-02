"""Tests for the yield-table adapters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from params import ishigaki_2018_yields
from src.load_data import load_hw2002
from src.potluck.constants import ELEMENTS, ELEMENT_INDEX, METALS
from src.potluck.yields import (
    PopIITable,
    PopIIITable,
    element_vector,
    load_pisn_table,
)


def test_element_vector_matches_raw_sum():
    entry = load_hw2002()[0]
    vector = element_vector(entry)
    assert vector.shape == (len(ELEMENTS),)
    assert np.isclose(vector.sum(), sum(entry["yields"].values()))


def test_pisn_table_grid_and_interpolation():
    table = load_pisn_table("HW2002")
    assert len(table) == 14
    assert table.m_min == 140.0
    assert table.m_max == 270.0
    np.testing.assert_allclose(table.at_mass(200.0), table.vectors[6])

    interpolated = table.at_masses(np.array([145.0, 205.0]))
    assert interpolated.shape == (2, len(ELEMENTS))
    assert np.all(np.isfinite(interpolated))

    fractions = table.check_mass_closure()
    assert np.all(fractions > 0.0)
    assert np.all(fractions < 1.0)


def test_pisn_table_rejects_out_of_range():
    table = load_pisn_table("HW2002")
    with pytest.raises(ValueError):
        table.at_mass(139.0)


def test_takahashi_table_loads():
    table = load_pisn_table("Takahashi-MR")
    assert len(table) == 6
    assert table.m_min >= 160.0


@pytest.mark.skipif(
    not Path(ishigaki_2018_yields).exists(),
    reason="raw Ishigaki grid is not available",
)
def test_popiii_raw_grid():
    table = PopIIITable.from_raw()
    assert len(table) == 129717
    assert table.n_combos == 10
    assert sum(table.combo_counts) == len(table)
    assert table.vectors.shape == (len(table), len(ELEMENTS))

    first = table.vectors[0]
    assert 9.0 < first.sum() < 10.0
    assert np.all(table.check_mass_closure() < 1.01)

    rng = np.random.default_rng(0)
    indices = table.sample_indices(rng, 16, mode="combo")
    assert indices.shape == (16,)
    assert np.all((indices >= 0) & (indices < len(table)))
    assert table.vectors_at(indices).shape == (16, len(ELEMENTS))


def test_popii_grid(tmp_path):
    table = PopIITable(
        source_name="Limongi18",
        z_grid_size=3,
        t_grid_size=3,
        z_min=1.0e-4,
        z_max=2.0e-2,
        t_min_myr=3.0,
        t_max_myr=50.0,
        cache_dir=tmp_path,
    )
    assert table.values.shape == (3, 3, len(ELEMENTS))
    assert np.any(table.values > 0.0)

    iz, it = 1, 2
    np.testing.assert_allclose(
        table.integrated(table.z_grid[iz], table.t_grid[it]),
        table.values[iz, it],
        rtol=1.0e-10,
        atol=1.0e-14,
    )
    metal = table.metal_fraction(table.z_grid[iz], table.t_grid[it])
    expected = sum(
        table.values[iz, it, ELEMENT_INDEX[element]] for element in METALS
    )
    assert np.isclose(metal, expected)


def test_popii_rejects_excess_m_max():
    with pytest.raises(ValueError):
        PopIITable(source_name="Limongi18", m_max=200.0, cache_dir=None)
