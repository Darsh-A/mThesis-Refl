"""Checks for Phase B1 calibration and real-star scoring."""

import numpy as np

from src.potluck.calibration import (
    fit_phase_b_models,
    load_li_records,
    prior_shift,
    real_matrix,
    score_real_stars,
)


def test_prior_shift() -> None:
    probability = np.array([0.5, 0.9])
    shifted = prior_shift(probability, prevalence=2.5e-3, training_rate=0.25)
    assert shifted[0] < 0.01
    assert shifted[1] > shifted[0]


def test_real_matrix_shape_and_independent_ratios() -> None:
    records = load_li_records()
    matrix = real_matrix(records)
    assert matrix.shape == (385, 42)
    ratio_elements = ("C", "Na", "Mg", "Si", "Ca", "Ti", "Cr", "Mn", "Co", "Ni", "Zn")
    na = ratio_elements.index("Na")
    mg = ratio_elements.index("Mg")
    available = (
        (records.flags[:, 1] > 0)
        & (records.flags[:, 2] > 0)
        & np.isfinite(records.xfe[:, 1])
        & np.isfinite(records.xfe[:, 2])
    )
    expected = matrix[available, na] - matrix[available, mg]
    assert np.allclose(matrix[available, 34], expected)


def test_small_dataset_scoring(small_dataset) -> None:
    arrays, _ = small_dataset
    models = fit_phase_b_models(arrays, seed=0)
    table = score_real_stars(arrays, models, prevalence=2.5e-3, n_boot=5)
    assert len(table) == 385
    assert {"p_posterior", "p_low", "p_high", "expected_fz"}.issubset(table.columns)
    assert table["p_posterior"].between(0.0, 1.0).all()
    assert table["expected_fz"].between(0.0, 1.0).all()
    assert np.isfinite(table["support_threshold"].iloc[0])
