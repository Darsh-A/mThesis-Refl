"""Tests for the direct Ti diagnostic."""

from __future__ import annotations

import json

import numpy as np

from src.potluck.diagnostics.ti_diagnostic import (
    TI_COLUMN,
    _bootstrap_offset,
    _mock_ti,
    _real_ti,
    run_ti_diagnostic,
)
from src.potluck.observation import load_li_records


def test_ti_column_and_real_ti():
    records = load_li_records()
    assert TI_COLUMN == 5
    real = _real_ti(records)
    assert real["ti"].shape == (records.n,)
    assert np.isfinite(real["ti"]).sum() > 300
    assert np.isnan(real["ti"][~real["detected"]]).all()


def test_mock_ti(small_dataset):
    arrays, _ = small_dataset
    mock = _mock_ti(arrays, load_li_records())
    n = arrays["feh_obs"].size
    for key in ("ti", "ti_true", "feh", "logg", "snr", "weight", "detected"):
        assert mock[key].shape == (n,)
    assert np.isfinite(mock["ti_true"]).sum() > 0
    assert np.isfinite(mock["snr"]).sum() > 0


def test_bootstrap_offset_identical():
    values = np.linspace(-1.0, 1.0, 41)
    result = _bootstrap_offset(values, values, np.ones_like(values), 50, 0)
    assert abs(result["offset"]) < 1e-12
    assert result["ks"] < 1e-12
    assert result["n_real"] == result["n_mock"] == 41


def test_run_ti_diagnostic_smoke(small_dataset, tmp_path):
    arrays, _ = small_dataset
    summary = run_ti_diagnostic(
        arrays,
        outdir=tmp_path,
        n_boot=50,
        seed=1,
        include_yields=False,
    )
    assert summary["n_real"] == 385
    assert summary["n_mock"] == arrays["feh_obs"].size
    assert "overall" in summary and "by_feh" in summary and "by_snr" in summary
    assert summary["giant_dwarf"]["inside"]["n_real"] > 0
    assert summary["giant_dwarf"]["outside"]["n_real"] > 0
    with (tmp_path / "ti_diagnostic.json").open() as handle:
        stored = json.load(handle)
    assert stored["n_mock"] == summary["n_mock"]
    assert (tmp_path / "ti_diagnostic.png").stat().st_size > 0
