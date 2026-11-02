"""Tests for the matched-prior yield-table cross."""

from __future__ import annotations

from src.potluck.config import ModelConfig
from src.potluck.diagnostics.yield_cross import (
    _median_offsets,
    run_yield_cross,
    yield_variants,
)
from src.potluck.observation import load_li_records


def test_yield_variants():
    variants = yield_variants(ModelConfig())
    assert set(variants) == {
        "popii_Nomoto13",
        "popii_Limongi18",
        "popii_WW95",
        "pisn_HW2002",
        "pisn_Takahashi-MR",
        "pisn_Takahashi-NM",
        "pisn_Takahashi-NR",
    }
    assert variants["popii_Nomoto13"].yields.popii_table == "Nomoto13"
    assert variants["pisn_Takahashi-NR"].yields.pisn_table == "Takahashi-NR"


def test_median_offsets(small_dataset):
    arrays, _ = small_dataset
    result = _median_offsets(load_li_records(), arrays)
    assert "Ti" in result
    assert set(result["Ti"]) == {"real_median", "mock_median", "offset"}
    assert result["Ti"]["real_median"] is not None
    assert result["Ti"]["mock_median"] is not None


def test_run_yield_cross_smoke(small_config, tmp_path):
    summary = run_yield_cross(
        config=small_config,
        n_samples=30,
        seed=1,
        outdir=tmp_path,
    )
    assert len(summary) == 7
    assert (tmp_path / "yield_cross.json").is_file()
    assert (tmp_path / "yield_cross.png").stat().st_size > 0
    for metrics in summary.values():
        assert 0.0 <= metrics["in_model_fraction"] <= 1.0
        assert "Ti" in metrics["ks"]
