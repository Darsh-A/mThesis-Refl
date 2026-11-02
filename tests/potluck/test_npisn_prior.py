"""Tests for the number-of-PISN prior diagnostic."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.potluck.config import ModelConfig, RuntimeConfig, YieldConfig
from src.potluck.diagnostics.npisn_prior import _summarise, run_npisn_prior


def test_summarise_counts_and_fractions() -> None:
    arrays = {
        "f_zp": np.array([0.0, 0.0, 0.2, 0.6, 1.0]),
        "n_pisn": np.array([0, 0, 1, 2, 3]),
    }
    summary = _summarise(arrays, (0.5, 0.3, 0.15, 0.05))

    assert summary["n_mocks"] == 5
    assert summary["n_pisn_counts"] == [2, 1, 1, 1]
    assert summary["prior_probs"] == [0.5, 0.3, 0.15, 0.05]
    assert summary["zero_fraction"] == 0.4
    assert summary["positive_fraction"] == 0.4
    assert summary["by_group"]["1"]["median"] == 0.2
    assert summary["by_group"]["2"]["positive_fraction"] == 1.0


def test_run_npisn_prior_writes_figure(tmp_path: Path) -> None:
    config = ModelConfig(
        yields=YieldConfig(popii_z_grid_size=4, popii_t_grid_size=4),
        runtime=RuntimeConfig(
            n_samples=200,
            chunk_size=100,
            max_draw_factor=50,
            cache_dir=str(tmp_path / "cache"),
            output_dir=str(tmp_path),
        ),
    )
    outdir = tmp_path / "v8"
    result = run_npisn_prior(n_samples=200, seed=1, outdir=outdir, config=config)

    assert result["default"]["n_mocks"] == 200
    assert result["uniform"]["n_mocks"] == 200
    assert set(result["ks_by_group"]) == {"1", "2", "3"}
    assert (outdir / "npisn_prior.png").stat().st_size > 0
    assert (outdir / "npisn_prior.json").exists()
