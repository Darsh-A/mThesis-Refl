"""Tests for the Pop II yield-table comparison."""

from __future__ import annotations

import json
from pathlib import Path

from src.potluck.config import ModelConfig, RuntimeConfig, YieldConfig
from src.potluck.diagnostics.popii_metallicity import (
    LIMONGI_Z_NODES,
    _config,
    run_popii_metallicity,
)


def test_config_switches_table_and_nodes() -> None:
    base = ModelConfig()
    nomoto = _config("nomoto13", base)
    limongi = _config("limongi18", base)

    assert nomoto.yields.popii_table == "Nomoto13"
    assert nomoto.yields.popii_z_nodes is None
    assert nomoto.yields.popii_z_grid_size == 24
    assert limongi.yields.popii_table == "Limongi18"
    assert limongi.yields.popii_z_nodes == LIMONGI_Z_NODES
    assert limongi.yields.popii_z_grid_size == len(LIMONGI_Z_NODES)


def test_run_popii_metallicity_writes_figure(tmp_path: Path) -> None:
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
    outdir = tmp_path / "v9"
    result = run_popii_metallicity(
        n_samples=200,
        seed=1,
        outdir=outdir,
        config=config,
        auc_per_pool=150,
    )

    assert set(result["pools"]) == {"nomoto13", "limongi18"}
    for pool in result["pools"].values():
        assert pool["n_mocks"] == 200
    assert 0.0 <= result["separability"]["auc"] <= 1.0
    names = set(result["pools"]["nomoto13"]["feature_names"])
    assert set(result["element_median_shift"]) == names

    assert (outdir / "popii_metallicity.png").stat().st_size > 0
    payload = json.loads((outdir / "popii_metallicity.json").read_text())
    assert payload["n_samples"] == 200
