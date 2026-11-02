"""Tests for the mock-only explanation figures."""

from __future__ import annotations

import json
from pathlib import Path

from src.potluck.diagnostics.explanation_plots import run_explanation_plots
from src.potluck.yields.popii import PopIITable


def test_explanation_plots(small_dataset, tmp_path: Path) -> None:
    """All eight figures and the summary are written from mock arrays."""
    arrays, meta = small_dataset
    popii = PopIITable(
        source_name="Nomoto13",
        z_grid_size=4,
        t_grid_size=4,
        t_min_myr=3.0,
        t_max_myr=40.0,
        cache_dir=str(tmp_path / "cache"),
    )
    outdir = tmp_path / "figures"
    summary = run_explanation_plots(small_dataset, outdir=outdir, popii=popii)

    assert summary["n_mocks"] == arrays["f_zp"].size
    assert summary["feature_names"] == list(meta["feature_names"])
    assert len(summary["figures"]) == 8
    for name in summary["figures"]:
        assert (outdir / name).stat().st_size > 0

    payload = json.loads((outdir / "explanation_plots.json").read_text())
    assert payload["n_mocks"] == summary["n_mocks"]
    assert set(payload["ratio_spearman"]) == {"Na/Mg", "Co/Ni", "Mn/Cr", "Zn/Mg"}
