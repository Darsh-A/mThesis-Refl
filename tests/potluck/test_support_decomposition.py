"""Tests for support decomposition and gate ablation."""

import numpy as np

from src.potluck.diagnostics.support_decomposition import (
    ablation_variants,
    decompose_support,
    plot_decomposition,
)
from src.potluck.observation import load_li_records


def test_decompose_support(small_dataset, tmp_path):
    arrays, meta = small_dataset
    records = load_li_records()
    table, summary = decompose_support(records, arrays, n_reference=20, seed=0)

    assert len(table) == records.n
    assert summary["n"] == records.n
    assert set(["distance", "threshold", "driver", "in_model"]).issubset(table.columns)
    assert summary["in_model"] == int(table["in_model"].sum())
    assert np.isfinite(summary["threshold"])
    assert table["driver"].isin(["", "Na", "Mg", "Ca", "Ti", "Cr", "Ni"]).all()

    path = plot_decomposition(table, summary, tmp_path / "support.png")
    assert path.exists() and path.stat().st_size > 0


def test_ablation_variants(small_config):
    variants = ablation_variants(small_config)
    assert set(variants) == {
        "current",
        "no_r4",
        "no_zcr",
        "cap_0.3",
        "wide_lambda",
        "no_100",
    }
    assert variants["no_r4"].prior.feh_range is None
    assert variants["no_zcr"].prior.z_cr is None
    assert variants["cap_0.3"].prior.mstar_total_max_ratio == 0.3
