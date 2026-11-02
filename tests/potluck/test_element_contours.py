"""Tests for the per-element contour plots."""

from __future__ import annotations

import json
from pathlib import Path

from src.potluck.diagnostics.element_contours import run_element_contours


def test_run_element_contours_writes_one_figure_per_element(
    small_dataset, tmp_path: Path
) -> None:
    arrays, meta = small_dataset
    outdir = tmp_path / "v14"
    summary = run_element_contours(small_dataset, outdir=outdir)

    elements = [name.split("/")[0] for name in meta["feature_names"]]
    assert summary["n_mocks"] == arrays["feh"].size
    assert list(summary["elements"]) == elements

    for element in elements:
        figure = outdir / f"{element}_Fe.png"
        assert figure.stat().st_size > 0
        assert summary["elements"][element]["figure"] == figure.name

    payload = json.loads((outdir / "element_contours.json").read_text())
    assert payload["n_mocks"] == summary["n_mocks"]
    assert list(payload["elements"]) == elements

    if {"Mg", "Si"} <= set(elements):
        ratio = summary["ratios"]["Mg/Si"]
        assert ratio["figure"] == "Mg_Si.png"
        assert (outdir / "Mg_Si.png").stat().st_size > 0
        assert ratio["n_mocks"] > 0
        assert payload["ratios"]["Mg/Si"]["figure"] == "Mg_Si.png"
