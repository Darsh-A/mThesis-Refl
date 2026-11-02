"""Tests for the per-element audit."""

from __future__ import annotations

import json
from pathlib import Path

from src.potluck.constants import LI_FEATURE_ELEMENTS
from src.potluck.diagnostics.element_audit import AUDIT_ELEMENTS, run_element_audit


def test_run_element_audit_writes_figure(small_dataset, tmp_path: Path) -> None:
    arrays, meta = small_dataset
    outdir = tmp_path / "v10"
    summary = run_element_audit(small_dataset, outdir=outdir)

    assert summary["n_mocks"] == arrays["features"].shape[0]
    assert set(summary["elements"]) == set(AUDIT_ELEMENTS)
    assert set(summary["offsets_mock_minus_obs"]) <= set(LI_FEATURE_ELEMENTS)
    assert summary["largest_offset_element"] in summary["offsets_mock_minus_obs"]
    assert (outdir / "element_audit.png").stat().st_size > 0

    payload = json.loads((outdir / "element_audit.json").read_text())
    assert payload["n_mocks"] == summary["n_mocks"]
    assert set(payload["elements"]) == set(AUDIT_ELEMENTS)
