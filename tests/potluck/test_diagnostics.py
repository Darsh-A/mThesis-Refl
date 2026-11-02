"""Tests for the prior and rejection diagnostics."""

from __future__ import annotations

from src.potluck.constants import DEFAULT_FEATURE_ELEMENTS
from src.potluck.diagnostics import (
    diagnose,
    format_ledger,
    format_report,
    ledger_fractions,
)


def test_report_has_expected_sections(small_dataset):
    arrays, meta = small_dataset
    report = diagnose(arrays, meta)

    assert report["n"] == len(arrays["y"])
    assert 0.0 <= report["y_rate"] <= 1.0
    assert set(report["ledger"]) == {
        "r1",
        "r2",
        "r3",
        "r4",
        "accepted",
        "accepted_total",
    }
    assert set(report["r2_reasons"]) == {
        "no_pre_enrichment",
        "above_z_max",
        "below_z_cr",
        "other",
    }
    assert len(report["feature_coverage"]) == len(DEFAULT_FEATURE_ELEMENTS) - 1
    assert sum(cell["count"] for cell in report["mixture_cells"].values()) == report["n"]
    assert sum(cell["count"] for cell in report["y_by_n_pisn"].values()) == report["n"]


def test_ledger_fractions_are_bounded(small_dataset):
    _, meta = small_dataset
    fractions = ledger_fractions(meta)
    assert all(0.0 <= value <= 1.0 for value in fractions.values())
    assert fractions["accepted"] <= fractions["accepted_total"] + 1e-12
    total = (
        fractions["r1"]
        + fractions["r2"]
        + fractions["r3"]
        + fractions["accepted_total"]
    )
    assert total <= 1.0 + 1e-12


def test_formatters_return_text(small_dataset):
    arrays, meta = small_dataset
    report = diagnose(arrays, meta)
    assert "samples" in format_report(report)
    assert "drawn" in format_ledger(meta)
