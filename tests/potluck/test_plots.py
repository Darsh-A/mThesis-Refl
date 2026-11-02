"""Tests for the diagnostic figures."""

from __future__ import annotations

from pathlib import Path

from src.potluck.diagnostics.ceiling import ceiling_report, noise_sweep
from src.potluck.diagnostics.plots import (
    plot_ablation,
    plot_all,
    plot_ceiling,
    plot_eq13,
    plot_noise_sweep,
)


def test_plot_all_writes_every_figure(small_dataset, tmp_path: Path) -> None:
    arrays, meta = small_dataset
    paths = plot_all(arrays, meta, tmp_path)

    assert len(paths) == 11
    for path in paths.values():
        assert Path(path).stat().st_size > 0
    assert (tmp_path / "diagnostics.json").exists()


def test_ceiling_report(small_dataset, tmp_path: Path) -> None:
    arrays, meta = small_dataset
    report = ceiling_report(arrays, meta, cv=3, seed=0)
    assert report["n"] == len(arrays["y"])
    assert 0.0 <= report["auc"] <= 1.0
    assert 0.0 <= report["nuisance_auc"] <= 1.0
    assert "auc_by_z_star" in report
    assert "auc_by_mstar_ii" in report
    plot_ceiling(report, tmp_path / "ceiling.png")
    assert (tmp_path / "ceiling.png").stat().st_size > 0


def test_noise_sweep(small_dataset, tmp_path: Path) -> None:
    arrays, meta = small_dataset
    report = noise_sweep(arrays, meta, sigmas=(0.0, 0.2), cv=3, seed=0)
    assert report["sigmas"] == [0.0, 0.2]
    assert all(0.0 <= value <= 1.0 for value in report["auc"])
    plot_noise_sweep(report, tmp_path / "noise.png")
    assert (tmp_path / "noise.png").stat().st_size > 0


def test_plot_eq13_and_ablation(tmp_path: Path) -> None:
    eq13 = {
        "differences": {"C": 1.0e-16, "O": -2.0e-16, "Fe": 0.0},
        "max_abs_difference": 2.0e-16,
        "mean_abs_difference": 1.0e-16,
    }
    plot_eq13(eq13, tmp_path / "eq13.png")
    assert (tmp_path / "eq13.png").stat().st_size > 0

    summary = {
        "base": {
            "acceptance": 0.5,
            "y_rate": 0.6,
            "r1": 0.25,
            "r2": 0.24,
            "r3": 0.01,
        },
        "no_pisn": {
            "acceptance": 0.5,
            "y_rate": 0.0,
            "r1": 0.25,
            "r2": 0.25,
            "r3": 0.0,
        },
    }
    plot_ablation(summary, tmp_path / "ablation.png")
    assert (tmp_path / "ablation.png").stat().st_size > 0
