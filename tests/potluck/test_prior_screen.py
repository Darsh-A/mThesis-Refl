"""Tests for the C3 prior screen."""

from __future__ import annotations

from collections import Counter

import pytest

from src.potluck.calibration import CORE_SUPPORT_ELEMENTS
from src.potluck.config import ModelConfig
from src.potluck.diagnostics.prior_screen import (
    FACTOR_LEVELS,
    _level_label,
    _main_effects,
    _variant,
    design_points,
    run_confirmation,
    run_prior_screen,
)


def test_design_points():
    points = design_points(n_points=5, seed=0)
    assert len(points) == 6
    assert points[-1]["label"] == "baseline"
    for factor, levels in FACTOR_LEVELS.items():
        counts = Counter(point["levels"][factor] for point in points[:-1])
        assert set(counts) == set(levels)
        assert set(counts.values()) == {1}
        assert points[-1]["levels"][factor] == levels[0]
    with pytest.raises(ValueError):
        design_points(n_points=0)


def test_variant():
    points = design_points(n_points=5, seed=0)
    levels = points[0]["levels"]
    config = _variant(ModelConfig(), levels)
    assert config.prior.pisn_mass_range == tuple(levels["pisn_mass"])
    assert config.prior.mdil_range == tuple(levels["mdil"])
    assert config.prior.z_cr == levels["z_cr"]
    assert config.prior.lambda_iii_range == tuple(levels["lambda_iii"])
    assert config.runtime.max_draw_factor >= 20

    row_levels = dict(levels, popiii="row")
    row_config = _variant(ModelConfig(), row_levels)
    assert row_config.prior.popiii_sampling == "row"
    assert row_config.prior.popiii_mass_energy_weights is None


def test_main_effects():
    base = {factor: levels[0] for factor, levels in FACTOR_LEVELS.items()}
    second = {factor: levels[1] for factor, levels in FACTOR_LEVELS.items()}
    points = [
        {"levels": base, "metrics": {"in_model_fraction": 0.25}},
        {"levels": second, "metrics": {"in_model_fraction": 0.75}},
        {"levels": base, "metrics": None},
    ]
    effects = _main_effects(points)
    assert set(effects) == set(FACTOR_LEVELS)
    for factor, levels in FACTOR_LEVELS.items():
        first = effects[factor][_level_label(levels[0])]
        assert first["n"] == 1
        assert first["mean_in_model"] == pytest.approx(0.25)
        assert effects[factor][_level_label(levels[1])]["mean_in_model"] == pytest.approx(0.75)


def test_run_prior_screen_smoke(tmp_path):
    summary = run_prior_screen(
        n_samples=150,
        n_points=2,
        seed=1,
        outdir=tmp_path,
    )
    assert summary["n_points"] == 3
    assert len(summary["points"]) == 3
    assert summary["baseline"]["label"] == "baseline"
    assert set(summary["effects"]) == set(FACTOR_LEVELS)
    valid = [point for point in summary["points"] if point["metrics"] is not None]
    assert valid
    for point in valid:
        assert set(point["metrics"]["core_elements"]) == set(CORE_SUPPORT_ELEMENTS)
    assert (tmp_path / "prior_screen.json").is_file()
    assert (tmp_path / "prior_screen.png").stat().st_size > 0


def test_run_confirmation_smoke(tmp_path):
    summary = run_confirmation(
        labels=("baseline", "p11"),
        n_samples=120,
        seeds=(1,),
        seed=0,
        outdir=tmp_path,
    )
    assert summary["n_samples"] == 120
    assert set(summary["aggregate"]) == {"baseline", "p11"}
    assert summary["aggregate"]["baseline"]["n_runs"] == 1
    assert len(summary["runs"]) == 2
    assert (tmp_path / "prior_screen_confirmation.json").is_file()
    assert (tmp_path / "prior_screen_confirmation.png").stat().st_size > 0
