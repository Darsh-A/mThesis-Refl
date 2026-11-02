"""Tests for the Salvadori cross-check and ablation scaffolding."""

from __future__ import annotations

from src.potluck.diagnostics import diagnose
from src.potluck.validation import ablation_configs, eq13_check, validate_dataset


def test_eq13_matches_legacy_implementation(small_config):
    result = eq13_check(small_config)
    assert result["max_abs_difference"] is not None
    assert result["max_abs_difference"] < 1e-10
    assert result["z_star"] > 0.0
    assert result["m_star_ii"] > 0.0


def test_validate_dataset_passes(small_dataset):
    arrays, meta = small_dataset
    result = validate_dataset(arrays, meta)
    assert result["ok"]


def test_ablation_variants_change_one_prior_at_a_time(small_config):
    variants = ablation_configs(small_config)
    assert set(variants) == {
        "base",
        "no_pisn",
        "no_iii",
        "wide_lambda",
        "no_r4",
        "low_zcr",
        "zcr_1e-6",
        "cap_0.05",
    }
    assert variants["base"] is small_config
    assert variants["no_pisn"].prior.n_pisn_probs == (1.0, 0.0, 0.0, 0.0)
    assert max(variants["no_iii"].prior.lambda_iii_range) < 1e-8
    assert variants["wide_lambda"].prior.n_pisn_probs == (0.25, 0.25, 0.25, 0.25)
    assert variants["no_r4"].prior.feh_range is None
    assert variants["low_zcr"].prior.z_cr == 1.0e-4
    assert variants["zcr_1e-6"].prior.z_cr == 1.0e-6
    assert variants["cap_0.05"].prior.mstar_total_max_ratio == 0.05
