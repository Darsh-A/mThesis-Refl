"""Tests for excluding an observation element from the model."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.potluck.calibration import active_core_elements, support_gate
from src.potluck.config import ModelConfig, ObservationConfig
from src.potluck.constants import LI_FEATURE_ELEMENTS, REFERENCE
from src.potluck.dataset import generate_dataset
from src.potluck.observation import load_li_records


def test_observation_config_validation():
    with pytest.raises(ValueError):
        ObservationConfig(exclude_elements=("Fe",))
    with pytest.raises(ValueError):
        ObservationConfig(exclude_elements=("O",))
    restored = ModelConfig.from_dict({"observation": {"exclude_elements": []}})
    assert restored.observation.exclude_elements == ()


def test_no_observation_element_is_excluded_by_default():
    config = ModelConfig()
    assert config.observation.exclude_elements == ()
    assert "Ti" in config.observation.active_elements


def test_ti_excluded_from_pool(small_config):
    config = replace(
        small_config,
        observation=ObservationConfig(exclude_elements=("Ti",)),
    )
    arrays, meta = generate_dataset(config, seed=7)
    ratio_elements = tuple(
        element for element in LI_FEATURE_ELEMENTS if element != REFERENCE
    )
    column = ratio_elements.index("Ti")
    assert not np.isfinite(arrays["obs_features"][:, column]).any()
    assert not arrays["obs_flags"][:, column].any()
    assert "Ti/Fe" not in meta["observation_feature_names"]
    assert meta["observation_excluded_elements"] == ("Ti",)
    assert active_core_elements(arrays) == ("Na", "Mg", "Ca", "Cr", "Ni")


def test_support_gate_ignores_excluded_element(small_config):
    config = replace(
        small_config,
        observation=ObservationConfig(exclude_elements=("Ti",)),
    )
    arrays, _ = generate_dataset(config, seed=8)
    distance, threshold = support_gate(load_li_records(), arrays)
    assert np.isfinite(threshold)
    assert np.isfinite(distance).sum() > 0
