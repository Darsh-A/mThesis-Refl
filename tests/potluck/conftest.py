"""Shared fixtures for the potluck tests."""

from __future__ import annotations

import pytest

from src.potluck.config import (
    ModelConfig,
    ObservationConfig,
    RuntimeConfig,
    YieldConfig,
)
from src.potluck.dataset import generate_dataset


@pytest.fixture(scope="session")
def small_config(tmp_path_factory: pytest.TempPathFactory) -> ModelConfig:
    """A small, cached configuration for fast integration tests."""
    cache = tmp_path_factory.mktemp("potluck-cache")
    output = tmp_path_factory.mktemp("potluck-output")
    return ModelConfig(
        yields=YieldConfig(popii_z_grid_size=4, popii_t_grid_size=4),
        observation=ObservationConfig(exclude_elements=()),
        runtime=RuntimeConfig(
            n_samples=30,
            chunk_size=20,
            max_draw_factor=300,
            cache_dir=str(cache),
            output_dir=str(output),
        ),
    )


@pytest.fixture(scope="session")
def small_dataset(small_config: ModelConfig):
    """One generated dataset shared by the storage and diagnostics tests."""
    return generate_dataset(small_config, seed=123)
