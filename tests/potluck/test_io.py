"""Round-trip and provenance tests for stored datasets."""

from __future__ import annotations

import numpy as np

from src.potluck.config import ModelConfig
from src.potluck.io import config_hash, load_dataset, save_dataset


def test_save_load_roundtrip(tmp_path, small_dataset):
    arrays, meta = small_dataset
    path = tmp_path / "dataset.npz"
    save_dataset(path, arrays, meta)

    assert path.exists()
    assert path.with_suffix(".meta.json").exists()

    loaded, loaded_meta = load_dataset(path)
    assert set(loaded) == set(arrays)
    for name, values in arrays.items():
        if values.dtype.kind == "f":
            np.testing.assert_allclose(loaded[name], values, equal_nan=True)
        else:
            np.testing.assert_array_equal(loaded[name], values)
    assert loaded_meta["config_hash"] == meta["config_hash"]


def test_load_without_suffix(tmp_path, small_dataset):
    arrays, meta = small_dataset
    path = tmp_path / "dataset.npz"
    save_dataset(path, arrays, meta)
    loaded, _ = load_dataset(tmp_path / "dataset")
    assert len(loaded["y"]) == len(arrays["y"])


def test_config_hash_is_stable(small_dataset):
    _, meta = small_dataset
    stored = ModelConfig.from_dict(meta["config"])
    assert config_hash(stored) == meta["config_hash"]
    assert len(meta["config_hash"]) == 16
    assert meta["format_version"] == 1
