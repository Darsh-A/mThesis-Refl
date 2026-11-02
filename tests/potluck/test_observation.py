"""Checks for the Li+ observation model."""

from dataclasses import replace

import numpy as np

from src.potluck.constants import ELEMENT_INDEX, LI_FEATURE_ELEMENTS
from src.potluck.observation import (
    DETECTED,
    LIMIT,
    LIRecords,
    apply_test_offset,
    load_li_records,
    sample_observation,
    within_feh_range,
)
from src.potluck.config import ModelConfig


def test_li_records_cover_elements_and_limits() -> None:
    records = load_li_records()
    assert records.n == 385
    assert records.flags.shape[1] == 12
    assert np.any(records.flags == LIMIT)
    assert np.all(np.isfinite(records.feh))
    assert np.all(np.isfinite(records.logg))


def test_small_dataset_observation_arrays(small_dataset) -> None:
    arrays, meta = small_dataset
    n = len(arrays["y_f_05"])
    assert arrays["obs_features"].shape[0] == n
    assert arrays["obs_flags"].shape == arrays["obs_features"].shape
    assert arrays["obs_mask"].shape == arrays["obs_features"].shape
    assert np.all(np.isin(arrays["obs_flags"], [0, 1, 2]))
    assert np.array_equal(
        arrays["obs_mask"], arrays["obs_flags"] > 0
    )
    assert np.all(np.isfinite(arrays["feh_obs"]))
    assert np.all(arrays["importance_weight"] > 0.0)
    assert 0.0 < (arrays["importance_weight"].sum() ** 2) / (
        arrays["importance_weight"] ** 2
    ).sum() <= n
    assert meta["observation_feature_names"]


def test_within_feh_range() -> None:
    class Dummy:
        feh_obs = np.array([-4.0, -3.0, -1.0])

    config = ModelConfig()
    assert within_feh_range(Dummy(), config).tolist() == [False, True, False]


def test_na_offset_is_test_time_only(small_dataset) -> None:
    from src.potluck.observation import Observation

    arrays, _ = small_dataset
    observation = Observation(
        ratios=arrays["obs_features"].astype(float),
        flags=arrays["obs_flags"],
        sigma=arrays["obs_sigma"].astype(float),
        mask=arrays["obs_mask"],
        independent_ratios=arrays["obs_independent_features"].astype(float),
        independent_mask=arrays["obs_independent_mask"],
        independent_sigma=arrays["obs_independent_sigma"].astype(float),
        feh_obs=arrays["feh_obs"].astype(float),
        feh_err=arrays["feh_err"].astype(float),
        logg=arrays["logg"].astype(float),
        is_giant=arrays["is_giant"],
        record_index=arrays["record_index"],
        weight=arrays["importance_weight"].astype(float),
        feh_true=arrays["feh"].astype(float),
        ratio_names=tuple(
            f"{element}/Fe" for element in ("C", "Na", "Mg", "Si", "Ca", "Ti", "Cr", "Mn", "Co", "Ni", "Zn")
        ),
        independent_names=("Na/Mg", "Co/Ni", "Mn/Cr", "Zn/Mg"),
    )
    shifted = apply_test_offset(observation, "Na", 0.1)
    column = observation.ratio_names.index("Na/Fe")
    finite = np.isfinite(observation.ratios[:, column])
    assert np.allclose(
        shifted.ratios[finite, column],
        observation.ratios[finite, column] + 0.1,
    )
    assert np.array_equal(
        np.isfinite(shifted.ratios[:, column]), finite
    )


def test_independent_ratios_share_ratio_noise(small_dataset) -> None:
    arrays, _ = small_dataset
    ratios = arrays["obs_features"].astype(float)
    independent = arrays["obs_independent_features"].astype(float)
    mask = arrays["obs_independent_mask"]
    ratio_names = tuple(
        f"{element}/Fe"
        for element in ("C", "Na", "Mg", "Si", "Ca", "Ti", "Cr", "Mn", "Co", "Ni", "Zn")
    )
    for column, (numerator, denominator) in enumerate(
        (("Na", "Mg"), ("Co", "Ni"), ("Mn", "Cr"), ("Zn", "Mg"))
    ):
        num = ratio_names.index(f"{numerator}/Fe")
        den = ratio_names.index(f"{denominator}/Fe")
        expected = ratios[:, num] - ratios[:, den]
        assert np.allclose(
            independent[mask[:, column], column],
            expected[mask[:, column]],
            equal_nan=True,
        )


def test_low_abundance_forced_missing() -> None:
    n = 20
    elements = LI_FEATURE_ELEMENTS
    xfe = np.full((n, len(elements)), -2.0)
    zn = elements.index("Zn")
    xfe[:, zn] = np.linspace(-0.5, -0.4, n)
    records = LIRecords(
        labels=tuple(f"star{i}" for i in range(n)),
        feh=np.full(n, -3.0),
        e_feh=np.full(n, 0.1),
        logg=np.full(n, 4.5),
        snr=np.linspace(10.0, 100.0, n),
        flags=np.full((n, len(elements)), DETECTED, dtype=np.int8),
        sigma=np.full((n, len(elements)), 0.1),
        limit=np.full((n, len(elements)), np.nan),
        xfe=xfe,
    )

    class DummyEjecta:
        feh = np.full(n, -3.0)

        def __len__(self) -> int:
            return n

    class DummyFeatures:
        log_abundance = np.zeros((n, 32))
        log_abundance[:, ELEMENT_INDEX["Zn"]] = -10.0

    config = replace(
        ModelConfig(),
        observation=replace(
            ModelConfig().observation,
            non_detection_elements=("Zn",),
        ),
    )
    observation = sample_observation(
        DummyFeatures(), DummyEjecta(), config, np.random.default_rng(1), records
    )
    column = observation.ratio_names.index("Zn/Fe")
    assert np.all(observation.flags[:, column] == 0)
