"""Li+ observation model: masks, errors, limits, and [Fe/H] weighting."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from src.load_data import load_obs_li
from src.utils import _ion_to_element

from .config import ModelConfig
from .constants import ELEMENT_INDEX, LI_FEATURE_ELEMENTS, REFERENCE
from .ejecta import Ejecta
from .features import Features

MISSING = 0
DETECTED = 1
LIMIT = 2
GIANT_LOGG = 3.5


@dataclass(frozen=True)
class LIRecords:
    """Per-star Li+ masks, errors, and limits in catalogue order."""

    labels: tuple[str, ...]
    feh: np.ndarray
    e_feh: np.ndarray
    logg: np.ndarray
    snr: np.ndarray
    flags: np.ndarray
    sigma: np.ndarray
    limit: np.ndarray
    xfe: np.ndarray

    @property
    def n(self) -> int:
        return int(self.feh.size)


@dataclass
class Observation:
    """Observed Li+-style ratios and their metadata."""

    ratios: np.ndarray
    flags: np.ndarray
    sigma: np.ndarray
    mask: np.ndarray
    independent_ratios: np.ndarray
    independent_mask: np.ndarray
    independent_sigma: np.ndarray
    feh_obs: np.ndarray
    feh_err: np.ndarray
    logg: np.ndarray
    is_giant: np.ndarray
    record_index: np.ndarray
    weight: np.ndarray
    feh_true: np.ndarray
    ratio_names: tuple[str, ...]
    independent_names: tuple[str, ...]

    def subset(self, index: np.ndarray) -> "Observation":
        """Slice the observation bundle by integer index."""
        return replace(
            self,
            ratios=self.ratios[index],
            flags=self.flags[index],
            sigma=self.sigma[index],
            mask=self.mask[index],
            independent_ratios=self.independent_ratios[index],
            independent_mask=self.independent_mask[index],
            independent_sigma=self.independent_sigma[index],
            feh_obs=self.feh_obs[index],
            feh_err=self.feh_err[index],
            logg=self.logg[index],
            is_giant=self.is_giant[index],
            record_index=self.record_index[index],
            weight=self.weight[index],
            feh_true=self.feh_true[index],
        )

    @property
    def effective_sample_size(self) -> float:
        """Kish effective sample size of the importance weights."""
        total = float(self.weight.sum())
        squared = float(np.square(self.weight).sum())
        return total * total / squared if squared > 0.0 else 0.0


@lru_cache(maxsize=1)
def load_li_records() -> LIRecords:
    """Load the Li+ catalogue into fixed element-order arrays."""
    entries = load_obs_li()
    n = len(entries)
    elements = LI_FEATURE_ELEMENTS
    feh = np.full(n, np.nan)
    e_feh = np.full(n, np.nan)
    logg = np.full(n, np.nan)
    snr = np.full(n, np.nan)
    flags = np.zeros((n, len(elements)), dtype=np.int8)
    sigma = np.full((n, len(elements)), np.nan)
    limit = np.full((n, len(elements)), np.nan)
    xfe = np.full((n, len(elements)), np.nan)
    labels = []

    for i, entry in enumerate(entries):
        labels.append(entry["label"])
        params = entry["params"]
        feh[i] = float(params["FeH"])
        e_feh[i] = float(params["e_FeH"])
        logg[i] = float(params["logg"])
        snr[i] = float(params["SNR"])

        for j, element in enumerate(elements):
            if element == REFERENCE:
                flags[i, j] = DETECTED
                continue
            records = [
                record
                for ion, record in entry["abundances"].items()
                if _ion_to_element(ion) == element
            ]
            records = [
                record
                for record in records
                if record.get("XFe") is not None
                or record.get("l_XFe") == "<"
            ]
            if not records:
                continue
            record = min(
                records,
                key=lambda item: item.get("sigma_Total") or np.inf,
            )
            value = record.get("XFe")
            if record.get("l_XFe") == "<":
                flags[i, j] = LIMIT
                limit[i, j] = float(value)
                xfe[i, j] = float(value)
            elif value is not None:
                flags[i, j] = DETECTED
                xfe[i, j] = float(value)
            error = record.get("sigma_Total")
            if error is not None and np.isfinite(error):
                sigma[i, j] = float(error)

    return LIRecords(
        labels=tuple(labels),
        feh=feh,
        e_feh=e_feh,
        logg=logg,
        snr=snr,
        flags=flags,
        sigma=sigma,
        limit=limit,
        xfe=xfe,
    )


def sample_records(
    feh_true: np.ndarray,
    records: LIRecords,
    rng: np.random.Generator,
    kernel_dex: float,
) -> np.ndarray:
    """Pick a real record whose [Fe/H] is within ``kernel_dex`` of the mock."""
    order = np.argsort(records.feh)
    sorted_feh = records.feh[order]
    chosen = np.zeros(feh_true.size, dtype=np.int64)
    for i, value in enumerate(feh_true):
        left = int(np.searchsorted(sorted_feh, value - kernel_dex, side="left"))
        right = int(np.searchsorted(sorted_feh, value + kernel_dex, side="right"))
        if right > left:
            chosen[i] = order[rng.integers(left, right)]
        else:
            nearest = int(np.searchsorted(sorted_feh, value))
            nearest = min(max(nearest, 0), sorted_feh.size - 1)
            chosen[i] = order[nearest]
    return chosen


def _c_threshold(
    records: LIRecords, default_sigma: float
) -> tuple[float, float]:
    """Fit the C upper-limit threshold in [C/H] against sigma_C."""
    column = LI_FEATURE_ELEMENTS.index("C")
    selected = records.flags[:, column] == LIMIT
    selected &= np.isfinite(records.limit[:, column]) & np.isfinite(records.feh)
    if np.count_nonzero(selected) == 0:
        return 0.0, -np.inf
    if np.count_nonzero(selected) == 1:
        value = records.limit[selected, column][0] + records.feh[selected][0]
        return 0.0, float(value)
    sigma = records.sigma[selected, column]
    sigma = np.where(np.isfinite(sigma), sigma, default_sigma)
    threshold = records.limit[selected, column] + records.feh[selected]
    design = np.column_stack([sigma, np.ones_like(sigma)])
    slope, intercept = np.linalg.lstsq(design, threshold, rcond=None)[0]
    return float(slope), float(intercept)


def _non_detection_thresholds(
    records: LIRecords,
    elements: tuple[str, ...],
    quantile: float,
    n_bins: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build a monotone lower-envelope [X/H] threshold against log SNR."""
    thresholds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for element in elements:
        if element == REFERENCE or element == "C":
            continue
        column = LI_FEATURE_ELEMENTS.index(element)
        selected = (
            (records.flags[:, column] == DETECTED)
            & np.isfinite(records.xfe[:, column])
            & np.isfinite(records.feh)
            & np.isfinite(records.snr)
            & (records.snr > 0.0)
        )
        if np.count_nonzero(selected) < n_bins:
            continue
        snr = records.snr[selected]
        xh = records.xfe[selected, column] + records.feh[selected]
        order = np.argsort(snr)
        snr = snr[order]
        xh = xh[order]
        chunks = np.array_split(np.arange(snr.size), n_bins)
        x = np.array([np.log10(np.median(snr[chunk])) for chunk in chunks])
        y = np.array([np.quantile(xh[chunk], quantile) for chunk in chunks])
        finite = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(finite) < 2:
            continue
        x = x[finite]
        y = np.minimum.accumulate(y[finite])
        thresholds[element] = (x, y)
    return thresholds


def sample_observation(
    features: Features,
    ejecta: Ejecta,
    config: ModelConfig,
    rng: np.random.Generator,
    records: LIRecords | None = None,
) -> Observation:
    """Draw a real Li+ record, add element-level ratio noise, and censor."""
    settings = config.observation
    records = records if records is not None else load_li_records()
    n = len(ejecta)
    chosen = sample_records(
        ejecta.feh, records, rng, settings.feh_kernel_dex
    )
    feh_err = records.e_feh[chosen]
    feh_err = np.where(np.isfinite(feh_err), feh_err, 0.1)
    feh_obs = ejecta.feh + rng.normal(0.0, feh_err)
    logg = records.logg[chosen]
    is_giant = logg < GIANT_LOGG

    elements = tuple(settings.feature_elements)
    ratio_elements = tuple(e for e in elements if e != REFERENCE)
    ratio_names = tuple(f"{element}/{REFERENCE}" for element in ratio_elements)
    ratios = np.full((n, len(ratio_elements)), np.nan)
    flags = np.zeros((n, len(ratio_elements)), dtype=np.int8)
    sigma = np.full((n, len(ratio_elements)), np.nan)
    element_sigma = np.full((n, len(elements)), np.nan)
    element_flag = np.zeros((n, len(elements)), dtype=np.int8)

    slope, intercept = _c_threshold(records, settings.default_sigma_dex)
    thresholds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if settings.non_detection_mode == "threshold":
        thresholds = _non_detection_thresholds(
            records,
            settings.non_detection_elements,
            settings.non_detection_quantile,
            settings.non_detection_snr_bins,
        )
    snr = records.snr[chosen]
    snr = np.where(np.isfinite(snr) & (snr > 0.0), snr, 10.0)

    for j, element in enumerate(elements):
        column = LI_FEATURE_ELEMENTS.index(element)
        if element == REFERENCE:
            element_flag[:, j] = DETECTED
            element_sigma[:, j] = 0.0
            continue
        if element in settings.exclude_elements:
            element_flag[:, j] = MISSING
            continue
        flag = records.flags[chosen, column].copy()
        error = records.sigma[chosen, column].copy()
        error = np.where(np.isfinite(error), error, settings.default_sigma_dex)

        if element == "C":
            if settings.c_mode == "drop":
                flag[is_giant] = MISSING
            elif settings.c_mode == "inflate":
                error[is_giant] = np.sqrt(
                    error[is_giant] ** 2 + settings.c_giant_sigma_dex**2
                )

        true_ratio = (
            features.log_abundance[:, ELEMENT_INDEX[element]]
            - features.log_abundance[:, ELEMENT_INDEX[REFERENCE]]
        )
        if element in thresholds:
            x_knots, y_knots = thresholds[element]
            threshold = np.interp(np.log10(snr), x_knots, y_knots)
            flag = flag.copy()
            flag[(true_ratio + ejecta.feh) < threshold] = MISSING

        element_flag[:, j] = flag
        element_sigma[:, j] = error
        noisy = true_ratio + rng.normal(0.0, error)
        value = noisy.copy()

        censored = flag == LIMIT
        if np.any(censored):
            threshold = slope * error[censored] + intercept
            observed_xh = noisy[censored] + feh_obs[censored]
            below = observed_xh < threshold
            value[censored] = np.where(
                below,
                threshold - feh_obs[censored],
                noisy[censored],
            )
            flag[censored] = np.where(below, LIMIT, DETECTED)

        ratio_column = ratio_elements.index(element)
        ratios[:, ratio_column] = np.where(flag == MISSING, np.nan, value)
        flags[:, ratio_column] = flag
        sigma[:, ratio_column] = np.where(
            flag == MISSING, np.nan, error
        )

    independent_names = tuple(
        f"{numerator}/{denominator}"
        for numerator, denominator in _independent_pairs(elements)
    )
    independent = np.full((n, len(independent_names)), np.nan)
    independent_mask = np.zeros_like(independent, dtype=bool)
    independent_sigma = np.full_like(independent, np.nan)

    ratio_index = {element: i for i, element in enumerate(ratio_elements)}
    for column, (numerator, denominator) in enumerate(_independent_pairs(elements)):
        num = elements.index(numerator)
        den = elements.index(denominator)
        available = (
            (element_flag[:, num] != MISSING)
            & (element_flag[:, den] != MISSING)
        )
        if numerator == REFERENCE:
            value = -ratios[:, ratio_index[denominator]]
        elif denominator == REFERENCE:
            value = ratios[:, ratio_index[numerator]]
        else:
            value = (
                ratios[:, ratio_index[numerator]]
                - ratios[:, ratio_index[denominator]]
            )
        error = np.sqrt(
            element_sigma[:, num] ** 2 + element_sigma[:, den] ** 2
        )
        independent[:, column] = np.where(available, value, np.nan)
        independent_mask[:, column] = available
        independent_sigma[:, column] = np.where(available, error, np.nan)

    observation = Observation(
        ratios=ratios,
        flags=flags,
        sigma=sigma,
        mask=flags > MISSING,
        independent_ratios=independent,
        independent_mask=independent_mask,
        independent_sigma=independent_sigma,
        feh_obs=feh_obs,
        feh_err=feh_err,
        logg=logg,
        is_giant=is_giant,
        record_index=chosen,
        weight=np.ones(n),
        feh_true=ejecta.feh,
        ratio_names=ratio_names,
        independent_names=independent_names,
    )
    if settings.weight_by_feh:
        observation = match_feh_weights(observation, records, config)
    return observation


def match_feh_weights(
    observation: Observation,
    records: LIRecords,
    config: ModelConfig,
) -> Observation:
    """Reweight accepted mocks to the Li+ [Fe/H] histogram by giant/dwarf."""
    bounds = config.prior.feh_range
    if bounds is None:
        return observation
    lo, hi = bounds
    bins = np.linspace(lo, hi, 21)
    weights = np.ones(observation.feh_obs.size)
    target = (records.feh >= lo) & (records.feh <= hi)
    for giant in (False, True):
        mock_group = observation.is_giant == giant
        target_group = target & ((records.logg < GIANT_LOGG) == giant)
        if not np.any(mock_group) or not np.any(target_group):
            continue
        target_hist, _ = np.histogram(records.feh[target_group], bins=bins)
        mock_hist, _ = np.histogram(observation.feh_obs[mock_group], bins=bins)
        target_density = target_hist / max(target_hist.sum(), 1)
        mock_density = mock_hist / max(mock_hist.sum(), 1)
        index = np.clip(np.digitize(observation.feh_obs[mock_group], bins) - 1, 0, bins.size - 2)
        ratio = target_density[index] / np.maximum(mock_density[index], 1e-6)
        ratio = np.clip(ratio, 1.0 / config.observation.weight_clip, config.observation.weight_clip)
        ratio /= np.mean(ratio)
        weights[mock_group] = ratio
    return replace(observation, weight=weights)


def apply_test_offset(
    observation: Observation, element: str, offset_dex: float
) -> Observation:
    """Apply a catalogue-level offset at test time only."""
    ratios = observation.ratios.copy()
    independent = observation.independent_ratios.copy()
    for column, name in enumerate(observation.ratio_names):
        if name.split("/")[0] == element:
            ratios[:, column] += offset_dex
    for column, name in enumerate(observation.independent_names):
        if element in name.split("/"):
            independent[:, column] += offset_dex
    return replace(observation, ratios=ratios, independent_ratios=independent)


def within_feh_range(
    observation: Observation, config: ModelConfig
) -> np.ndarray:
    """Observed-[Fe/H] R4 box."""
    bounds = config.prior.feh_range
    if bounds is None:
        return np.isfinite(observation.feh_obs)
    lo, hi = bounds
    return (
        np.isfinite(observation.feh_obs)
        & (observation.feh_obs >= lo)
        & (observation.feh_obs <= hi)
    )


def _independent_pairs(
    elements: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    """Fe-independent pairs available in the observation element set."""
    from .constants import FE_INDEPENDENT_RATIOS

    return tuple(
        pair
        for pair in FE_INDEPENDENT_RATIOS
        if pair[0] in elements and pair[1] in elements
    )
