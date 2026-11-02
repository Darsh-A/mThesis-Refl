"""Counterfactual PISN detectability from the observed abundance pattern."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ModelConfig
from .constants import ELEMENTS, ELEMENT_INDEX, SOLAR_MASS_FRACTION
from .ejecta import PISN, Ejecta
from .features import Features

DETECT_REASONS: tuple[str, ...] = (
    "normal",
    "baseline_zero",
    "insufficient",
    "no_pisn",
    "undefined",
)
_NORMAL, _BASELINE_ZERO, _INSUFFICIENT, _NO_PISN, _UNDEFINED = range(5)


@dataclass
class Detection:
    """Standardised PISN-removal distance and its binary label."""

    D: np.ndarray
    D2: np.ndarray
    y: np.ndarray
    reason: np.ndarray
    n_used: np.ndarray


def detect_labels(
    features: Features,
    ejecta: Ejecta,
    config: ModelConfig,
    observation=None,
) -> Detection:
    """Remove the PISN row and test whether the pattern shifts by more than noise.

    If ``observation`` is given, use its detected ratios and diagonal
    Li+-style errors; otherwise use the placeholder constant noise.
    """
    n = len(ejecta)
    D = np.full(n, np.nan, dtype=float)
    D2 = np.full(n, np.nan, dtype=float)
    y = np.zeros(n, dtype=bool)
    reason = np.full(n, _UNDEFINED, dtype=np.int8)
    n_used = np.zeros(n, dtype=np.int64)

    accepted = ejecta.accepted
    has_pisn = ejecta.matrix[:, PISN, :].sum(axis=1) > 0.0
    quiet = accepted & ~has_pisn
    reason[quiet] = _NO_PISN
    D[quiet] = 0.0
    D2[quiet] = 0.0
    reason[accepted & has_pisn] = _NORMAL

    reference = config.reference
    ref_index = ELEMENT_INDEX[reference]
    feature_elements = tuple(
        element for element in config.feature_elements if element != reference
    )
    solar = np.asarray([SOLAR_MASS_FRACTION[element] for element in ELEMENTS])
    sigma = config.detect.sigma_dex
    sigma_ref = config.detect.sigma_fe_dex
    threshold = config.detect.k
    min_used = config.detect.min_detected

    ratio_lookup = {
        name: column for column, name in enumerate(features.ratio_names)
    }
    for index in np.flatnonzero(accepted & has_pisn):
        if observation is None:
            mask = features.ratio_mask[index]
            used_names = [
                f"{feature_elements[j]}/{reference}"
                for j in np.flatnonzero(mask)
            ]
            ratio_values = features.ratios[index, mask]
            sigma_values = None
        else:
            mask = observation.flags[index] == 1
            used_names = [
                observation.ratio_names[j] for j in np.flatnonzero(mask)
            ]
            used_names = [name for name in used_names if name in ratio_lookup]
            ratio_values = np.asarray(
                [features.ratios[index, ratio_lookup[name]] for name in used_names]
            )
            sigma_values = np.asarray(
                [
                    observation.sigma[index, observation.ratio_names.index(name)]
                    for name in used_names
                ]
            )

        count = len(used_names)
        n_used[index] = count
        if count < min_used:
            reason[index] = _INSUFFICIENT
            D[index] = 0.0
            D2[index] = 0.0
            continue

        used_elements = [name.split("/")[0] for name in used_names]
        used_indices = [ELEMENT_INDEX[element] for element in used_elements]
        used_indices.append(ref_index)
        M_minus = ejecta.M_tot[index] - ejecta.matrix[index, PISN, :]
        if np.any(M_minus[used_indices] <= 0.0):
            reason[index] = _BASELINE_ZERO
            D[index] = np.inf
            D2[index] = np.inf
            y[index] = True
            continue

        a_minus = np.log10(M_minus[used_indices] / solar[used_indices])
        x_minus = a_minus[:-1] - a_minus[-1]
        delta = ratio_values - x_minus

        if sigma_values is None:
            covariance = np.full((count, count), sigma_ref**2)
            covariance.flat[:: count + 1] += sigma**2
        else:
            covariance = np.diag(sigma_values**2)
        squared = float(delta @ np.linalg.solve(covariance, delta))
        squared = max(squared, 0.0)
        D2[index] = squared
        D[index] = np.sqrt(squared)
        y[index] = D[index] > threshold

    return Detection(D=D, D2=D2, y=y, reason=reason, n_used=n_used)
