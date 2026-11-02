"""Noiseless abundance features and PISN mass-fraction labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ModelConfig
from .constants import (
    ELEMENTS,
    ELEMENT_INDEX,
    FE_INDEPENDENT_RATIOS,
    METALS,
    SOLAR_MASS_FRACTION,
)
from .ejecta import PISN, Ejecta


@dataclass
class Features:
    """Log abundances, abundance ratios, and PISN labels."""

    log_abundance: np.ndarray
    ratios: np.ndarray
    ratio_mask: np.ndarray
    independent_ratios: np.ndarray
    independent_mask: np.ndarray
    f_zp: np.ndarray
    f_fep: np.ndarray
    f_xp: np.ndarray
    ratio_names: tuple[str, ...]
    independent_names: tuple[str, ...]


def build_features(ejecta: Ejecta, config: ModelConfig) -> Features:
    """Compute noiseless features and labels from the ejecta matrix."""
    n = len(ejecta)
    solar = np.asarray([SOLAR_MASS_FRACTION[element] for element in ELEMENTS])
    mass_ratio = np.full_like(ejecta.M_tot, np.nan)
    np.divide(ejecta.M_tot, solar[None, :], out=mass_ratio, where=ejecta.M_tot > 0.0)
    log_abundance = np.log10(mass_ratio)

    reference = config.reference
    ref_index = ELEMENT_INDEX[reference]
    feature_position = {
        element: index for index, element in enumerate(config.feature_elements)
    }
    feature_elements = tuple(
        element for element in config.feature_elements if element != reference
    )
    feature_indices = np.asarray(
        [ELEMENT_INDEX[element] for element in feature_elements]
    )
    feature_positions = np.asarray(
        [feature_position[element] for element in feature_elements]
    )
    ref_position = feature_position[reference]

    ratios = (
        log_abundance[:, feature_indices]
        - log_abundance[:, ref_index][:, None]
    )
    ratio_mask = (
        ejecta.valid_features[:, feature_positions]
        & ejecta.valid_features[:, ref_position][:, None]
        & np.isfinite(ratios)
    )
    ratios = np.where(ratio_mask, ratios, np.nan)

    independent_ratios = np.full(
        (n, len(FE_INDEPENDENT_RATIOS)), np.nan, dtype=float
    )
    independent_mask = np.zeros_like(independent_ratios, dtype=bool)
    for column, (numerator, denominator) in enumerate(FE_INDEPENDENT_RATIOS):
        if numerator not in feature_position or denominator not in feature_position:
            continue
        values = (
            log_abundance[:, ELEMENT_INDEX[numerator]]
            - log_abundance[:, ELEMENT_INDEX[denominator]]
        )
        valid = (
            ejecta.valid_features[:, feature_position[numerator]]
            & ejecta.valid_features[:, feature_position[denominator]]
            & np.isfinite(values)
        )
        independent_ratios[valid, column] = values[valid]
        independent_mask[:, column] = valid

    metal_indices = [ELEMENT_INDEX[element] for element in METALS]
    pisn = ejecta.matrix[:, PISN, :]
    f_xp = np.full_like(ejecta.M_tot, np.nan)
    np.divide(pisn, ejecta.M_tot, out=f_xp, where=ejecta.M_tot > 0.0)

    f_zp = np.full(n, np.nan, dtype=float)
    np.divide(
        pisn[:, metal_indices].sum(axis=1),
        ejecta.M_z_tot,
        out=f_zp,
        where=ejecta.M_z_tot > 0.0,
    )

    f_fep = np.full(n, np.nan, dtype=float)
    np.divide(
        pisn[:, ELEMENT_INDEX["Fe"]],
        ejecta.M_fe,
        out=f_fep,
        where=ejecta.M_fe > 0.0,
    )

    return Features(
        log_abundance=log_abundance,
        ratios=ratios,
        ratio_mask=ratio_mask,
        independent_ratios=independent_ratios,
        independent_mask=independent_mask,
        f_zp=f_zp,
        f_fep=f_fep,
        f_xp=f_xp,
        ratio_names=tuple(f"{element}/{reference}" for element in feature_elements),
        independent_names=tuple(
            f"{numerator}/{denominator}"
            for numerator, denominator in FE_INDEPENDENT_RATIOS
        ),
    )
