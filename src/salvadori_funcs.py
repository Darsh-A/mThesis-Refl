# Contains formula to replicate Salvadori et al. 2019

import copy
import json

import numpy as np

from .utils import _combine_elements
from params import asplund, atomic_mass


def salvadori_yields(data: list[dict]) -> list[dict]:
    """Compute yield in salvadori2019 sense
        Yx = isotope_mass/M_pisn

    Returns a NEW list of dicts; the input is not modified.

    Args:
        data: A list of dictionaries containing yield data.
        [
            {
                "label": "M=65",
                "params": {"mass":130, "mass_he": 65},
                "yields": {"H1": 0.1, "He4": 0.2, ...},
            },
            ...
        ]
    """
    result = copy.deepcopy(data)
    for entry in result:
        mass = entry["params"]["mass"]
        yields = entry["yields"]

        for isotope, yield_mass in yields.items():
            yields[isotope] = yield_mass / mass

    return result


def salvadori_Zism(f_ratio: float, data: list[dict]) -> list[dict]:
    """Compute Z_ism in salvadori2019 sense
        Z_ism = (f_star/f_dil) * Yx

    Returns a NEW list of dicts; the input is not modified.

    Args:
        f_ratio: The ratio of f_star to f_dil.
        data: A list of dictionaries containing yield data.
        [
            {
                "label": "M=65",
                "params": {"mass":130, "mass_he": 65},
                "yields": {"H1": 0.1, "He4": 0.2, ...},
            },
            ...
        ]
    """
    result = copy.deepcopy(data)
    for entry in result:
        yields = entry["yields"]

        for isotope, yield_mass in yields.items():
            yields[isotope] = f_ratio * yield_mass

    return result


def salvadori_abundance_ratio(elem1: str, elem2: str, entry: dict) -> float:
    """Compute the abundance ratio [elem1/elem2] for a single PISN entry in the salvadori2019 sense.

        [X/Y] = log10((Y_X / A_X) / (Y_Y / A_Y)) - (A(X) - A(Y))

    where Y_X is the mass yield of element X (summed over its isotopes),
    A_X is the atomic mass of X, and A(X) is the Asplund+09 solar abundance.
    The M_pisn normalization from salvadori_yields cancels in the ratio, so
    the yields may be raw masses or already normalized.

    Args:
        elem1: First element symbol (numerator), e.g. "Fe".
        elem2: Second element symbol (denominator), e.g. "H".
        entry: A single dict with a "yields" mapping (isotope- or element-level).

    Returns:
        The abundance ratio [elem1/elem2].
    """
    yields = _combine_elements(entry["yields"])

    if elem1 not in yields or elem2 not in yields:
        raise ValueError(f"Element {elem1} or {elem2} not found in yields.")

    with open(asplund, "r") as f:
        solar = json.load(f)

    A1 = solar[elem1]["val"]
    A2 = solar[elem2]["val"]

    abundance_ratio = np.log10((yields[elem1]) / (yields[elem2])) - (A1 - A2)

    return abundance_ratio

def salvadori_H_ratio(elem1: str, f_ratio: float, entry: dict) -> float:
    """From Eq.3 of Salvadori et al. 2019

        [X/H] = log10(f_ratio * Y_X) - (A(X) - A(H))

    Args:
        elem1: Element symbol (numerator), e.g. "Fe".
        entry: A single dict with a "yields" mapping (isotope- or element-level).

    Returns:
        The abundance ratio [elem1/H].
    """
    yields = _combine_elements(entry["yields"])

    if elem1 not in yields or "H" not in yields:
        raise ValueError(f"Element {elem1} or H not found in yields.")

    with open(asplund, "r") as f:
        solar = json.load(f)

    A1 = solar[elem1]["val"]
    A2 = solar["H"]["val"]

    abundance_ratio = np.log10(f_ratio * yields[elem1]) - (A1 - A2)

    return abundance_ratio