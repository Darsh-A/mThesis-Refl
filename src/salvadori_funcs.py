# Contains formula to replicate Salvadori et al. 2019

import copy
import json

import numpy as np

from .utils import _combine_elements, _get_element
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

    abundance_ratio = np.log10((yields[elem1]) / (yields[elem2])) - (A1 - A2) #- np.log10(atomic_mass[elem1]/atomic_mass[elem2])

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

    abundance_ratio = np.log10(f_ratio * yields[elem1]) - (A1 - A2) - np.log10(atomic_mass[elem1]/atomic_mass["H"])

    return abundance_ratio

def salvadori_combined_abundratio(elem1_pisn: str, elem1_sn: str, elem2_pisn: str, elem2_sn: str, pisn_data: list[dict], sn_data: list[dict], f_pisn: float) -> float:
    """Compute the combined abundance ratio for a mixture of PISN and SN yields.
        Refer to Eq. 13 of Salvadori et al. 2019.

    Args:
        pisn_data:
        {
            "label": "M=65",
            "params": {"mass":130, "mass_he": 65},
            "yields": {"H1": 0.1, "He4": 0.2, ...},
        }
        sn_data: (Same as pisn_data)
    Returns:
        The combined abundance ratio [elem1/elem2].
    """

    pisn_data["yields"] = _combine_elements(pisn_data["yields"])
    sn_data["yields"] = _combine_elements(sn_data["yields"])

    pisn_yields = pisn_data
    sn_yields = sn_data

    if elem1_pisn not in pisn_yields["yields"] or elem2_pisn not in pisn_yields["yields"]:
        raise ValueError(f"Element {elem1_pisn} or {elem2_pisn} not found in PISN yields.")
    if elem1_sn not in sn_yields["yields"] or elem2_sn not in sn_yields["yields"]:
        raise ValueError(f"Element {elem1_sn} or {elem2_sn} not found in SN yields.")
    
    with open(asplund, "r") as f:
        solar = json.load(f)

    A1 = solar[elem1_pisn]["val"]
    A2 = solar[elem2_pisn]["val"]

    beta = (1-f_pisn)/f_pisn

    Yz_pisn = sum(
        e for element, e in pisn_yields["yields"].items()
        if element not in ("H", "He")
    ) / pisn_yields["params"]["mass"]

    Yz_sn = sum(
        e for element, e in sn_yields["yields"].items()
        if element not in ("H", "He")
    ) / sn_yields["params"]["mass"]

    
    Yx1_pisn = _get_element(pisn_yields, elem1_pisn) / pisn_yields["params"]["mass"]
    Yx2_pisn = _get_element(pisn_yields, elem2_pisn) / pisn_yields["params"]["mass"]
    Yx1_sn = _get_element(sn_yields, elem1_sn) / sn_yields["params"]["mass"]
    Yx2_sn = _get_element(sn_yields, elem2_sn) / sn_yields["params"]["mass"]

    combined_ratio = np.log10(
        (Yx1_pisn + beta*(Yz_pisn/Yz_sn)*Yx1_sn)/(Yx2_pisn + beta*(Yz_pisn/Yz_sn)*Yx2_sn)
    ) - (A1 - A2) - np.log10(atomic_mass[elem1_pisn]/atomic_mass[elem2_pisn])

    # print(f"Salvadori {elem1_pisn}/{elem2_pisn} ratio: {combined_ratio}")
    # print(f"Yields: PISN {elem1_pisn}: {Yx1_pisn}, PISN {elem2_pisn}: {Yx2_pisn}, SN {elem1_sn}: {Yx1_sn}, SN {elem2_sn}: {Yx2_sn}")
    # print(f"Yields: PISN Z: {Yz_pisn}, SN Z: {Yz_sn}, beta: {beta}")

    return combined_ratio

def salvadori_combined_abundratio_WrtH(elem1_pisn: str, elem1_sn: str, pisn_data: list[dict], sn_data: list[dict], f_pisn: float, f_ratio: float) -> float:
    """Compute the X/H abundance ratio for a mixture of PISN and SN yields.
        Refer to Eq. 12 of Salvadori et al. 2019.

    Args:
        pisn_data:
        {
            "label": "M=65",
            "params": {"mass":130, "mass_he": 65},
            "yields": {"H1": 0.1, "He4": 0.2, ...},
        }
        sn_data: (Same as pisn_data)
    Returns:
        The combined abundance ratio [elem1/elem2].
    """

    pisn_data["yields"] = _combine_elements(pisn_data["yields"])
    sn_data["yields"] = _combine_elements(sn_data["yields"])

    pisn_yields = pisn_data
    sn_yields = sn_data

    if elem1_pisn not in pisn_yields["yields"] or "H" not in pisn_yields["yields"]:
        raise ValueError(f"Element {elem1_pisn} or {"H"} not found in PISN yields.")
    if elem1_sn not in sn_yields["yields"] or "H" not in sn_yields["yields"]:
        raise ValueError(f"Element {elem1_sn} or {"H"} not found in SN yields.")

    with open(asplund, "r") as f:
        solar = json.load(f)

    A1 = solar[elem1_pisn]["val"]
    A2 = solar["H"]["val"]

    beta = (1-f_pisn)/f_pisn

    Yz_pisn = sum(
        e for element, e in pisn_yields["yields"].items()
        if element not in ("H", "He")
    ) / pisn_yields["params"]["mass"]

    Yz_sn = sum(
        e for element, e in sn_yields["yields"].items()
        if element not in ("H", "He")
    ) / sn_yields["params"]["mass"]

    Yx1_pisn = _get_element(pisn_yields, elem1_pisn) / pisn_yields["params"]["mass"]
    Yx1_sn = _get_element(sn_yields, elem1_sn) / sn_yields["params"]["mass"]

    combined_ratio = np.log10(
        (f_ratio * (Yx1_pisn + beta*(Yx1_sn*Yz_pisn/Yz_sn)))
    ) - (A1 - A2) - np.log10(atomic_mass[elem1_pisn]/atomic_mass["H"])

    # print(f"Salvadori {elem1_pisn}/H ratio: {combined_ratio}")
    # print(f"Yields: PISN {elem1_pisn}: {Yx1_pisn}, SN {elem1_sn}: {Yx1_sn}")
    # print(f"Yields: PISN Z: {Yz_pisn}, SN Z: {Yz_sn}, beta: {beta}")

    return combined_ratio