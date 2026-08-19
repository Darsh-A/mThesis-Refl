# Contains formula to replicate Salvadori et al. 2019

import copy
import json

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from src.formulas import larson_imf, raiteri_mass_from_lifetime, raiteri_lifetime
from src.load_data import load_ww95

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

    abundance_ratio = np.log10((yields[elem1]) / (yields[elem2])) - (A1 - A2) - np.log10(atomic_mass[elem1]/atomic_mass[elem2])

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

def salvadori_combined_abundratio(elem1_pisn: str, elem1_sn: str, elem2_pisn: str, elem2_sn: str, pisn_data: list[dict],f_pisn: float, f_ratio: float, tpop2: float) -> float:
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
    pisn_yields = pisn_data

    # if elem1_pisn not in pisn_yields["yields"] or elem2_pisn not in pisn_yields["yields"]:
    #     raise ValueError(f"Element {elem1_pisn} or {elem2_pisn} not found in PISN yields.")
    # if elem1_sn not in sn_yields["yields"] or elem2_sn not in sn_yields["yields"]:
    #     raise ValueError(f"Element {elem1_sn} or {elem2_sn} not found in SN yields.")
    
    with open(asplund, "r") as f:
        solar = json.load(f)

    A1 = solar[elem1_pisn]["val"]
    A2 = solar[elem2_pisn]["val"]

    beta = (1-f_pisn)/f_pisn

    Yx1_pisn = _get_element(pisn_yields, elem1_pisn)
    Yx2_pisn = _get_element(pisn_yields, elem2_pisn)

    Yz_pisn = sum(
        e for element, e in pisn_yields["yields"].items()
        if element not in ("H", "He")
    )

    Z_star = f_ratio * Yz_pisn

    ww_model = salvadori_select_ww95_model(Z_star)
    sn_data = load_ww95(ww_model)

    m_popII = raiteri_mass_from_lifetime(lifetime=tpop2,Z=Z_star)
    if m_popII is None:
        return np.nan
    
    if m_popII >= 100.0:
        sn_term_1 = 0.0
        sn_term_2 = 0.0
    else:
        Yx1_sn = salvadori_Y_X_II(data=sn_data, elem=elem1_sn, m_popII=m_popII, model="A", m_max=100.0)
        Yx2_sn = salvadori_Y_X_II(data=sn_data, elem=elem2_sn, m_popII=m_popII, model="A", m_max=100.0)
        Yz_sn  = salvadori_Y_Z_II(data=sn_data, m_popII=m_popII, model="A", m_max=100.0)
        if Yz_sn <= 0:
            sn_term_1 = 0.0
            sn_term_2 = 0.0
        else:
            sn_term_1 = (Yz_pisn/Yz_sn) * Yx1_sn
            sn_term_2 = (Yz_pisn/Yz_sn) * Yx2_sn

    combined_ratio = np.log10(
        (Yx1_pisn + beta*sn_term_1) / (Yx2_pisn + beta*sn_term_2)
    ) - (A1 - A2) - np.log10(atomic_mass[elem1_pisn]/atomic_mass[elem2_pisn])

    return combined_ratio

def salvadori_combined_abundratio_WrtH(elem1_pisn: str, elem1_sn: str, pisn_data: list[dict], f_pisn: float, f_ratio: float, tpop2: float) -> float:
    """Compute the X/H abundance ratio for a mixture of PISN and SN yields.
        Refer to Eq. 12 of Salvadori et al. 2019.

    Args:
        pisn_data:
        {
            "label": "M=65",
            "params": {"mass":130, "mass_he": 65},
            "yields": {"H1": 0.1, "He4": 0.2, ...},
        }
        sn_data: (Same as pisn_data) #NOTE: NOT salvadori processed data, but raw yields
    Returns:
        The combined abundance ratio [elem1/elem2].
    """

    pisn_data["yields"] = _combine_elements(pisn_data["yields"])
    pisn_yields = pisn_data
    
    with open(asplund, "r") as f:
        solar = json.load(f)

    A1 = solar[elem1_pisn]["val"]
    A2 = solar["H"]["val"]

    beta = (1-f_pisn)/f_pisn

    Yx1_pisn = _get_element(pisn_yields, elem1_pisn) #/ pisn_yields["params"]["mass"]
    Yz_pisn = sum(
        e for element, e in pisn_yields["yields"].items()
        if element not in ("H", "He")
    ) #/ pisn_yields["params"]["mass"]

    Z_star = f_ratio * Yz_pisn

    ww_model = salvadori_select_ww95_model(Z_star)
    sn_data = load_ww95(ww_model)

    m_popII = raiteri_mass_from_lifetime(
        lifetime=tpop2,
        Z=Z_star,
    )

    # print(f"m_popII: {m_popII}, tpop2: {tpop2}, Z_star: {Z_star}")

    if m_popII is None:
        return np.nan
    
    if m_popII >= 100.0:
        sn_term = 0.0
    else:
        Yx1_sn = salvadori_Y_X_II(data=sn_data, elem=elem1_sn, m_popII=m_popII, model="A", m_max=100.0)
        Yz_sn  = salvadori_Y_Z_II(data=sn_data, m_popII=m_popII, model="A", m_max=100.0)
        sn_term = 0.0 if Yz_sn <= 0 else (Yx1_sn * Yz_pisn / Yz_sn)

    combined_ratio = np.log10(
        f_ratio * (Yx1_pisn + beta * sn_term)
    ) - (A1 - A2) - np.log10(atomic_mass[elem1_pisn]/atomic_mass["H"])

    return combined_ratio

def salvadori_Y_X_II(data: list[dict], elem: str, m_popII: float, model: str = "A", m_max: float = 100.0):
    """Compute the time-dependent SNII yield from Salvadori et al. (2019).

    Y_X^II(t) = integral[m_popII(t), 100 Msun]
                m_X^II(m) * Phi(m) dm

    Args:
        data: Raw WW95 yield entries from load_ww95() (or any other raw yield)
        elem: Element to calculate, e.g. "Fe", "O".
        m_popII: Lower integration limit in Msun.
        m_max: Upper integration limit in Msun.

    Returns:
        Y_X^II.
    """

    masses = []
    yields = []

    for entry in data:
        if entry['params']['model'] != model:
            continue

        mass = entry["params"]["mass"]
        element_yields = _combine_elements(entry["yields"])

        if elem == "Fe":
            element_yields["Fe"] *= 0.5

        if elem not in element_yields:
            continue

        masses.append(mass)
        yields.append(element_yields[elem])

    masses = np.asarray(masses, dtype=float)
    yields = np.asarray(yields, dtype=float)

    # Sort by progenitor mass
    order = np.argsort(masses)
    masses = masses[order]
    yields = yields[order]

    m_X_II = interp1d(
        masses,
        yields,
        kind="linear",
        bounds_error=False,
        fill_value=(yields[0], 0),
    )

    return quad(
        lambda m: float(m_X_II(m)) * larson_imf(m),
        m_popII,
        100.0,
    )[0]

def salvadori_Y_Z_II(data: list[dict], m_popII: float, model: str = "A", m_max: float = 100.0) -> float:
    """Compute the time-dependent, IMF-integrated total metal yield Y_Z^II(t)
    from Salvadori et al. (2019), consistent with salvadori_Y_X_II.

        Y_Z^II(t) = integral[m_popII(t), 100 Msun] ( sum_X m_X^II(m) ) * Phi(m) dm

    Args:
        data: Raw WW95 (or equivalent) yield entries.
        m_popII: Lower integration limit in Msun (from raiteri_mass_from_lifetime).
        model: Which model tag to filter on (e.g. "A").
        m_max: Upper integration limit in Msun.

    Returns:
        Y_Z^II(t).
    """
    masses = []
    total_metal_mass = []

    for entry in data:
        if entry["params"]["model"] != model:
            continue

        mass = entry["params"]["mass"]
        element_yields = _combine_elements(entry["yields"])

        element_yields["Fe"] = element_yields.get("Fe", 0.0) * 0.5

        metal_sum = sum(
            v for el, v in element_yields.items()
            if el not in ("H", "He")
        )

        masses.append(mass)
        total_metal_mass.append(metal_sum)

    if not masses:
        raise ValueError(f"No entries found for model='{model}' in salvadori_Y_Z_II.")

    masses = np.asarray(masses, dtype=float)
    total_metal_mass = np.asarray(total_metal_mass, dtype=float)

    order = np.argsort(masses)
    masses = masses[order]
    total_metal_mass = total_metal_mass[order]

    m_Z_II = interp1d(
        masses,
        total_metal_mass,
        kind="linear",
        bounds_error=False,
        fill_value=(total_metal_mass[0], 0) ,
    )

    return quad(
        lambda m: float(m_Z_II(m)) * larson_imf(m),
        m_popII,
        100.0,
    )[0]


from params import ww95_1Z, ww95_01Z, ww95_001Z, ww95_00001Z


def salvadori_select_ww95_model(Z_star: float) -> str:
    """Select the nearest available WW95 metallicity grid.

    Z_star is expressed in units of Z_sun.
    """
    if Z_star < 0.001:
        return ww95_00001Z
    elif Z_star < 0.055:
        return ww95_001Z
    elif Z_star < 0.316:
        return ww95_01Z
    else:
        return ww95_1Z