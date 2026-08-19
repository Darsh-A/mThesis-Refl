# General mathy stuff

import json
import numpy as np
from scipy.integrate import quad

from params import asplund, atomic_mass


def abundance_ratio(elem1: str, elem2: str, mass1: float, mass2: float) -> float:
    """Compute the abundance ratio of two isotopes given their masses.
    [X/Fe] = log(Mx/MFe)star - log(Mx/MFe)⊙
    how much more/less of element X relative to Fe does this object have, compared to the Sun

    Args:
        elem1: The first element > num
        elem2: The second element > denom
        mass1: The mass of the first element
        mass2: The mass of the second element

    Returns:
        The abundance ratio [X/Fe]
    """
    if elem1 not in atomic_mass or elem2 not in atomic_mass:
        raise ValueError(f"Unknown element: {elem1} or {elem2}")
    if mass1 < 0 or mass2 < 0:
        raise ValueError(f"Masses must be positive: {mass1}, {mass2}")
    
    with open(asplund, "r") as f:
        solar = json.load(f)

    if elem1 not in solar or elem2 not in solar:
        return None
    
    A1 = solar[elem1]["val"]
    A2 = solar[elem2]["val"]

    abundance_ratio = np.log10((mass1 / atomic_mass[elem1]) / (mass2 / atomic_mass[elem2])) - (A1 - A2)

    return abundance_ratio
        

IMF_M_MIN = 0.1
IMF_M_MAX = 100.0


def normalize_larson_imf() -> float:
    integral = quad(
        lambda m: m * (m / 0.35)**(-2.35),
        IMF_M_MIN,
        IMF_M_MAX,
    )[0]

    return 1.0 / integral


A_LARSON = normalize_larson_imf()


def larson_imf(m: float) -> float:
    return A_LARSON * (m / 0.35)**(-2.35)


def raiteri_lifetime(mass: float, Z: float) -> float:
    """
    Stellar lifetime from Raiteri et al. (1996).

    Args:
        mass: Stellar mass in solar masses.
        Z: Metallicity (mass fraction of elements heavier than helium).

    Returns:
        Stellar lifetime in years.
    """

    log_Z = np.log10(Z)
    log_M = np.log10(mass)

    a0 = (
        10.13
        + 0.07547 * log_Z
        - 0.008084 * log_Z**2
    )

    a1 = (
        -4.424
        - 0.7939 * log_Z
        - 0.1187 * log_Z**2
    )

    a2 = (
        1.262
        + 0.3385 * log_Z
        + 0.05417 * log_Z**2
    )

    log_tau = a0 + a1 * log_M + a2 * log_M**2

    return 10**log_tau

from scipy.optimize import brentq


def raiteri_mass_from_lifetime(
    lifetime: float,
    Z: float,
    m_min: float = 0.1,
    m_max: float = 40.0,
) -> float | None:

    tau_min_mass = raiteri_lifetime(m_max, Z)
    tau_max_mass = raiteri_lifetime(m_min, Z)

    if lifetime < tau_min_mass:
        # Even a 100 Msun star has not died yet
        return m_max

    if lifetime > tau_max_mass:
        # Even a 0.1 Msun star has already died
        return m_min

    def equation(mass):
        return (
            np.log10(raiteri_lifetime(mass, Z))
            - np.log10(lifetime)
        )

    return brentq(equation, m_min, m_max)