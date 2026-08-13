# General mathy stuff

import numpy as np
from params import asplund, atomic_mass

import json

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
        

