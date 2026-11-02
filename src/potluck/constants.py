"""Physical constants and the common element set."""

from __future__ import annotations

import json

from params import Z_SUN, asplund, atomic_mass

ELEMENTS: tuple[str, ...] = (
    "Al", "Ar", "B", "Be", "C", "Ca", "Cl", "Co", "Cr", "Cu",
    "F", "Fe", "Ga", "Ge", "H", "He", "K", "Li", "Mg", "Mn",
    "N", "Na", "Ne", "Ni", "O", "P", "S", "Sc", "Si", "Ti",
    "V", "Zn",
)
METALS: tuple[str, ...] = tuple(e for e in ELEMENTS if e not in ("H", "He"))
ELEMENT_INDEX: dict[str, int] = {e: i for i, e in enumerate(ELEMENTS)}

REFERENCE: str = "Fe"
FE_INDEPENDENT_RATIOS: tuple[tuple[str, str], ...] = (
    ("Na", "Mg"),
    ("Co", "Ni"),
    ("Mn", "Cr"),
    ("Zn", "Mg"),
)
# Every element the three yield tables (HW2002 PISN, Pop III, Limongi18 Pop II)
# synthesise above ``min_element_fraction``. H and He are the reference
# reservoir, B and Be never clear the floor, and Li is the label species with no
# Li+ catalogue column, so none of the four can be a feature.
DEFAULT_FEATURE_ELEMENTS: tuple[str, ...] = (
    "Al", "Ar", "C", "Ca", "Cl", "Co", "Cr", "Cu", "F", "Fe",
    "Ga", "Ge", "K", "Mg", "Mn", "N", "Na", "Ne", "Ni", "O",
    "P", "S", "Sc", "Si", "Ti", "V", "Zn",
)
LI_FEATURE_ELEMENTS: tuple[str, ...] = (
    "C", "Na", "Mg", "Si", "Ca", "Ti", "Cr", "Mn", "Fe", "Co", "Ni", "Zn",
)

X_H_SUN: float = 0.7381

with open(asplund) as _f:
    _solar = json.load(_f)

SOLAR_ABUNDANCE: dict[str, float] = {e: float(_solar[e]["val"]) for e in ELEMENTS}
SOLAR_MASS_FRACTION: dict[str, float] = {
    e: 10.0 ** (SOLAR_ABUNDANCE[e] - 12.0) * atomic_mass[e] / atomic_mass["H"] * X_H_SUN
    for e in ELEMENTS
}
Z_SUN_COMPUTED: float = sum(SOLAR_MASS_FRACTION[e] for e in METALS)

for _e in ELEMENTS:
    if _e not in atomic_mass or _e not in _solar:
        raise KeyError(f"Missing atomic mass or solar abundance for {_e}")
