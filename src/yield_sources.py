"""
Add a new source by writing its extract/select/mass_from_lifetime hooks and
registering it in SOURCES
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .utils import _combine_elements, _apply_radioactive_decay, limongi_lifetime, limongi_mass_from_lifetime, salvadori_select_limongi_feh
from src.formulas import raiteri_mass_from_lifetime
from src.load_data import load_limongi18, load_ww95
from params import ww95_1Z, ww95_01Z, ww95_001Z, ww95_00001Z, Z_SUN

"""
Three main hooks are needed to define a new source:
- extract(entry: dict) -> dict: given a yield entry, return a dictionary of element yields (after any necessary processing, e.g. radioactive decay)
- select(entry: dict, ctx: dict) -> bool: given a yield entry and a context dictionary, return True if the entry should be used for the current calculation. 
    The context dictionary can contain any relevant information, such as metallicity, model, etc.
- mass_from_lifetime(lifetime: float, ctx: dict)

Other params scalars like: name, m_max, load_for_metallicity can be set as needed.
"""

@dataclass
class YieldSource:
    name: str
    m_max: float
    extract: Callable[[dict], dict] = None
    select: Callable[[dict, dict], bool] = None
    # mass_from_lifetime may mutate ctx to record derived selection keys
    # (e.g. feh) it needed to compute the mass, so select() downstream
    # filters on the same grid point.
    mass_from_lifetime: Callable[[float, dict], Optional[float]] = None
    load_for_metallicity: Optional[Callable[[float], list[dict]]] = None


def salvadori_select_ww95_model(Z_rel: float) -> str:
    """Select the nearest WW95 metallicity grid.

    Z_rel = Z_star / Z_sun.
    """
    grids = {
        1.0: ww95_1Z,
        0.1: ww95_01Z,
        0.01: ww95_001Z,
        1e-4: ww95_00001Z,
    }
    return grids[min(grids, key=lambda z: abs(np.log10(Z_rel) - np.log10(z)))]


def _ww95_extract(entry: dict) -> dict:
    yields = _combine_elements(_apply_radioactive_decay(entry["yields"]))
    if "Fe" in yields:
        yields["Fe"] *= 0.5
    return yields


def _ww95_select(entry: dict, ctx: dict) -> bool:
    return entry["params"]["model"] == ctx.get("model", "A")


def _ww95_mass_from_lifetime(lifetime: float, ctx: dict) -> Optional[float]:
    return raiteri_mass_from_lifetime(lifetime=lifetime, Z=ctx["Z_star"])


def _ww95_load_for_metallicity(Z_rel: float) -> list[dict]:
    return load_ww95(salvadori_select_ww95_model(Z_rel))


WW95 = YieldSource(
    name="WW95",
    m_max=40.0,
    extract=_ww95_extract,
    select=_ww95_select,
    mass_from_lifetime=_ww95_mass_from_lifetime,
    load_for_metallicity=_ww95_load_for_metallicity,
)


def _limongi_extract(entry: dict) -> dict:
    return entry["yields"]


def _limongi_select(entry: dict, ctx: dict) -> bool:
    feh = ctx.get("feh")
    return feh is None or entry["params"]["feh"] == feh


def _limongi_mass_from_lifetime(lifetime: float, ctx: dict) -> Optional[float]:
    feh = salvadori_select_limongi_feh(ctx["Z_star"] / Z_SUN)
    ctx["feh"] = feh
    return limongi_mass_from_lifetime(lifetime, feh=feh, velocity=0)

def _limongi_load_for_metallicity(Z_star: float) -> list[dict]:
    """Load LC18 yields, filtered to the [Fe/H] grid point nearest Z_star
    """
    feh = salvadori_select_limongi_feh(Z_star / Z_SUN)
    all_entries = load_limongi18()
    return [e for e in all_entries if e["params"]["feh"] == feh and e["params"]["velocity"] == 0]

LIMONGI18 = YieldSource(
    name="Limongi18",
    m_max=120.0,
    extract=_limongi_extract,
    select=_limongi_select,
    mass_from_lifetime=_limongi_mass_from_lifetime,
    load_for_metallicity=_limongi_load_for_metallicity
)


def _nomoto_extract(entry: dict) -> dict:
    """Nomoto13 yields are already element-level (no decay routing needed, per HW02-style post-decay convention)."""
    return entry["yields"]

def _nomoto_select(entry: dict, ctx: dict) -> bool:
    p = entry["params"]
    return p["energy"] == 1.0

NOMOTO13 = YieldSource(
    name="Nomoto13",
    m_max=40.0, 
    extract=_nomoto_extract,
    select=_nomoto_select,
    mass_from_lifetime=None,
    load_for_metallicity=None,
)

SOURCES = {"WW95": WW95, "Limongi18": LIMONGI18, "Nomoto13": NOMOTO13}


def get_source(name: str) -> YieldSource:
    if name not in SOURCES:
        raise ValueError(f"Unrecognized sn_input: {name!r}")
    return SOURCES[name]