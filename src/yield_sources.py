"""
Add a new source by writing its extract/select/mass_from_lifetime hooks and
registering it in SOURCES
"""

import functools
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .utils import _combine_elements, _apply_radioactive_decay, limongi_feh_continuous, limongi_mass_from_lifetime, salvadori_select_limongi_feh
from src.formulas import raiteri_mass_from_lifetime
from src.load_data import load_limongi18, load_nomoto13, load_ww95
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
    grid_label: Optional[Callable[[float], object]] = None


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

def _ww95_grid_key(Z_rel: float) -> float:
    grids = (1.0, 0.1, 0.01, 1e-4)
    return min(grids, key=lambda z: abs(np.log10(Z_rel) - np.log10(z)))

def _ww95_grid_label(Z_star: float) -> float:
    return _ww95_grid_key(Z_star)


def _ww95_extract(entry: dict) -> dict:
    yields = _combine_elements(_apply_radioactive_decay(entry["yields"]))
    if "Fe" in yields:
        yields["Fe"] *= 0.5
    return yields


def _ww95_select(entry: dict, ctx: dict) -> bool:
    return entry["params"]["model"] == ctx.get("model", "A")


def _ww95_mass_from_lifetime(lifetime: float, ctx: dict) -> Optional[float]:
    return raiteri_mass_from_lifetime(lifetime=lifetime, Z=ctx["Z_star"])


@functools.lru_cache(maxsize=None)
def _ww95_load_for_metallicity(Z: float) -> list[dict]:
    """Load WW95 yields at absolute metallicity Z (mass fraction).

    Keeps the same contract as the Limongi18/Nomoto13 load_for_metallicity
    hooks (absolute Z); the WW95 grid selector itself works in Z/Z_sun.
    """
    return load_ww95(salvadori_select_ww95_model(Z / Z_SUN))


WW95 = YieldSource(
    name="WW95",
    m_max=40.0,
    extract=_ww95_extract,
    select=_ww95_select,
    mass_from_lifetime=_ww95_mass_from_lifetime,
    load_for_metallicity=_ww95_load_for_metallicity,
    grid_label=_ww95_grid_label,
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


@functools.lru_cache(maxsize=None)
def _limongi_load_for_metallicity_fe(feh: int) -> list[dict]:
    all_entries = load_limongi18()
    return [e for e in all_entries if e["params"]["feh"] == feh and e["params"]["velocity"] == 0]


@functools.lru_cache(maxsize=None)
def _limongi_entries_by_feh(feh: int) -> dict[int, dict]:
    return {e["params"]["mass"]: e for e in _limongi_load_for_metallicity_fe(feh)}


@functools.lru_cache(maxsize=None)
def _limongi_load_for_metallicity_cont(feh: float) -> list[dict]:
    """LC18 yields linearly interpolated to a continuous [Fe/H] (velocity=0)."""
    feh = float(np.clip(feh, -3.0, 0.0))
    lo = max(f for f in (-3, -2, -1, 0) if f <= feh)
    hi = min(f for f in (-3, -2, -1, 0) if f >= feh)
    if lo == hi:
        return _limongi_load_for_metallicity_fe(lo)

    w = (feh - lo) / (hi - lo)
    lo_by_mass = _limongi_entries_by_feh(lo)
    hi_by_mass = _limongi_entries_by_feh(hi)

    out = []
    for mass in sorted(lo_by_mass):
        lo_e, hi_e = lo_by_mass[mass], hi_by_mass[mass]
        elems = set(lo_e["yields"]) | set(hi_e["yields"])
        yields = {
            el: (1.0 - w) * lo_e["yields"].get(el, 0.0) + w * hi_e["yields"].get(el, 0.0)
            for el in elems
        }
        out.append({
            "label": f"M={mass} v=0 [Fe/H]={feh:.4f}",
            "params": {"mass": mass, "velocity": 0, "feh": feh},
            "yields": yields,
        })
    return out


def _limongi_load_for_metallicity(Z_star: float) -> list[dict]:
    """Load LC18 yields interpolated to the continuous [Fe/H] of Z_star."""
    return _limongi_load_for_metallicity_cont(limongi_feh_continuous(Z_star / Z_SUN))


def _limongi_grid_label(Z_star: float) -> float:
    return limongi_feh_continuous(Z_star / Z_SUN)

LIMONGI18 = YieldSource(
    name="Limongi18",
    m_max=120.0,
    extract=_limongi_extract,
    select=_limongi_select,
    mass_from_lifetime=_limongi_mass_from_lifetime,
    load_for_metallicity=_limongi_load_for_metallicity,
    grid_label=_limongi_grid_label
)


# Nomoto13 CCSN metallicity grid (mass fraction), energy=1 models only.
_NOMOTO_Z_GRID = (0.001, 0.004, 0.008, 0.02, 0.05)


def _nomoto_extract(entry: dict) -> dict:
    """Nomoto13 yields are per-isotope and already post-decay, so just fold
    isotopes into elements (no decay routing needed)."""
    return _combine_elements(entry["yields"])

def _nomoto_select(entry: dict, ctx: dict) -> bool:
    p = entry["params"]
    return p["energy"] == 1.0

def _nomoto_mass_from_lifetime(lifetime: float, ctx: dict) -> Optional[float]:
    return raiteri_mass_from_lifetime(lifetime=lifetime, Z=ctx["Z_star"])


@functools.lru_cache(maxsize=None)
def _nomoto_entries_by_z(z: float) -> list[dict]:
    return [e for e in load_nomoto13() if e["params"]["Z"] == z and e["params"]["energy"] == 1.0]


@functools.lru_cache(maxsize=None)
def _nomoto_load_for_metallicity(Z_star: float) -> list[dict]:
    """Nomoto13 (energy=1) yields linearly interpolated to a continuous Z."""
    log_grid = np.log10(_NOMOTO_Z_GRID)
    logZ = float(np.clip(np.log10(Z_star), log_grid[0], log_grid[-1]))
    idx = int(np.searchsorted(log_grid, logZ))
    if idx == 0:
        return _nomoto_entries_by_z(_NOMOTO_Z_GRID[0])
    if idx >= len(log_grid):
        return _nomoto_entries_by_z(_NOMOTO_Z_GRID[-1])

    lo_z, hi_z = _NOMOTO_Z_GRID[idx - 1], _NOMOTO_Z_GRID[idx]
    w = (logZ - log_grid[idx - 1]) / (log_grid[idx] - log_grid[idx - 1])
    lo_by_mass = {e["params"]["mass"]: e for e in _nomoto_entries_by_z(lo_z)}
    hi_by_mass = {e["params"]["mass"]: e for e in _nomoto_entries_by_z(hi_z)}

    out = []
    for mass in sorted(set(lo_by_mass) & set(hi_by_mass)):
        lo_e, hi_e = lo_by_mass[mass], hi_by_mass[mass]
        elems = set(lo_e["yields"]) | set(hi_e["yields"])
        yields = {
            el: (1.0 - w) * lo_e["yields"].get(el, 0.0) + w * hi_e["yields"].get(el, 0.0)
            for el in elems
        }
        out.append({
            "label": f"Z={Z_star:g} M={mass:g} E=1",
            "params": {
                "Z": float(Z_star),
                "mass": mass,
                "energy": 1.0,
                "Mrem": (1.0 - w) * lo_e["params"].get("Mrem", 0.0) + w * hi_e["params"].get("Mrem", 0.0),
            },
            "yields": yields,
        })
    return out


def _nomoto_grid_label(Z_star: float) -> float:
    return float(np.clip(np.log10(Z_star), np.log10(_NOMOTO_Z_GRID[0]), np.log10(_NOMOTO_Z_GRID[-1])))


NOMOTO13 = YieldSource(
    name="Nomoto13",
    m_max=40.0, 
    extract=_nomoto_extract,
    select=_nomoto_select,
    mass_from_lifetime=_nomoto_mass_from_lifetime,
    load_for_metallicity=_nomoto_load_for_metallicity,
    grid_label=_nomoto_grid_label,
)

SOURCES = {"WW95": WW95, "Limongi18": LIMONGI18, "Nomoto13": NOMOTO13}


def get_source(name: str) -> YieldSource:
    if name not in SOURCES:
        raise ValueError(f"Unrecognized sn_input: {name!r}")
    return SOURCES[name]