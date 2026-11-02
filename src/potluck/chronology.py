"""Stellar lifetimes and the two-clock enrichment gates."""

from __future__ import annotations

import numpy as np

from src.formulas import raiteri_lifetime, raiteri_mass_from_lifetime

# Schaerer (2002), A&A 382, 28, Table 4 (no mass loss).
_POPIII_MASS = np.array(
    [500.0, 400.0, 300.0, 200.0, 120.0, 80.0, 60.0, 40.0, 25.0, 15.0, 9.0, 5.0]
)
_POPIII_LIFETIME_MYR = np.array(
    [1.899, 1.974, 2.047, 2.204, 2.521, 3.012, 3.464, 3.864, 6.459, 10.40, 20.22, 61.90]
)

DEFAULT_RAITERI_Z_FLOOR = 7.0e-5


def popiii_lifetime(mass: np.ndarray | float) -> np.ndarray | float:
    """Zero-metallicity lifetime in Myr from the Schaerer (2002) table."""
    query = np.asarray(mass, dtype=float)
    log_tau = np.interp(
        np.log10(np.clip(query, _POPIII_MASS.min(), _POPIII_MASS.max())),
        np.log10(_POPIII_MASS[::-1]),
        np.log10(_POPIII_LIFETIME_MYR[::-1]),
    )
    result = 10.0**log_tau
    return float(result) if np.ndim(mass) == 0 else result


def popii_lifetime(
    mass: np.ndarray | float,
    metallicity: np.ndarray | float,
    z_floor: float = DEFAULT_RAITERI_Z_FLOOR,
) -> np.ndarray | float:
    """Pop II lifetime in Myr from Raiteri et al. (1996)."""
    z = np.maximum(np.asarray(metallicity, dtype=float), z_floor)
    result = raiteri_lifetime(mass, z) / 1.0e6
    if np.ndim(mass) == 0 and np.ndim(metallicity) == 0:
        return float(result)
    return result


def popii_turnoff(
    t_myr: float,
    metallicity: float,
    m_max: float,
    z_floor: float = DEFAULT_RAITERI_Z_FLOOR,
) -> float:
    """Lowest Pop II mass that has exploded by ``t_myr`` after formation."""
    z = max(float(metallicity), z_floor)
    mass = raiteri_mass_from_lifetime(
        lifetime=float(t_myr) * 1.0e6,
        Z=z,
        m_min=0.1,
        m_max=float(m_max),
    )
    return float(m_max if mass is None else mass)


def popiii_pre_enriches(
    mass: np.ndarray | float,
    delta_myr: np.ndarray | float,
) -> np.ndarray | bool:
    """Whether a Pop III event exploded before Pop II formed."""
    return popiii_lifetime(mass) <= np.asarray(delta_myr)


def popiii_reaches_final(
    mass: np.ndarray | float,
    delta_myr: np.ndarray | float,
    t_myr: np.ndarray | float,
) -> np.ndarray | bool:
    """Whether a Pop III event exploded before the observed star formed."""
    return popiii_lifetime(mass) <= (np.asarray(delta_myr) + np.asarray(t_myr))
