"""Observed-element frequency vector for a stellar abundance catalogue.

Walks a loaded observed catalogue (Li+ by default) and, for every element,
returns the fraction of stars in which that element was observed.  The
catalogue-level frequency of an element is the fraction of stars in which it
appears.

The entry point :func:`getObsMask` is importable anywhere:

    from src.getObsMask import getObsMask

    freq_vector = getObsMask("Li")   # {"C": 0.91, "Fe": 1.0, ...}
"""

from __future__ import annotations

from .load_data import load_obs_li
from .utils import _ion_to_element

# Catalogue name -> no-argument loader callable.  Add new catalogues here to
# expose them via getObsMask; the loader must return the same per-star shape
# as ``load_obs_li`` (a list of dicts with "label", "params", "abundances").
CATALOGUES = {
    "li": load_obs_li,
    "li+": load_obs_li,
}

DEFAULT_CATALOGUE = "li"


def load_catalogue(name: str = DEFAULT_CATALOGUE) -> list[dict]:
    """Load an observed catalogue by name (case-insensitive; see CATALOGUES)."""
    key = name.lower()
    if key not in CATALOGUES:
        raise ValueError(f"Unknown catalogue {name!r}; choose from {sorted(CATALOGUES)}")
    return CATALOGUES[key]()


def observed_elements(star: dict, include_upper_limits: bool = True) -> set[str]:
    """Set of elements with a measured abundance for a single star.

    A star loaded by ``load_obs_li`` has an ``abundances`` dict keyed by ion
    label.  An element is observed if any of its ions has a non-blank
    ``logeps`` value.  Upper-limit rows (``l_logeps == '<'``) carry a value
    but are non-detections; pass ``include_upper_limits=False`` to exclude
    them from the observed set.
    """
    elements: set[str] = set()
    for ion, rec in star.get("abundances", {}).items():
        if rec.get("logeps") is None:
            continue
        if not include_upper_limits and rec.get("l_logeps") == "<":
            continue
        elements.add(_ion_to_element(ion))
    return elements


def get_freq(stars: list[dict], include_upper_limits: bool = True) -> dict[str, float]:
    """Fraction of stars (0.0-1.0) in which each element is observed.

    Parameters
    ----------
    stars:
        A list of per-star dicts as returned by ``load_obs_li`` (or any
        loader in ``CATALOGUES``).
    include_upper_limits:
        Count upper-limit abundance rows as observed (default True).

    Returns
    -------
    dict[str, float]
        Maps element symbol -> observed fraction across the whole catalogue.
    """
    n = len(stars)
    if n == 0:
        return {}

    counts: dict[str, int] = {}
    for star in stars:
        for el in observed_elements(star, include_upper_limits):
            counts[el] = counts.get(el, 0) + 1

    return {el: count / n for el, count in counts.items()}


def getObsMask(
    catalogue: str = DEFAULT_CATALOGUE,
    include_upper_limits: bool = True,
) -> dict[str, float]:
    """Load a catalogue by name and return its element frequency vector.

    Convenience wrapper combining :func:`load_catalogue` and :func:`get_freq`:

        from src.getObsMask import getObsMask

        freq_vector = getObsMask("Li")   # {"C": 0.91, "Fe": 1.0, ...}
    """
    return get_freq(load_catalogue(catalogue), include_upper_limits)
