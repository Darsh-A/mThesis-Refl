import re
import numpy as np
from scipy.stats import loguniform

def _get_element(data: dict, isotope: str) -> float:
    """Get the element mass value from data

    Args:
        data: A dictionary containing yield data.
        isotope: The isotope name (e.g., "C12", "He4").
    NOTE: This is meant to use after yield summation, so it will return the mass of the element, not the isotope.
    it may return wrong mass if summation is not done!!!

    Returns:
        The mass of the element
    """
    # Normalize isotope name to element symbol
    element = _isotope_to_element(isotope)

    # Check if the element is in the data
    if element in data["yields"]:
        return data["yields"][element]

    raise ValueError(f"Element {element} not found in data.")

def _isotope_label(raw: str) -> str:
    """Normalize an isotope name like 'h1' or 'he4' to 'H1', 'He4'."""
    name = raw.strip().lower()
    m = re.match(r'([a-z]+)(\d+)', name)
    if not m:
        return raw
    return m.group(1).capitalize() + m.group(2)


def _isotope_to_element(name:str) -> str:
    """Extract element symbol from an isotope name.

    "c12"      -> "C"
    "$^{12}$C" -> "C"   (LaTeX)
    "$^{}$p"   -> "H"   (proton)
    "$^{}$d"   -> "H"   (deuteron)
    """
    name = name.strip()
    name = name.replace('$', '').replace('{', '').replace('}', '').replace('^', '')
    name = name.lstrip('0123456789').lower()
    m = re.match(r'([a-z]+)', name)
    if not m:
        return name
    symbol = m.group(1)
    if symbol in ('p', 'd'):
        return 'H'
    return symbol.capitalize()


def _combine_elements(yields: dict[str, float]) -> dict[str, float]:
    """Combine isotopes into elements, summing their yields.

    For example, if yields contains "C12" and "C13", the result will contain
    a single entry for "C" with the sum of the two yields.
    """
    combined = {}
    for iso, val in yields.items():
        elem = _isotope_to_element(iso)
        combined[elem] = combined.get(elem, 0.0) + val
    return combined


# Each entry: isotope -> (final stable daughter element, half-life in years)
# Chains (52Fe->52Mn->52Cr, 60Zn->60Cu->60Ni) are collapsed straight to
# their final stable daughter, since every intermediate half-life is
# well under a year -- the compound decay is complete on any timescale
# you'd realistically use here, so there's no need to model the
# intermediate step separately.
_WW95_RADIOACTIVE_HALFLIVES: dict[str, tuple[str, float]] = {
    # Fe-group, sub-year half-lives
    "Ni56": ("Fe", 6.075 / 365.25),
    "Co56": ("Fe", 77.233 / 365.25),
    "Ni57": ("Fe", 35.6 / 24 / 365.25),
    "Co57": ("Fe", 271.74 / 365.25),
    "Co55": ("Mn", 17.53 / 24 / 365.25),
    "Fe55": ("Mn", 2.744),
    # newly added fast chains
    "Ti44": ("Ca", 58.9),
    "Cr48": ("Ti", 21.56 / 24 / 365.25),
    "Fe52": ("Cr", 8.275 / 24 / 365.25),        # via 52Mn, both hops < 1 week
    "Zn60": ("Ni", 2.38 / (60 * 24 * 365.25)),  # via 60Cu, both hops < 1 hr
    "Na22": ("Ne", 2.6018),
    # long-lived: timescale-dependent, worth checking against your model's
    # assumed mixing/enrichment delay rather than assuming full decay
    "Al26": ("Mg", 7.17e5),
    "Fe60": ("Ni", 2.6e6),
}

def _apply_radioactive_decay(
    yields: dict[str, float],
    timescale_yr: float = 1e6,
    decay_map: dict[str, tuple[str, float]] = _WW95_RADIOACTIVE_HALFLIVES
) -> dict[str, float]:
    """Route radioactive isotopes to their stable decay product.

    Must be called on RAW isotope-level yields, before _combine_elements.

    Parameters
    ----------
    yields : dict[str, float]
        Raw isotope-keyed yields (solar masses).
    timescale_yr : float
        Time assumed to elapse between ejection and the point you're
        evaluating abundances (e.g. time until this gas is incorporated
        into the next generation of stars). Governs how much of a
        long-lived isotope (26Al, 60Fe) has actually decayed by then.
        Default of 1 Myr is a fairly standard assumption for minihalo/ISM
        mixing timescales -- short-lived isotopes (Ni56, Ti44, etc.) are
        effectively 100% decayed at this timescale regardless, so this
        parameter only meaningfully changes 26Al/60Fe routing. Set it
        explicitly and document the choice; don't rely on the default
        without checking it matches your enrichment model's assumption.

    Isotopes not in _RADIOACTIVE_HALFLIVES pass through unchanged, under
    their own isotope label. _combine_elements still needs to be called
    afterward to fold everything (decayed + any undecayed remainder)
    into per-element totals.
    """
    result: dict[str, float] = {}
    for iso, val in yields.items():
        label = _isotope_label(iso)
        entry = decay_map.get(label)
        if entry is None:
            result[iso] = result.get(iso, 0.0) + val
            continue

        target_elem, half_life_yr = entry
        decayed_frac = 1.0 - 0.5 ** (timescale_yr / half_life_yr)
        decayed_frac = min(max(decayed_frac, 0.0), 1.0)  # numerical safety

        decayed_val = val * decayed_frac
        remaining_val = val - decayed_val

        result[target_elem] = result.get(target_elem, 0.0) + decayed_val
        if remaining_val > 0.0:
            # keep any undecayed remainder under its own isotope label,
            # rather than silently dropping it
            result[iso] = result.get(iso, 0.0) + remaining_val

    return result


def extract_yield_peaks(pisn_yields, sn_yields, elem, f_pisn=0.9,f_ratio=1e-4, n_samples=500):
    """Extract local density maxima of [elem/Fe] vs [Fe/H] and return, per peak,
    the PISN/SN parameters plus the abundance ratios needed by plot_one_yield."""
    from scipy.ndimage import maximum_filter
    from scipy.stats import gaussian_kde, loguniform
    from params import filter_elements
    from .salvadori_funcs import salvadori_combined_abundratio, salvadori_combined_abundratio_WrtH

    X_Fe, Fe_H, pisn_entries, sn_entries = [], [], [], []

    for _ in range(n_samples):
        pisn_entry = pisn_yields[np.random.randint(len(pisn_yields))]
        sn_entry = sn_yields[np.random.randint(len(sn_yields))]

        ratio = salvadori_combined_abundratio(elem, elem, "Fe", "Fe", pisn_entry, sn_entry, f_pisn=f_pisn)
        ratio_h = salvadori_combined_abundratio_WrtH("Fe", "Fe", pisn_entry, sn_entry, f_pisn=f_pisn, f_ratio=f_ratio)

        # if ratio < -3 or ratio_h < -3:
        #     continue

        X_Fe.append(ratio)
        Fe_H.append(ratio_h)
        pisn_entries.append(pisn_entry)
        sn_entries.append(sn_entry)

    if not X_Fe:
        return []

    X_Fe = np.array(X_Fe)
    Fe_H = np.array(Fe_H)

    kde = gaussian_kde(np.vstack([Fe_H, X_Fe]))
    Xg, Yg = np.mgrid[Fe_H.min():Fe_H.max():200j, X_Fe.min():X_Fe.max():200j]
    Z = np.reshape(kde(np.vstack([Xg.ravel(), Yg.ravel()])).T, Xg.shape)

    maxima = Z == maximum_filter(Z, size=3)
    maxima &= Z > 0.05 * Z.max()

    scale = np.array([Fe_H.std(), X_Fe.std()])
    scale[scale == 0] = 1.0

    idxs = []
    for ix, iy in np.argwhere(maxima):
        d = np.sqrt(((Fe_H - Xg[ix, iy]) / scale[0]) ** 2 + ((X_Fe - Yg[ix, iy]) / scale[1]) ** 2)
        idxs.append(int(np.argmin(d)))
    idxs = list(dict.fromkeys(idxs))

    centers = np.array([[Fe_H[i], X_Fe[i]] for i in idxs])
    assign = np.argmin(np.sqrt(((Fe_H[:, None] - centers[:, 0]) / scale[0]) ** 2
                               + ((X_Fe[:, None] - centers[:, 1]) / scale[1]) ** 2), axis=1)
    idxs = [idxs[k] for k in np.argsort(-np.bincount(assign, minlength=len(idxs)))]

    peaks = []
    for i in idxs:
        pisn_entry, sn_entry = pisn_entries[i], sn_entries[i]
        peaks.append({
            "pisn": {"mass": pisn_entry["params"]["mass"]},
            "sn": {
                "mass": sn_entry["params"]["mass"],
                "lfej": sn_entry["params"]["lfej"],
                "energy": sn_entry["params"]["energy"],
                "elem_value": sn_entry["yields"].get(elem, 0.0),
                "Fe_value": sn_entry["yields"].get("Fe", 0.0),
            },
            "ratios": {
                e: salvadori_combined_abundratio(e, e, "Fe", "Fe", pisn_entry, sn_entry, f_pisn=f_pisn)
                for e in filter_elements
            },
        })
    return peaks
