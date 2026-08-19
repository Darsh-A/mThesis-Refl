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

# Some raw yield tables (WW95 in particular) report freshly-synthesized,
# PRE-DECAY isotopic yields, including short-lived radioactive species.
# On any timescale relevant to this project (>= Myr), these have long
# since decayed to their stable daughter. Chief among them: 56Ni (t1/2=6.1d)
# -> 56Co (t1/2=77d) -> 56Fe, which is the dominant iron-production channel
# in core-collapse SNe. Filing 56Ni under elemental "Ni" (as a naive
# isotope->element mapping would) silently strips the majority of a CCSN's
# true iron yield out of "Fe" and into "Ni".
#
# Only decay chains with total half-lives << 1 Myr (our shortest tpopII of
# interest) are included; e.g. 60Fe (t1/2=2.6 Myr, -> 60Co -> 60Ni) is left
# alone since its decay is not fast on our timescales of interest.
_RADIOACTIVE_DECAY_MAP = {
    "Ni56": "Fe",  # 56Ni -> 56Co -> 56Fe
    "Co56": "Fe",
    "Ni57": "Fe",  # 57Ni -> 57Co -> 57Fe
    "Co57": "Fe",
    "Co55": "Mn",  # 55Co -> 55Fe -> 55Mn
    "Fe55": "Mn",
}


def _apply_radioactive_decay(yields: dict[str, float]) -> dict[str, float]:
    """Route short-lived isotopes to their stable decay product.

    Must be called on RAW isotope-level yields, before _combine_elements.
    Isotopes not in _RADIOACTIVE_DECAY_MAP pass through unchanged (under
    their own isotope label); _combine_elements still needs to be called
    afterward to fold everything into per-element totals.
    """
    result = {}
    for iso, val in yields.items():
        target_elem = _RADIOACTIVE_DECAY_MAP.get(_isotope_label(iso))
        if target_elem is not None:
            result[target_elem] = result.get(target_elem, 0.0) + val
        else:
            result[iso] = result.get(iso, 0.0) + val
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
