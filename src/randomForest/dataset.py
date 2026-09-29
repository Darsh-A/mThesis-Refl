"""Generate training datasets for the f_pisn regression. 

Each sample picks a random PISN entry (via the continuous-mass interpolator)
and a fixed SN source (IMF-integrated; single_sn=False, auto_sn=False), then
computes the abundance ratios used as model features.

Features:
    - [X/Y] for every unordered pair (X, Y) of elements in filter_elements.
      This is C(n, 2) ratios where n = len(filter_elements); H/He are excluded.

Target:
    - f_pisn (drawn ~ [0.1, 1])

Diagnostics (not features, stored for later inspection):
    - f_ratio  (~ loguniform 1e-4 .. 1e-1)
    - tpop2    (~ loguniform 3.2e6 .. 17.4e6)
    - pisn_mass (drawn ~ [150, 270]) -- optionally added as a feature later
      to test whether the ratios alone encode PISN mass.

Per-source routing:
    - Limongi18 / Nomoto13: element-level yields (pass entries directly).
    - WW95: isotope-level raw yields (extract applies decay + combine).
"""

import os
import time
from itertools import combinations

import numpy as np
from scipy.stats import loguniform

from params import filter_elements, Z_SUN
from src.formulas import raiteri_mass_from_lifetime
from src.load_data import load_hw2002, load_takahashi, load_ww95, load_limongi18, load_nomoto13
from src.salvadori_funcs import (
    salvadori_yields,
    salvadori_combined_abundratio,
    salvadori_sn_only_abundance_ratio,
)
from src.utils import build_pisn_interpolator
from src.yield_sources import get_source

# --- Parameter priors ---
F_PISN_RANGE = (0.01, 1.0)
F_RATIO_RANGE = (1e-4, 1e-1)
TPOP2_RANGE = (3.2e6, 17.4e6)
PISN_MASS_RANGE = (150.0, 270.0)
FEH_RANGE = (-3.0, 0.0)

# Clamp floor for abundance ratios
RATIO_FLOOR = -5.0

# Map source name -> element-level vs isotope-level handling of its yield
# entries. Element-level sources are passed to salvadori_Y_X_II directly;
# isotope-level sources (WW95) rely on source.extract to apply radioactive
# decay + combine isotopes.
ELEMENT_LEVEL_SOURCES = {"Limongi18", "Nomoto13"}
ISOTOPE_LEVEL_SOURCES = {"WW95"}


def _load_pisn(source: str) -> tuple[list[dict], object]:
    """Load PISN entries and return (entries, interpolator).

    The interpolator maps a total mass -> a yield entry dict (see
    build_pisn_interpolator).
    """
    if source == "HW2002":
        entries = load_hw2002()
        entries = entries[1:]
    elif source == "Takahashi":
        entries = load_takahashi()["NR"]
    else:
        raise ValueError(f"Unrecognized pisn_source: {source!r}")

    sal_pisn = salvadori_yields(entries)
    interp = build_pisn_interpolator(sal_pisn)
    return entries, interp


def _load_sn(source: str) -> tuple[list[dict], object]:
    """Load SN entries and the matching YieldSource instance."""
    get_source(source)
    if source == "Limongi18":
        entries = load_limongi18()
        entries = [e for e in entries if e["params"]["velocity"] == 0]
    elif source == "WW95":
        entries = load_ww95()
    elif source == "Nomoto13":
        entries = load_nomoto13()
    else:
        raise ValueError(f"Unrecognized sn_source: {source!r}")

    return entries, get_source(source)


def _abundance_elements() -> list[str]:
    """Elements used to build pairwise abundance-ratio features.
    """
    # return [e for e in filter_elements if e not in ("H", "He")]
    return ["C", "O", "Na", "Mg", "Al", "Si", "Ca", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"]


def _feature_pairs() -> list[tuple[str, str]]:
    """All unordered (X, Y) element pairs -> [X/Y] features."""
    return list(combinations(_abundance_elements(), 2))


def _feature_names() -> list[str]:
    """Feature column names: [X/Y] for each element pair."""
    return [f"[{x}/{y}]" for x, y in _feature_pairs()]


def _print_fpisn_summary(y: np.ndarray, pisn_source: str, sn_source: str) -> None:
    """Print the f_pisn bin counts and yield sources after generation."""
    edges = np.linspace(0.0, 1.0, 11)  # 0.0, 0.1, ..., 1.0
    counts, _ = np.histogram(y, bins=edges)

    print("Generated dataset summary:")
    print(f"  PISN source : {pisn_source}")
    print(f"  SN source   : {sn_source}")
    print(f"  total       : {len(y)}")
    print("  f_pisn bins :")
    for lo, hi, count in zip(edges[:-1], edges[1:], counts):
        if lo == 0.0:
            label = "f_pisn = 0"
        else:
            label = f"{lo:.1f} <= f_pisn < {hi:.1f}"
        print(f"    {label:<24} {int(count)}")


# Map Li+22 ion keys -> element symbols used by the mock feature pairs.
_LI_ION_TO_ELEMENT = {
    "C": "C", "NaI": "Na", "MgI": "Mg", "SiI": "Si", "CaI": "Ca",
    "CrI": "Cr", "MnI": "Mn", "FeI": "Fe", "CoI": "Co", "NiI": "Ni",
    "ZnI": "Zn",
}


def li_plus_element_sigmas() -> dict[str, float]:
    """Median Li+22 sigma_Total (dex) per element, used as realistic noise.

    Lazily imports ``load_obs_li`` to avoid import-order coupling during module
    load. Upper-limit stars (l_XFe == "<") and missing/non-finite sigma values
    are skipped. Returns {element_symbol: median_sigma_Total}.
    """
    from collections import defaultdict
    from src.load_data import load_obs_li

    per_element: dict[str, list[float]] = defaultdict(list)
    for star in load_obs_li():
        for ion, rec in star["abundances"].items():
            element = _LI_ION_TO_ELEMENT.get(ion)
            if element is None:
                continue
            sigma = rec.get("sigma_Total")
            if sigma is not None and np.isfinite(sigma):
                per_element[element].append(float(sigma))

    return {el: float(np.median(vals)) for el, vals in per_element.items() if vals}


def add_dex_noise(
    X: np.ndarray,
    pairs: list[tuple[str, str]],
    element_sigmas: dict[str, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Add element-specific Gaussian observational noise (dex) to ratios.

    Draws eps[el] ~ N(0, sigma_el) once per element, then adds eps[x] - eps[y]
    to every [X/Y] ratio whose two elements are both in ``element_sigmas``.
    Ratios sharing an element therefore stay correlated (as in real [X/Fe]
    measurement errors). NaN cells are left untouched (NaN + finite = NaN).
    Elements absent from ``element_sigmas`` leave their ratios unchanged.
    """
    eps = {el: rng.normal(0.0, sig, size=X.shape[0])
           for el, sig in element_sigmas.items()}
    for j, (x_elem, y_elem) in enumerate(pairs):
        if x_elem in eps and y_elem in eps:
            X[:, j] = X[:, j] + (eps[x_elem] - eps[y_elem])
    return X


def generate(
    pisn_source: str,
    sn_source: str,
    n_samples: int,
    seed: int,
    cache_path: str = None,
    p_sn_only: float = 0.5,
    feh_range: tuple[float, float] = FEH_RANGE,
    dex_error: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build (X, y, diagnostics) for f_pisn regression.

    f_pisn == 0 is the pure-SNII (Pop II) case: the PISN term drops out and
    the abundance ratios are computed with a free, continuously sampled
    [Fe/H] (no PISN enrichment). Any other f_pisn uses the normal
    PISN + SNII mixing of salvadori_combined_abundratio.

    Args:
        pisn_source: "HW2002" or "Takahashi".
        sn_source: "Limongi18", "WW95", or "Nomoto13".
        n_samples: number of samples to draw.
        seed: RNG seed for reproducibility.
        cache_path: if set, load from / save to this .npz instead of recomputing.
        p_sn_only: probability of drawing an f_pisn == 0 (SN-only) sample.
        feh_range: [Fe/H] sampling range for SN-only samples (continuous).
        dex_error: 1-sigma Gaussian observational error (dex) added to each
                   element abundance; ratio [X/Y] then picks up eps_X - eps_Y
                   (ratios sharing an element stay correlated, as in real
                   [X/Fe] measurement errors). 0.0 (default) = noiseless mock.

    Returns:
        X: (n_samples, n_features) float array of abundance ratios.
        y: (n_samples,) float array of f_pisn targets (0 for SN-only).
        diagnostics: dict with keys f_ratio, tpop2, pisn_mass, feh, m_pop2
                     (each length n_samples) and feature_names.
    """
    rng = np.random.default_rng(seed)

    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        return data["X"], data["y"], dict(data["diagnostics"].item())

    pisn_entries, pisn_interp = _load_pisn(pisn_source)
    sn_entries, sn_source_obj = _load_sn(sn_source)

    # Precomputed transforms (done once, outside the sample loop).
    salv_sn_yields = salvadori_yields(sn_entries)

    pairs = _feature_pairs()
    n_features = len(pairs)

    X = np.full((n_samples, n_features), np.nan)
    y = np.empty(n_samples)
    f_ratio_diag = np.empty(n_samples)
    tpop2_diag = np.empty(n_samples)
    pisn_mass_diag = np.empty(n_samples)
    feh_diag = np.empty(n_samples)
    m_pop2_diag = np.empty(n_samples)

    t0 = time.time()
    for i in range(n_samples):
        if p_sn_only > 0.0 and rng.random() < p_sn_only:
            f_pisn = 0.0
        else:
            f_pisn = rng.uniform(*F_PISN_RANGE)

        f_ratio = loguniform.rvs(*F_RATIO_RANGE, random_state=rng)
        tpop2 = loguniform.rvs(*TPOP2_RANGE, random_state=rng)

        feh_diag[i] = np.nan
        m_pop2_diag[i] = np.nan
        pisn_mass_diag[i] = np.nan

        if f_pisn == 0.0:
            feh = rng.uniform(*feh_range)
            Z = Z_SUN * 10.0 ** feh
            sn_dr_data = sn_source_obj.load_for_metallicity(Z)
            # Raiteri (1996) lifetime is analytic and continuous in Z, so the
            # turnoff mass doesn't snap to the Limongi18 [Fe/H] grid.
            m_pop2 = raiteri_mass_from_lifetime(tpop2, Z)

            feh_diag[i] = feh
            m_pop2_diag[i] = np.nan if m_pop2 is None else m_pop2

            if m_pop2 is not None:
                for j, (x_elem, y_elem) in enumerate(pairs):
                    X[i, j] = salvadori_sn_only_abundance_ratio(
                        x_elem, y_elem, sn_dr_data, m_pop2, sn_input=sn_source_obj
                    )
        else:
            pisn_mass = rng.uniform(*PISN_MASS_RANGE)
            pisn_entry = pisn_interp(pisn_mass)
            pisn_mass_diag[i] = pisn_mass

            for j, (x_elem, y_elem) in enumerate(pairs):
                X[i, j] = salvadori_combined_abundratio(
                    x_elem, x_elem, y_elem, y_elem,
                    pisn_data=pisn_entry,
                    sn_data=sn_entries,
                    salv_sn_data=salv_sn_yields,
                    auto_sn=False, single_sn=False,
                    sn_input=sn_source_obj,
                    f_pisn=f_pisn, f_ratio=f_ratio, tpop2=tpop2,
                )

        y[i] = f_pisn
        f_ratio_diag[i] = f_ratio
        tpop2_diag[i] = tpop2

        if n_samples >= 10 and (i + 1) % max(1, n_samples // 10) == 0:
            frac = (i + 1) / n_samples
            elapsed = time.time() - t0
            eta = elapsed / frac - elapsed
            print(
                f"  {i + 1}/{n_samples} ({frac * 100:5.1f}%)  "
                f"elapsed {elapsed:7.1f}s  eta {eta:7.1f}s",
                flush=True,
            )

    print(f"  {n_samples}/{n_samples} (100.0%)  total {time.time() - t0:7.1f}s", flush=True)

    # Hygiene: clamp extreme low ratios and replace non-finite with NaN
    # (trainer will handle NaNs via mask/drop).
    valid = np.isfinite(X)
    X = np.where(valid & (X < RATIO_FLOOR), RATIO_FLOOR, X)
    X = np.where(np.isfinite(X), X, np.nan)

    # Observational error: add +/- dex Gaussian noise to each element's
    # abundance. Ratio [X/Y] then picks up eps_X - eps_Y, so ratios sharing an
    # element stay correlated (matching real [X/Fe] measurement errors). NaN
    # cells are left untouched (NaN + finite = NaN).
    if dex_error > 0.0:
        add_dex_noise(X, pairs, {el: dex_error for el in _abundance_elements()}, rng)

    diagnostics = {
        "f_ratio": f_ratio_diag,
        "tpop2": tpop2_diag,
        "pisn_mass": pisn_mass_diag,
        "feh": feh_diag,
        "m_pop2": m_pop2_diag,
        "feature_names": _feature_names(),
        "pisn_source": pisn_source,
        "sn_source": sn_source,
        "n_samples": n_samples,
        "seed": seed,
        "dex_error": dex_error,
    }

    if cache_path:
        np.savez_compressed(
            cache_path,
            X=X,
            y=y,
            diagnostics=diagnostics,
        )

    _print_fpisn_summary(y, pisn_source, sn_source)

    return X, y, diagnostics


__all__ = ["generate", "add_dex_noise", "li_plus_element_sigmas"]
