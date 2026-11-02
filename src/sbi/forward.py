"""Noiseless Salvadori forward model: theta -> abundance ratios.

theta = (f_pisn, log10 f_ratio, log10 tpop2, M_pisn) -> m [X/Y] ratios.

This is the classical (non-ML) simulator, reusing the same machinery as
``src.randomForest.dataset.generate``.  It is used by the non-ML inference to
check the NPE against direct forward-model fitting.
"""

from __future__ import annotations

import numpy as np

from src.randomForest.dataset import _feature_pairs, _load_pisn, _load_sn
from src.salvadori_funcs import (
    _SOLAR,
    salvadori_combined_abundratio,
    salvadori_Y_X_II,
    salvadori_Y_Z_II,
    salvadori_yields,
)
from src.formulas import raiteri_mass_from_lifetime
from params import atomic_mass as ATOMIC_MASS


def make_forward(sn_source: str = "Limongi18", pisn_source: str = "Takahashi"):
    """Build and return a ``forward(theta) -> x`` callable.

    Returns (forward, pairs) where ``pairs`` is the list of (X, Y) element
    tuples matching the feature columns (used to build a noise covariance).
    """
    pisn_entries, pisn_interp = _load_pisn(pisn_source)
    sn_entries, sn_source_obj = _load_sn(sn_source)
    salv_sn_yields = salvadori_yields(sn_entries)
    pairs = _feature_pairs()

    def forward(theta: np.ndarray) -> np.ndarray:
        f_pisn, log10_f_ratio, log10_tpop2, m_pisn = theta
        f_ratio = 10.0 ** log10_f_ratio
        tpop2 = 10.0 ** log10_tpop2
        pisn_entry = pisn_interp(m_pisn)

        x = np.empty(len(pairs))
        for j, (x_elem, y_elem) in enumerate(pairs):
            x[j] = salvadori_combined_abundratio(
                x_elem, x_elem, y_elem, y_elem,
                pisn_data=pisn_entry,
                sn_data=sn_entries,
                salv_sn_data=salv_sn_yields,
                auto_sn=False,
                single_sn=False,
                sn_input=sn_source_obj,
                f_pisn=f_pisn,
                f_ratio=f_ratio,
                tpop2=tpop2,
            )
        return x

    return forward, pairs


def make_forward_fast(sn_source: str = "Limongi18", pisn_source: str = "Takahashi",
                      elements: list[str] | None = None):
    """Same forward model, but ~20x faster.

    The naive ``salvadori_combined_abundratio`` recomputes the IMF-integrated
    SN yields and the metallicity interpolation once per *ratio*.  Here those
    per-theta quantities (Z_star, m_popII, Yz_sn, and the per-element SN yields)
    are computed once, then the m ratios are assembled by cheap arithmetic.
    Mathematically identical to ``make_forward``; used to make MCMC tractable.

    ``elements`` optionally overrides the default 15-element feature set so the
    model can be built for arbitrary element subsets (e.g. including Zn).
    """
    pisn_entries, pisn_interp = _load_pisn(pisn_source)
    sn_entries, sn_source_obj = _load_sn(sn_source)
    if elements is None:
        pairs = _feature_pairs()
    else:
        from itertools import combinations

        pairs = list(combinations(elements, 2))
    elems = sorted({e for p in pairs for e in p})

    def forward(theta: np.ndarray) -> np.ndarray:
        f_pisn, log10_f_ratio, log10_tpop2, m_pisn = theta
        f_ratio = 10.0 ** log10_f_ratio
        tpop2 = 10.0 ** log10_tpop2

        # element-combined, mass-normalized PISN yields
        pisn_yields = pisn_interp(m_pisn)["yields"]
        Yz_pisn = sum(v for el, v in pisn_yields.items() if el not in ("H", "He"))
        Z_star = f_ratio * Yz_pisn / f_pisn

        if not np.isfinite(Z_star) or Z_star <= 0.0:
            return np.full(len(pairs), np.nan)

        try:
            sn_dr_data = sn_source_obj.load_for_metallicity(Z_star)
            m_popII = raiteri_mass_from_lifetime(tpop2, Z_star)
        except (ValueError, FloatingPointError, ZeroDivisionError):
            return np.full(len(pairs), np.nan)

        if m_popII is None:
            return np.full(len(pairs), np.nan)

        Y_sn = {
            el: salvadori_Y_X_II(sn_dr_data, el, m_popII, sn_input=sn_source_obj)
            for el in elems
        }
        Yz_sn = salvadori_Y_Z_II(sn_dr_data, m_popII, sn_input=sn_source_obj)

        beta = (1.0 - f_pisn) / f_pisn
        x = np.empty(len(pairs))
        for j, (x_elem, y_elem) in enumerate(pairs):
            Yx_pisn = pisn_yields.get(x_elem, 0.0)
            Yy_pisn = pisn_yields.get(y_elem, 0.0)
            if Yz_sn <= 0.0:
                sn_term_1 = sn_term_2 = 0.0
            else:
                sn_term_1 = (Yz_pisn / Yz_sn) * Y_sn[x_elem]
                sn_term_2 = (Yz_pisn / Yz_sn) * Y_sn[y_elem]

            num = Yx_pisn + beta * sn_term_1
            den = Yy_pisn + beta * sn_term_2
            A1 = _SOLAR[x_elem]["val"]
            A2 = _SOLAR[y_elem]["val"]
            x[j] = np.log10(num / den) - (A1 - A2) - np.log10(
                ATOMIC_MASS[x_elem] / ATOMIC_MASS[y_elem]
            )
        return x

    return forward, pairs
