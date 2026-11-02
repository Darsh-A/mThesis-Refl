"""Prior and label diagnostics for a generated dataset."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..detect import DETECT_REASONS
from .ledger import ledger_fractions

_QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    """Return selected quantiles, ignoring non-finite values."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {f"p{int(q * 100):02d}": None for q in _QUANTILES}
    return {f"p{int(q * 100):02d}": float(np.quantile(values, q)) for q in _QUANTILES}


def _rate(mask: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Return count and positive fraction for a boolean mask."""
    count = int(np.count_nonzero(mask))
    return {
        "count": count,
        "y_rate": float(np.count_nonzero(y[mask]) / count) if count else None,
    }


def _rate_by(values: np.ndarray, y: np.ndarray) -> dict[str, dict[str, Any]]:
    """Return class balance for each distinct integer value."""
    out: dict[str, dict[str, Any]] = {}
    for value in np.unique(values):
        out[str(int(value))] = _rate(values == value, y)
    return out


def _presence_diagnostics(meta: dict[str, Any]) -> dict[str, Any]:
    """Split the rejection ledger by whether a PISN was drawn."""
    ledger = meta["ledger"]
    drawn = float(ledger.get("drawn", 0))
    accepted_total = float(ledger.get("accepted", 0) + ledger.get("discarded", 0))
    pisn_drawn = float(ledger.get("pisn_drawn", 0))
    no_pisn_drawn = float(ledger.get("no_pisn_drawn", 0))
    pisn_accepted = float(
        ledger.get("pisn_accepted", 0) + ledger.get("pisn_discarded", 0)
    )
    no_pisn_accepted = float(
        ledger.get("no_pisn_accepted", 0) + ledger.get("no_pisn_discarded", 0)
    )
    prior = meta.get("config", {}).get("prior", {})
    p_zero = prior.get("n_pisn_probs", [0.5])[0]

    def fraction(numerator: float, denominator: float) -> float | None:
        return float(numerator / denominator) if denominator > 0 else None

    def stage_rates(prefix: str, total: float) -> dict[str, float | None]:
        return {
            "drawn": total,
            "r1": fraction(ledger.get(f"{prefix}_r1", 0), total),
            "r2": fraction(ledger.get(f"{prefix}_r2", 0), total),
            "r3": fraction(ledger.get(f"{prefix}_r3", 0), total),
            "r4": fraction(ledger.get(f"{prefix}_r4", 0), total),
            "accepted": fraction(
                ledger.get(f"{prefix}_accepted", 0)
                + ledger.get(f"{prefix}_discarded", 0),
                total,
            ),
        }

    return {
        "prior_pisn_fraction": 1.0 - p_zero,
        "drawn_pisn_fraction": fraction(pisn_drawn, drawn),
        "accepted_pisn_fraction": fraction(pisn_accepted, accepted_total),
        "retention_pisn": fraction(pisn_accepted, pisn_drawn),
        "retention_no_pisn": fraction(no_pisn_accepted, no_pisn_drawn),
        "pisn": stage_rates("pisn", pisn_drawn),
        "no_pisn": stage_rates("no_pisn", no_pisn_drawn),
    }


def _rate_by_decade(values: np.ndarray, y: np.ndarray) -> dict[str, dict[str, Any]]:
    """Return class balance per decade of a positive quantity."""
    positive = values > 0.0
    decades = np.full(values.shape, -999, dtype=int)
    decades[positive] = np.floor(np.log10(values[positive])).astype(int)
    out: dict[str, dict[str, Any]] = {}
    for decade in np.unique(decades[positive]):
        out[str(int(decade))] = _rate(decades == decade, y)
    return out


def _rate_by_feh_bins(feh: np.ndarray, y: np.ndarray) -> dict[str, dict[str, Any]]:
    """Return class balance in coarse [Fe/H] bins."""
    edges = np.asarray([-5.0, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, 0.0])
    finite = np.isfinite(feh)
    indices = np.digitize(feh[finite], edges) - 1
    indices = np.clip(indices, 0, len(edges) - 2)
    out: dict[str, dict[str, Any]] = {}
    for index in np.unique(indices):
        low = edges[index]
        high = edges[index + 1]
        mask = np.zeros(feh.shape, dtype=bool)
        mask[np.flatnonzero(finite)[indices == index]] = True
        out[f"{low:+.1f}..{high:+.1f}"] = _rate(mask, y)
    return out


def diagnose(
    arrays: dict[str, np.ndarray],
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Summarise labels, class balance, coverage, and the rejection ledger."""
    y = arrays["y"].astype(bool)
    n_pisn = arrays["n_pisn"]
    n_iii = arrays["n_iii"]
    z_star = arrays["z_star"].astype(float)
    t_myr = arrays["t_myr"].astype(float)
    feh = arrays["feh"].astype(float)
    f_zp = arrays["f_zp"].astype(float)
    f_fep = arrays["f_fep"].astype(float)

    metal = arrays["M_z_tot"].astype(float)
    source_metal = arrays["source_metal"].astype(float)
    popii_share = np.divide(
        source_metal[:, 2],
        metal,
        out=np.zeros_like(metal),
        where=metal > 0.0,
    )

    ledger = meta["ledger"]

    report: dict[str, Any] = {
        "n": int(len(y)),
        "y_rate": float(np.mean(y)),
        "ledger": ledger_fractions(meta),
        "presence": _presence_diagnostics(meta),
        "r2_reasons": {
            "no_pre_enrichment": int(ledger.get("r2_no_pre", 0)),
            "above_z_max": int(ledger.get("r2_above_z_max", 0)),
            "below_z_cr": int(ledger.get("r2_below_z_cr", 0)),
            "other": int(ledger.get("r2_other", 0)),
        },
        "detect_reasons": {
            name: int(np.count_nonzero(arrays["detect_reason"] == index))
            for index, name in enumerate(DETECT_REASONS)
        },
        "f_zp": {
            "zero_fraction": float(np.mean(f_zp == 0.0)),
            "near_zero_fraction": float(np.mean((f_zp > 0.0) & (f_zp < 0.01))),
            "near_one_fraction": float(np.mean(f_zp > 0.99)),
            "quantiles": _quantiles(f_zp),
        },
        "f_fep": {
            "zero_fraction": float(np.mean(f_fep == 0.0)),
            "log10_quantiles": _quantiles(
                np.log10(f_fep[f_fep > 0.0]) if np.any(f_fep > 0.0) else np.array([])
            ),
        },
        "z_star": {"log10_quantiles": _quantiles(np.log10(z_star[z_star > 0.0]))},
        "feh": {"quantiles": _quantiles(feh)},
        "mdil": {"log10_quantiles": _quantiles(np.log10(arrays["mdil"]))},
        "mstar_ii": {"log10_quantiles": _quantiles(np.log10(arrays["mstar_ii"]))},
        "y_by_n_pisn": _rate_by(n_pisn, y),
        "y_by_n_iii": _rate_by(n_iii, y),
        "y_by_z_decade": _rate_by_decade(z_star, y),
        "y_by_t_decade": _rate_by_decade(t_myr, y),
        "y_by_feh": _rate_by_feh_bins(feh, y),
        "feature_coverage": {
            name: float(np.mean(arrays["ratio_mask"][:, index]))
            for index, name in enumerate(meta["feature_names"])
        },
        "mixture_cells": {},
    }

    for pisn_present in (False, True):
        for iii_present in (False, True):
            for popii_dominant in (False, True):
                mask = (
                    (n_pisn > 0) == pisn_present
                ) & (
                    (n_iii > 0) == iii_present
                ) & (
                    (popii_share > 0.5) == popii_dominant
                )
                key = f"P{int(pisn_present)}_III{int(iii_present)}_II{int(popii_dominant)}"
                report["mixture_cells"][key] = _rate(mask, y)

    report["feature_names"] = list(meta["feature_names"])
    return report


def format_report(report: dict[str, Any]) -> str:
    """Format a diagnostics report for terminal output."""
    lines = [
        f"samples          {report['n']}",
        f"positive rate    {report['y_rate']:.3f}",
        "cut fractions    "
        + "  ".join(f"{key}={value:.3f}" for key, value in report["ledger"].items()),
        "presence         "
        + (
            f"prior={report['presence']['prior_pisn_fraction']:.3f}  "
            f"drawn={report['presence']['drawn_pisn_fraction']:.3f}  "
            f"accepted={report['presence']['accepted_pisn_fraction']:.3f}"
            if report["presence"]["drawn_pisn_fraction"] is not None
            and report["presence"]["accepted_pisn_fraction"] is not None
            else "n/a"
        ),
        "R2 reasons       "
        + "  ".join(
            f"{key}={value}" for key, value in report["r2_reasons"].items()
        ),
        "detect reasons   "
        + "  ".join(f"{key}={value}" for key, value in report["detect_reasons"].items()),
        f"f_ZP zero        {report['f_zp']['zero_fraction']:.3f}",
        f"f_ZP near one    {report['f_zp']['near_one_fraction']:.3f}",
        "f_ZP quantiles   "
        + "  ".join(
            f"{key}={value:.3f}"
            for key, value in report["f_zp"]["quantiles"].items()
            if value is not None
        ),
        "feature coverage "
        + "  ".join(
            f"{key}={value:.2f}" for key, value in report["feature_coverage"].items()
        ),
        "class balance by n_PISN",
    ]
    for key, value in report["y_by_n_pisn"].items():
        lines.append(
            f"  {key:>3s}  n={value['count']:>6d}  y={value['y_rate']}"
        )
    lines.append("mixture cells")
    for key, value in report["mixture_cells"].items():
        lines.append(
            f"  {key:<16s} n={value['count']:>6d}  y={value['y_rate']}"
        )
    return "\n".join(lines)
