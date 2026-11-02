"""Plots for generated datasets, limits, and validation runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..constants import ELEMENT_INDEX
from ..detect import DETECT_REASONS
from .ledger import ledger_fractions
from .priors import diagnose

_COLORS = {
    "blue": "#4c72b0",
    "orange": "#dd8452",
    "green": "#55a868",
    "red": "#c44e52",
    "purple": "#8172b3",
    "grey": "#8c8c8c",
}


def _output(path: str | Path) -> Path:
    """Create the parent directory and return the output path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save(fig: plt.Figure, path: str | Path) -> None:
    """Save and close a figure."""
    fig.savefig(_output(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _finite(values: np.ndarray) -> np.ndarray:
    """Return finite values as floats."""
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _bin_centres(edges: np.ndarray, log: bool = False) -> np.ndarray:
    """Return bin centres, geometric when the edges are logarithmic."""
    if log:
        return np.sqrt(edges[:-1] * edges[1:])
    return 0.5 * (edges[:-1] + edges[1:])


def _binned_quantiles(
    x: np.ndarray,
    y: np.ndarray,
    edges: np.ndarray,
    log: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Median and 16-84 percentiles of ``y`` in ``x`` bins."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    index = np.digitize(x, edges) - 1
    centres = _bin_centres(edges, log=log)
    median = np.full(centres.shape, np.nan)
    low = np.full(centres.shape, np.nan)
    high = np.full(centres.shape, np.nan)
    count = np.zeros(centres.shape, dtype=int)
    for i in range(len(centres)):
        values = y[index == i]
        count[i] = values.size
        if values.size:
            median[i] = np.median(values)
            low[i] = np.quantile(values, 0.16)
            high[i] = np.quantile(values, 0.84)
    return centres, median, low, high


def _draw_band(
    ax: plt.Axes,
    x: np.ndarray,
    median: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    label: str,
    color: str,
) -> None:
    """Draw a median line with a 16-84 band."""
    good = np.isfinite(median)
    ax.plot(x[good], median[good], color=color, label=label, lw=1.8)
    ax.fill_between(
        x[good],
        low[good],
        high[good],
        color=color,
        alpha=0.18,
        linewidth=0,
    )


def _rate_binned(
    x: np.ndarray,
    y: np.ndarray,
    edges: np.ndarray,
    log: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Positive fraction and Wilson interval in ``x`` bins."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=bool)
    good = np.isfinite(x)
    index = np.digitize(x[good], edges) - 1
    y = y[good]
    centres = _bin_centres(edges, log=log)
    rate = np.full(centres.shape, np.nan)
    error = np.full(centres.shape, np.nan)
    count = np.zeros(centres.shape, dtype=int)
    for i in range(len(centres)):
        values = y[index == i]
        count[i] = values.size
        if values.size:
            p = values.mean()
            rate[i] = p
            error[i] = np.sqrt(max(p * (1.0 - p) / values.size, 1e-12))
    return centres, rate, error, count


def _draw_rate(
    ax: plt.Axes,
    x: np.ndarray,
    rate: np.ndarray,
    error: np.ndarray,
    count: np.ndarray,
    label: str,
    color: str,
) -> None:
    """Draw a positive-fraction curve with error bars."""
    good = np.isfinite(rate)
    ax.errorbar(
        x[good],
        rate[good],
        yerr=error[good],
        color=color,
        marker="o",
        ms=4,
        lw=1.5,
        capsize=2,
        label=label,
    )
    for xi, ri, ni in zip(x[good], rate[good], count[good]):
        ax.annotate(
            str(ni),
            (xi, ri),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=7,
            color=_COLORS["grey"],
        )


def _correlation_matrix(values: np.ndarray) -> np.ndarray:
    """Pairwise-complete Pearson correlation matrix."""
    values = np.asarray(values, dtype=float)
    n = values.shape[1]
    out = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i, n):
            good = np.isfinite(values[:, i]) & np.isfinite(values[:, j])
            if good.sum() > 2:
                out[i, j] = out[j, i] = np.corrcoef(
                    values[good, i], values[good, j]
                )[0, 1]
    return out


def plot_ledger(meta: dict[str, Any], path: str | Path) -> None:
    """Plot the rejection fractions and the R2 reasons."""
    fractions = ledger_fractions(meta)
    ledger = meta["ledger"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    labels = ["R1", "R2", "R3", "accepted"]
    values = [
        fractions["r1"],
        fractions["r2"],
        fractions["r3"],
        fractions["accepted_total"],
    ]
    colors = [_COLORS["red"], _COLORS["orange"], _COLORS["purple"], _COLORS["green"]]
    bars = axes[0].barh(labels, values, color=colors)
    axes[0].set_xlim(0, max(values) * 1.25 + 0.02)
    axes[0].set_xlabel("fraction of draws")
    axes[0].set_title(f"Rejection ledger (n={ledger.get('drawn', 0):,})")
    for bar, value in zip(bars, values):
        axes[0].text(
            value + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=9,
        )
    if ledger.get("discarded"):
        axes[0].text(
            0.98,
            0.02,
            f"retained {ledger.get('accepted', 0):,} of "
            f"{ledger.get('accepted', 0) + ledger['discarded']:,} accepted",
            transform=axes[0].transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
        )

    reasons = {
        "no pre-enrichment": ledger.get("r2_no_pre", 0),
        "above z_max": ledger.get("r2_above_z_max", 0),
        "below z_cr": ledger.get("r2_below_z_cr", 0),
        "other": ledger.get("r2_other", 0),
    }
    drawn = max(ledger.get("drawn", 0), 1)
    total = sum(reasons.values())
    axes[1].barh(
        list(reasons),
        [value / drawn for value in reasons.values()],
        color=_COLORS["orange"],
    )
    axes[1].set_xlabel("fraction of draws")
    axes[1].set_title(f"R2 reasons (n={total:,})")
    for i, value in enumerate(reasons.values()):
        axes[1].text(
            value / drawn + 0.002,
            i,
            f"{value:,}",
            va="center",
            fontsize=9,
        )
    fig.tight_layout()
    _save(fig, path)


def plot_labels(arrays: dict[str, np.ndarray], path: str | Path) -> None:
    """Plot the regression labels, detection statistic, and label reasons."""
    f_zp = np.asarray(arrays["f_zp"], dtype=float)
    f_fep = np.asarray(arrays["f_fep"], dtype=float)
    y = np.asarray(arrays["y"], dtype=bool)
    d = np.asarray(arrays["D"], dtype=float)
    reason = np.asarray(arrays["detect_reason"], dtype=int)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].hist(f_zp, bins=np.linspace(0, 1, 41), color=_COLORS["blue"])
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel(r"$f_Z^P$")
    axes[0, 0].set_ylabel("count")
    axes[0, 0].set_title(f"zero: {np.mean(f_zp == 0):.3f}   near one: {np.mean(f_zp > 0.99):.3f}")

    positive = f_fep[f_fep > 0]
    if positive.size:
        axes[0, 1].hist(
            np.log10(positive),
            bins=np.linspace(np.log10(positive.min()), 0.0, 60),
            color=_COLORS["orange"],
        )
    axes[0, 1].set_xlabel(r"$\log_{10} f_{\rm Fe}^P$")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].set_title(
        f"zero: {np.mean(f_fep == 0):.3f}   exactly one: {np.mean(f_fep == 1):.3f}"
    )

    finite = d[np.isfinite(d) & (d > 0)]
    if finite.size:
        axes[1, 0].hist(np.log10(finite), bins=40, color=_COLORS["green"])
    axes[1, 0].axvline(np.log10(3.0), color=_COLORS["red"], ls="--", label="k=3")
    axes[1, 0].set_xlabel(r"$\log_{10} D$")
    axes[1, 0].set_ylabel("count")
    axes[1, 0].legend()
    axes[1, 0].set_title(f"y rate: {y.mean():.3f}")

    counts = [np.count_nonzero(reason == i) for i in range(len(DETECT_REASONS))]
    axes[1, 1].bar(DETECT_REASONS, counts, color=_COLORS["purple"])
    axes[1, 1].set_ylabel("count")
    axes[1, 1].tick_params(axis="x", rotation=30)
    axes[1, 1].set_title("detection reasons")
    fig.tight_layout()
    _save(fig, path)


def plot_class_balance(
    arrays: dict[str, np.ndarray],
    path: str | Path,
) -> None:
    """Plot positive fraction against the main nuisance variables."""
    y = np.asarray(arrays["y"], dtype=bool)
    n_pisn = np.asarray(arrays["n_pisn"], dtype=int)
    n_iii = np.asarray(arrays["n_iii"], dtype=int)
    z_star = np.asarray(arrays["z_star"], dtype=float)
    t_myr = np.asarray(arrays["t_myr"], dtype=float)
    mdil = np.asarray(arrays["mdil"], dtype=float)
    mstar = np.asarray(arrays["mstar_ii"], dtype=float)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, values in zip(axes[0, :2], (n_pisn, n_iii)):
        unique = np.unique(values)
        rates = [y[values == value].mean() for value in unique]
        counts = [np.count_nonzero(values == value) for value in unique]
        ax.bar(unique, rates, color=_COLORS["blue"])
        ax.set_ylim(0, 1)
        ax.set_xlabel("count")
        ax.set_ylabel("positive fraction")
        for value, rate, count in zip(unique, rates, counts):
            ax.text(value, rate + 0.03, f"{count}", ha="center", fontsize=8)
    axes[0, 0].set_title("PISN count")
    axes[0, 1].set_title("Pop III count")

    panels = [
        (axes[0, 2], np.log10(z_star), np.linspace(-8, -1, 8), "log10 Z*", False),
        (axes[1, 0], np.log10(t_myr), np.linspace(0, 2, 7), "log10 t [Myr]", False),
        (axes[1, 1], np.log10(mdil), np.linspace(3, 7, 7), "log10 M_dil", False),
        (axes[1, 2], np.log10(mstar), np.linspace(3, 7, 7), "log10 M*II", False),
    ]
    for ax, values, edges, label, log in panels:
        centres, rate, error, count = _rate_binned(values, y, edges, log=log)
        _draw_rate(ax, centres, rate, error, count, "positive fraction", _COLORS["red"])
        ax.axhline(y.mean(), color=_COLORS["grey"], ls=":", lw=1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(label)
        ax.set_ylabel("positive fraction")
    axes[0, 2].set_title("metallicity")
    axes[1, 0].set_title("clock t")
    axes[1, 1].set_title("dilution mass")
    axes[1, 2].set_title("Pop II mass")
    fig.tight_layout()
    _save(fig, path)


def plot_features(
    arrays: dict[str, np.ndarray],
    meta: dict[str, Any],
    path: str | Path,
) -> None:
    """Plot feature coverage, correlations, and typical abundance ratios."""
    features = np.asarray(arrays["features"], dtype=float)
    names = list(meta["feature_names"])
    mask = np.asarray(arrays["ratio_mask"], dtype=bool)
    fig = plt.figure(figsize=(14, 8))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.5, 1.0))

    ax = fig.add_subplot(grid[0, 0])
    corr = _correlation_matrix(features)
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title("feature correlation")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(grid[0, 1])
    coverage = mask.mean(axis=0)
    order = np.argsort(coverage)
    ax.barh(np.asarray(names)[order], coverage[order], color=_COLORS["green"])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("fraction valid")
    ax.set_title("feature coverage")

    ax = fig.add_subplot(grid[1, :])
    medians = []
    lower = []
    upper = []
    for index in range(len(names)):
        values = _finite(features[:, index])
        if values.size:
            medians.append(np.median(values))
            lower.append(np.quantile(values, 0.16))
            upper.append(np.quantile(values, 0.84))
        else:
            medians.append(np.nan)
            lower.append(np.nan)
            upper.append(np.nan)
    medians = np.asarray(medians)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    ax.errorbar(
        range(len(names)),
        medians,
        yerr=[medians - lower, upper - medians],
        fmt="o",
        color=_COLORS["blue"],
        capsize=2,
    )
    ax.axhline(0, color=_COLORS["grey"], lw=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, fontsize=8)
    ax.set_ylabel("[X/Fe]")
    ax.set_title("median and 16-84 percentile")
    fig.tight_layout()
    _save(fig, path)


def plot_mixture_cells(report: dict[str, Any], path: str | Path) -> None:
    """Plot the eight mixture-cell counts and positive fractions."""
    columns = ["III0_II0", "III0_II1", "III1_II0", "III1_II1"]
    rate = np.full((2, 4), np.nan)
    count = np.zeros((2, 4), dtype=int)
    for p in (0, 1):
        for j, column in enumerate(columns):
            cell = report["mixture_cells"][f"P{p}_{column}"]
            count[p, j] = cell["count"]
            if cell["y_rate"] is not None:
                rate[p, j] = cell["y_rate"]

    fig, ax = plt.subplots(figsize=(8, 3.8))
    image = ax.imshow(rate, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(4))
    ax.set_xticklabels(columns)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["no PISN", "PISN"])
    for p in (0, 1):
        for j in range(4):
            text = "n=0" if count[p, j] == 0 else f"n={count[p, j]}\ny={rate[p, j]:.2f}"
            ax.text(j, p, text, ha="center", va="center", color="white", fontsize=9)
    ax.set_title("mixture-cell coverage")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="positive fraction")
    fig.tight_layout()
    _save(fig, path)


def _event_metal(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return PISN and Pop III metal mass per event."""
    source = np.asarray(arrays["source_metal"], dtype=float)
    n_pisn = np.asarray(arrays["n_pisn"], dtype=float)
    n_iii = np.asarray(arrays["n_iii"], dtype=float)
    pisn = np.full(n_pisn.shape, np.nan)
    iii = np.full(n_iii.shape, np.nan)
    np.divide(source[:, 0], n_pisn, out=pisn, where=n_pisn > 0)
    np.divide(source[:, 1], n_iii, out=iii, where=n_iii > 0)
    return pisn, iii


def _quantile_summary(values: np.ndarray) -> dict[str, float | None]:
    """Return 5th, 50th, and 95th percentiles."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return {"p05": None, "p50": None, "p95": None}
    return {
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
    }


def event_mass_summary(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Summarise per-event metal masses and source shares."""
    pisn, iii = _event_metal(arrays)
    metal = np.asarray(arrays["M_z_tot"], dtype=float)
    source = np.asarray(arrays["source_metal"], dtype=float)
    share = np.full(source.shape, np.nan)
    np.divide(source, metal[:, None], out=share, where=metal[:, None] > 0)
    return {
        "pisn_per_event": _quantile_summary(pisn),
        "popiii_per_event": _quantile_summary(iii),
        "source_share": {
            name: _quantile_summary(share[:, index])
            for index, name in enumerate(("PISN", "POPIII", "POPII", "SNIA"))
        },
    }


def plot_event_masses(
    arrays: dict[str, np.ndarray],
    meta: dict[str, Any],
    path: str | Path,
) -> None:
    """Plot PISN and Pop III event masses and source shares."""
    pisn, iii = _event_metal(arrays)
    source = np.asarray(arrays["source_metal"], dtype=float)
    metal = np.asarray(arrays["M_z_tot"], dtype=float)
    share = np.full(source.shape, np.nan)
    np.divide(source, metal[:, None], out=share, where=metal[:, None] > 0)
    f_zp = np.asarray(arrays["f_zp"], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    masses = np.asarray(arrays["pisn_masses"], dtype=float)
    first_mass = masses[:, 0]
    good = np.isfinite(pisn) & (first_mass > 0)
    axes[0].scatter(
        first_mass[good],
        pisn[good],
        c=f_zp[good],
        s=10,
        cmap="viridis",
        alpha=0.6,
    )
    axes[0].set_xlabel(r"first PISN mass [$M_\odot$]")
    axes[0].set_ylabel(r"PISN metal mass per event [$M_\odot$]")
    axes[0].set_yscale("log")
    axes[0].set_title("PISN event mass")
    cbar = fig.colorbar(axes[0].collections[0], ax=axes[0])
    cbar.set_label(r"$f_{ZP}$")

    positive = np.concatenate([pisn[np.isfinite(pisn)], iii[np.isfinite(iii)]])
    positive = positive[positive > 0.0]
    if positive.size:
        bins = np.geomspace(positive.min(), positive.max(), 45)
        axes[1].hist(
            pisn[np.isfinite(pisn) & (pisn > 0)],
            bins=bins,
            histtype="step",
            lw=2,
            color=_COLORS["blue"],
            label="PISN",
        )
        axes[1].hist(
            iii[np.isfinite(iii) & (iii > 0)],
            bins=bins,
            histtype="step",
            lw=2,
            color=_COLORS["orange"],
            label="Pop III",
        )
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"metal mass per event [$M_\odot$]")
    axes[1].set_ylabel("events")
    axes[1].set_title("event metal mass")
    axes[1].legend()

    labels = ("PISN", "Pop III", "Pop II")
    for index, (name, color) in enumerate(zip(labels, _COLORS.values())):
        values = share[:, index]
        values = values[np.isfinite(values) & (values >= 0.0)]
        axes[2].hist(
            values,
            bins=np.linspace(0.0, 1.0, 41),
            histtype="step",
            lw=2,
            color=color,
            label=name,
        )
    axes[2].set_xlabel(r"metal-mass share $M_Z^{s}/M_Z^{\rm tot}$")
    axes[2].set_ylabel("samples")
    axes[2].set_title("source shares")
    axes[2].legend()
    fig.suptitle(
        "PISN masses uniform in [140, 270]; Pop III combo-uniform over 10 "
        "(mass, energy) groups, then uniform within group",
        fontsize=9,
    )
    fig.tight_layout()
    _save(fig, path)


def plot_pisn_mass(
    arrays: dict[str, np.ndarray],
    meta: dict[str, Any],
    path: str | Path,
) -> None:
    """Plot labels and abundance ratios against the first PISN mass."""
    n_pisn = np.asarray(arrays["n_pisn"], dtype=int)
    selected = n_pisn > 0
    mass = np.asarray(arrays["pisn_masses"], dtype=float)[selected, 0]
    f_zp = np.asarray(arrays["f_zp"], dtype=float)[selected]
    f_fep = np.asarray(arrays["f_fep"], dtype=float)[selected]
    y = np.asarray(arrays["y"], dtype=bool)[selected]
    features = np.asarray(arrays["features"], dtype=float)[selected]
    names = list(meta["feature_names"])
    edges = np.arange(135, 276, 10)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    centres, median, low, high = _binned_quantiles(mass, f_zp, edges)
    _draw_band(axes[0, 0], centres, median, low, high, r"$f_Z^P$", _COLORS["blue"])
    axes[0, 0].set_ylabel(r"$f_Z^P$")
    axes[0, 0].set_title("PISN metal share")

    positive = np.where(f_fep > 0, f_fep, np.nan)
    centres, median, low, high = _binned_quantiles(mass, np.log10(positive), edges)
    _draw_band(
        axes[0, 1], centres, median, low, high, r"$f_{\rm Fe}^P$", _COLORS["orange"]
    )
    axes[0, 1].set_ylabel(r"$\log_{10} f_{\rm Fe}^P$")
    axes[0, 1].set_title("PISN iron share")

    centres, rate, error, count = _rate_binned(mass, y, edges)
    _draw_rate(axes[1, 0], centres, rate, error, count, "positive fraction", _COLORS["red"])
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_ylabel("positive fraction")
    axes[1, 0].set_title("detectability")

    for element, color in zip(
        ("O", "Mg", "Si", "Ca", "Zn"),
        (_COLORS["blue"], _COLORS["orange"], _COLORS["green"], _COLORS["red"], _COLORS["purple"]),
    ):
        name = f"{element}/{meta['reference']}"
        if name not in names:
            continue
        index = names.index(name)
        centres, median, low, high = _binned_quantiles(
            mass, features[:, index], edges
        )
        _draw_band(axes[1, 1], centres, median, low, high, name, color)
    axes[1, 1].set_ylabel("[X/Fe]")
    axes[1, 1].set_title("PISN yield signature")
    axes[1, 1].legend(fontsize=7, ncol=2)

    for ax in axes.ravel():
        ax.set_xlabel("first PISN mass [M$_{\\odot}$]")
    fig.tight_layout()
    _save(fig, path)


def plot_clock(arrays: dict[str, np.ndarray], path: str | Path) -> None:
    """Plot the two-clock behaviour: pre-enrichment, Z*, and detectability."""
    delta = np.asarray(arrays["delta_myr"], dtype=float)
    m_z_pre = np.asarray(arrays["M_z_pre"], dtype=float)
    m_z_tot = np.asarray(arrays["M_z_tot"], dtype=float)
    z_star = np.asarray(arrays["z_star"], dtype=float)
    feh = np.asarray(arrays["feh"], dtype=float)
    y = np.asarray(arrays["y"], dtype=bool)
    edges = np.geomspace(1, 100, 9)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    pre_share = np.divide(
        m_z_pre, m_z_tot, out=np.full_like(m_z_pre, np.nan), where=m_z_tot > 0
    )
    centres, median, low, high = _binned_quantiles(delta, pre_share, edges, log=True)
    _draw_band(axes[0, 0], centres, median, low, high, "pre-enriched share", _COLORS["blue"])
    axes[0, 0].set_ylabel(r"$M_{Z,\rm pre}/M_{Z,\rm tot}$")
    axes[0, 0].set_title("metals present before Pop II")

    centres, median, low, high = _binned_quantiles(delta, np.log10(z_star), edges, log=True)
    _draw_band(axes[0, 1], centres, median, low, high, r"$Z_\star$", _COLORS["orange"])
    axes[0, 1].set_ylabel(r"$\log_{10} Z_\star$")
    axes[0, 1].set_title("pre-enrichment metallicity")

    centres, median, low, high = _binned_quantiles(delta, feh, edges, log=True)
    _draw_band(axes[1, 0], centres, median, low, high, "[Fe/H]", _COLORS["green"])
    axes[1, 0].set_ylabel("[Fe/H]")
    axes[1, 0].set_title("final iron abundance")

    centres, rate, error, count = _rate_binned(delta, y, edges, log=True)
    _draw_rate(axes[1, 1], centres, rate, error, count, "positive fraction", _COLORS["red"])
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_ylabel("positive fraction")
    axes[1, 1].set_title("detectability")

    for ax in axes.ravel():
        ax.set_xscale("log")
        ax.set_xlabel(r"$\Delta$ [Myr]")
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
        ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
        ax.minorticks_off()
        for lifetime, label in ((2.5, "PISN"), (10.4, "15 M$_\\odot$"), (20.2, "9 M$_\\odot$")):
            ax.axvline(lifetime, color=_COLORS["grey"], ls=":", lw=0.8)
    fig.tight_layout()
    _save(fig, path)


def plot_dilution(arrays: dict[str, np.ndarray], path: str | Path) -> None:
    """Plot the dilution mass against metallicity and the labels."""
    mdil = np.asarray(arrays["mdil"], dtype=float)
    z_star = np.asarray(arrays["z_star"], dtype=float)
    z_final = np.asarray(arrays["z_final"], dtype=float)
    feh = np.asarray(arrays["feh"], dtype=float)
    mstar = np.asarray(arrays["mstar_ii"], dtype=float)
    y = np.asarray(arrays["y"], dtype=bool)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    panels = [
        (axes[0, 0], z_star, r"$Z_\star$", "pre-enrichment"),
        (axes[0, 1], z_final, r"$Z_{\rm final}$", "final metallicity"),
    ]
    for ax, values, label, title in panels:
        good = np.isfinite(values) & (values > 0)
        ax.scatter(
            mdil[good],
            values[good],
            c=np.where(y[good], _COLORS["red"], _COLORS["blue"]),
            s=5,
            alpha=0.35,
            linewidths=0,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("M$_{\\rm dil}$ [M$_{\\odot}$]")
        ax.set_ylabel(label)
        ax.set_title(title)

    good = np.isfinite(feh)
    axes[1, 0].scatter(
        mdil[good],
        feh[good],
        c=np.where(y[good], _COLORS["red"], _COLORS["blue"]),
        s=5,
        alpha=0.35,
        linewidths=0,
    )
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlabel("M$_{\\rm dil}$ [M$_{\\odot}$]")
    axes[1, 0].set_ylabel("[Fe/H]")
    axes[1, 0].set_title("final iron abundance")

    ratio = np.divide(
        mstar, mdil, out=np.full_like(mstar, np.nan), where=mdil > 0
    )
    axes[1, 1].hist(np.log10(ratio[np.isfinite(ratio) & (ratio > 0)]), bins=40, color=_COLORS["green"])
    axes[1, 1].axvline(0.0, color=_COLORS["red"], ls="--", label="M*II = M_dil")
    axes[1, 1].set_xlabel(r"$\log_{10}(M_{*II}/M_{\rm dil})$")
    axes[1, 1].set_ylabel("count")
    axes[1, 1].legend()
    axes[1, 1].set_title("Pop II mass fraction")
    fig.tight_layout()
    _save(fig, path)


def plot_detection(arrays: dict[str, np.ndarray], path: str | Path) -> None:
    """Plot the counterfactual detection statistic against both labels."""
    d = np.asarray(arrays["D"], dtype=float)
    d2 = np.asarray(arrays["D2"], dtype=float)
    f_zp = np.asarray(arrays["f_zp"], dtype=float)
    f_fep = np.asarray(arrays["f_fep"], dtype=float)
    y = np.asarray(arrays["y"], dtype=bool)
    n_detected = np.asarray(arrays["n_detected"], dtype=float)
    finite = np.isfinite(d) & (d > 0)
    color = np.where(y, _COLORS["red"], _COLORS["blue"])

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].scatter(f_zp[finite], d[finite], c=color[finite], s=5, alpha=0.35, linewidths=0)
    axes[0, 0].set_yscale("log")
    axes[0, 0].axhline(3.0, color=_COLORS["grey"], ls="--")
    axes[0, 0].set_xlabel(r"$f_Z^P$")
    axes[0, 0].set_ylabel("D")
    axes[0, 0].set_title("metal share vs detectability")

    good = finite & (f_fep > 0)
    axes[0, 1].scatter(f_fep[good], d[good], c=color[good], s=5, alpha=0.35, linewidths=0)
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].axhline(3.0, color=_COLORS["grey"], ls="--")
    axes[0, 1].set_xlabel(r"$f_{\rm Fe}^P$")
    axes[0, 1].set_ylabel("D")
    axes[0, 1].set_title("iron share vs detectability")

    axes[1, 0].hist(np.log10(d[finite]), bins=40, color=_COLORS["green"])
    axes[1, 0].axvline(np.log10(3.0), color=_COLORS["red"], ls="--", label="k=3")
    axes[1, 0].set_xlabel(r"$\log_{10} D$")
    axes[1, 0].set_ylabel("count")
    axes[1, 0].legend()
    axes[1, 0].set_title(f"finite D: {finite.mean():.3f}")

    axes[1, 1].scatter(n_detected[finite], d[finite], c=color[finite], s=5, alpha=0.35, linewidths=0)
    axes[1, 1].set_yscale("log")
    axes[1, 1].axhline(3.0, color=_COLORS["grey"], ls="--")
    axes[1, 1].set_xlabel("number of usable features")
    axes[1, 1].set_ylabel("D")
    axes[1, 1].set_title("feature coverage vs detectability")
    fig.tight_layout()
    _save(fig, path)


def plot_limit_sweep(
    arrays: dict[str, np.ndarray],
    meta: dict[str, Any],
    path: str | Path,
) -> None:
    """Plot post-hoc sensitivity to thresholds, floors, and the Pop II cap."""
    d = np.asarray(arrays["D"], dtype=float)
    z_star = np.asarray(arrays["z_star"], dtype=float)
    feh = np.asarray(arrays["feh"], dtype=float)
    n_usable = np.asarray(arrays["ratio_mask"], dtype=bool).sum(axis=1)
    y = np.asarray(arrays["y"], dtype=bool)
    config = meta["config"]

    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    k_grid = np.linspace(0.0, 8.0, 33)
    rates = [np.mean(d > k) for k in k_grid]
    axes[0, 0].plot(k_grid, rates, color=_COLORS["blue"], lw=2)
    axes[0, 0].axvline(
        config["detect"]["k"], color=_COLORS["red"], ls="--", label="default"
    )
    axes[0, 0].set_xlabel("k")
    axes[0, 0].set_ylabel("positive fraction")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].legend()
    axes[0, 0].set_title("detection threshold")

    z_grid = np.geomspace(1e-5, 5e-2, 30)
    retained = np.asarray([np.mean(z_star <= value) for value in z_grid])
    positive = np.asarray(
        [
            y[z_star <= value].mean() if np.any(z_star <= value) else np.nan
            for value in z_grid
        ]
    )
    axes[0, 1].plot(z_grid, retained, color=_COLORS["green"], label="retained")
    axes[0, 1].plot(z_grid, positive, color=_COLORS["orange"], label="positive fraction")
    axes[0, 1].axvline(
        config["prior"]["z_max"] or 5e-2,
        color=_COLORS["red"],
        ls="--",
        label="default",
    )
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlabel(r"$Z_\star$ cap")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()
    axes[0, 1].set_title("metallicity cap")

    counts = np.arange(1, len(meta["feature_names"]) + 1)
    retained = np.asarray([np.mean(n_usable >= value) for value in counts])
    positive = np.asarray(
        [
            y[n_usable >= value].mean() if np.any(n_usable >= value) else np.nan
            for value in counts
        ]
    )
    axes[1, 0].plot(counts, retained, color=_COLORS["green"], label="retained")
    axes[1, 0].plot(counts, positive, color=_COLORS["orange"], label="positive fraction")
    axes[1, 0].axvline(
        config["detect"]["min_detected"],
        color=_COLORS["red"],
        ls="--",
        label="default",
    )
    axes[1, 0].set_xlabel("minimum usable features")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend()
    axes[1, 0].set_title("feature-count floor")

    feh_grid = np.linspace(-8.0, -1.0, 36)
    good = np.isfinite(feh)
    retained = np.asarray([np.mean(feh[good] >= value) for value in feh_grid])
    positive = np.asarray(
        [
            y[good & (feh >= value)].mean()
            if np.any(good & (feh >= value))
            else np.nan
            for value in feh_grid
        ]
    )
    axes[1, 1].plot(feh_grid, retained, color=_COLORS["green"], label="retained")
    axes[1, 1].plot(feh_grid, positive, color=_COLORS["orange"], label="positive fraction")
    axes[1, 1].set_xlabel("[Fe/H] floor")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend()
    axes[1, 1].set_title("iron-abundance floor")

    feature_names = [
        name for name in meta["feature_elements"] if name != meta["reference"]
    ]
    feature_idx = np.asarray([ELEMENT_INDEX[name] for name in feature_names])
    ref_idx = ELEMENT_INDEX[meta["reference"]]
    metal_mass = np.asarray(arrays["M_z_tot"], dtype=float)
    total_mass = np.asarray(arrays["M_tot"], dtype=float)
    values = np.asarray(arrays["features"], dtype=float)
    floor_grid = np.geomspace(1e-12, 1e-4, 25)
    retained = []
    positive = []
    for floor in floor_grid:
        valid = total_mass[:, feature_idx] / metal_mass[:, None] >= floor
        valid_ref = total_mass[:, ref_idx] / metal_mass >= floor
        mask = valid & valid_ref[:, None] & np.isfinite(values)
        keep = mask.sum(axis=1) >= config["min_valid_features"]
        retained.append(keep.mean())
        positive.append(y[keep].mean() if np.any(keep) else np.nan)
    axes[0, 2].plot(floor_grid, retained, color=_COLORS["green"], label="retained")
    axes[0, 2].plot(
        floor_grid, positive, color=_COLORS["orange"], label="positive fraction"
    )
    axes[0, 2].axvline(
        config["min_element_fraction"],
        color=_COLORS["red"],
        ls="--",
        label="default",
    )
    axes[0, 2].set_xscale("log")
    axes[0, 2].set_xlabel("element-fraction floor")
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].legend()
    axes[0, 2].set_title("feature-validity floor")

    ratio = np.asarray(arrays["mstar_ii"], dtype=float) / np.asarray(
        arrays["mdil"], dtype=float
    )
    cap_grid = np.linspace(0.05, 1.0, 20)
    retained = [np.mean(ratio <= cap) for cap in cap_grid]
    positive = [
        y[ratio <= cap].mean() if np.any(ratio <= cap) else np.nan
        for cap in cap_grid
    ]
    axes[1, 2].plot(cap_grid, retained, color=_COLORS["green"], label="retained")
    axes[1, 2].plot(
        cap_grid, positive, color=_COLORS["orange"], label="positive fraction"
    )
    axes[1, 2].axvline(
        config["prior"]["mstar_ii_max_ratio"]
        if config["prior"]["mstar_ii_max_ratio"] is not None
        else config["prior"]["mstar_total_max_ratio"] or 1.0,
        color=_COLORS["red"],
        ls="--",
        label="default",
    )
    axes[1, 2].set_xlabel(r"$M_{\star,II}/M_{\rm dil}$ cap")
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].legend()
    axes[1, 2].set_title("Pop II mass cap")
    fig.tight_layout()
    _save(fig, path)


def plot_eq13(result: dict[str, Any], path: str | Path) -> None:
    """Plot the per-element Eq. 13 residuals."""
    differences = result.get("differences", {})
    fig, ax = plt.subplots(figsize=(10, 4))
    if not differences:
        ax.text(0.5, 0.5, "no finite residuals", ha="center", va="center")
        ax.set_axis_off()
    else:
        names = list(differences)
        values = np.asarray([abs(differences[name]) for name in names])
        floor = 1e-20
        ax.bar(names, np.log10(np.maximum(values, floor)), color=_COLORS["blue"])
        ax.set_ylabel(r"$\log_{10}|{\rm ours}-{\rm Salvadori}|$")
        ax.tick_params(axis="x", rotation=60)
        ax.set_title(
            f"Eq. 13 residuals: max={result['max_abs_difference']:.2e}, "
            f"mean={result['mean_abs_difference']:.2e}"
        )
        ax.axhline(np.log10(1e-12), color=_COLORS["red"], ls="--", label="1e-12")
        ax.legend()
    fig.tight_layout()
    _save(fig, path)


def plot_ceiling(report: dict[str, Any], path: str | Path) -> None:
    """Plot the noiseless classifier and regressor cross-validation."""
    from sklearn.metrics import roc_curve

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
    y_true = np.asarray(report.get("y_true", []))
    probability = np.asarray(report.get("y_probability", []))
    nuisance = np.asarray(report.get("nuisance_probability", []))
    if len(y_true) and len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, probability)
        axes[0].plot(
            fpr,
            tpr,
            color=_COLORS["blue"],
            lw=2,
            label=f"chemistry (AUC={report['auc']:.3f})",
        )
        if len(nuisance) == len(y_true):
            fpr, tpr, _ = roc_curve(y_true, nuisance)
            axes[0].plot(
                fpr,
                tpr,
                color=_COLORS["grey"],
                lw=1.5,
                ls="--",
                label=f"nuisance (AUC={report['nuisance_auc']:.3f})",
            )
        axes[0].plot([0, 1], [0, 1], color=_COLORS["red"], lw=1, ls=":")
        axes[0].set_xlabel("false positive rate")
        axes[0].set_ylabel("true positive rate")
        axes[0].legend(loc="lower right", fontsize=8)
        axes[0].set_title("detection ROC")

    true = np.asarray(report.get("f_fep_true", []), dtype=float)
    predicted = np.asarray(report.get("f_fep_prediction", []), dtype=float)
    good = np.isfinite(true) & np.isfinite(predicted)
    if good.any():
        axes[1].hexbin(
            true[good],
            predicted[good],
            gridsize=35,
            bins="log",
            cmap="viridis",
            mincnt=1,
        )
        axes[1].plot([0, 1], [0, 1], color=_COLORS["red"], ls="--", lw=1)
        axes[1].set_xlabel(r"true $f_{\rm Fe}^{\rm P}$")
        axes[1].set_ylabel(r"predicted $f_{\rm Fe}^{\rm P}$")
        axes[1].set_title(
            f"$f_{{\\rm Fe}}^{{\\rm P}}$: $R^2$={report['f_fep_r2']:.3f}"
        )

    if len(y_true) and len(probability) == len(y_true):
        bins = np.linspace(0.0, 1.0, 31)
        axes[2].hist(
            probability[y_true == 0],
            bins=bins,
            alpha=0.6,
            color=_COLORS["orange"],
            label="y=0",
        )
        axes[2].hist(
            probability[y_true == 1],
            bins=bins,
            alpha=0.6,
            color=_COLORS["blue"],
            label="y=1",
        )
        axes[2].set_xlabel("predicted probability")
        axes[2].set_ylabel("count")
        axes[2].legend()
        axes[2].set_title("class probabilities")

    rows = report.get("auc_by_z_star", [])
    nuisance_rows = report.get("nuisance_auc_by_z_star", [])
    for data, color, marker, label in (
        (rows, _COLORS["blue"], "o", "chemistry"),
        (nuisance_rows, _COLORS["grey"], "s", "nuisance"),
    ):
        usable = [row for row in data if row.get("auc") is not None]
        if usable:
            x = [(row["low"] + row["high"]) / 2 for row in usable]
            axes[3].plot(
                x,
                [row["auc"] for row in usable],
                marker=marker,
                color=color,
                label=label,
            )
    axes[3].axhline(
        report.get("auc", np.nan), color=_COLORS["red"], ls="--", lw=1
    )
    axes[3].set_xlabel(r"$\log_{10} Z_\star$ bin centre")
    axes[3].set_ylabel("AUC")
    axes[3].set_ylim(0.4, 1.02)
    axes[3].legend(fontsize=8)
    axes[3].set_title("AUC vs metallicity")

    fig.suptitle(
        f"noiseless ceiling: n={report.get('n', 0)}, "
        f"AUC={report.get('auc', np.nan):.3f}, "
        f"nuisance AUC={report.get('nuisance_auc', np.nan):.3f}"
    )
    fig.tight_layout()
    _save(fig, path)


def plot_stratified(report: dict[str, Any], path: str | Path) -> None:
    """Plot AUC and regression scores in label and mass bins."""
    panels = (
        ("by_fzp", r"$f_Z^{\rm P}$"),
        ("by_pisn_mass", r"PISN mass [$M_\odot$]"),
        ("by_feh", "[Fe/H]"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (key, label) in zip(axes, panels):
        rows = [row for row in report.get(key, []) if row.get("n", 0) > 0]
        if not rows:
            ax.text(0.5, 0.5, "no samples", ha="center", va="center")
            ax.set_axis_off()
            continue
        x = np.asarray([row["center"] for row in rows], dtype=float)
        auc = np.asarray(
            [np.nan if row["auc"] is None else row["auc"] for row in rows],
            dtype=float,
        )
        ax.plot(x, auc, "o-", color=_COLORS["blue"], label="AUC")
        ax.set_ylim(0.4, 1.02)
        ax.set_xlabel(label)
        ax.set_ylabel("AUC", color=_COLORS["blue"])
        ax.tick_params(axis="y", labelcolor=_COLORS["blue"])

        twin = ax.twinx()
        twin.plot(
            x,
            np.clip(
                [
                    np.nan if row["f_fep_r2"] is None else row["f_fep_r2"]
                    for row in rows
                ],
                -1.0,
                1.0,
            ),
            "s--",
            color=_COLORS["orange"],
            label=r"$R^2$ $f_{\rm Fe}^{\rm P}$",
        )
        twin.plot(
            x,
            np.clip(
                [
                    np.nan if row["f_zp_r2"] is None else row["f_zp_r2"]
                    for row in rows
                ],
                -1.0,
                1.0,
            ),
            "^:",
            color=_COLORS["green"],
            label=r"$R^2$ $f_Z^{\rm P}$",
        )
        twin.axhline(0.0, color=_COLORS["grey"], lw=0.8, ls=":")
        twin.set_ylim(-1.05, 1.05)
        twin.set_ylabel(r"AUC / $R^2$ (clipped at $-1$)")
        lines = ax.get_lines() + twin.get_lines()
        ax.legend(lines, [line.get_label() for line in lines], fontsize=8, loc="best")
        ax.set_title(f"{label}: AUC and $R^2$")
    fig.suptitle("where the ceiling is lost")
    fig.tight_layout()
    _save(fig, path)


def plot_reliability(report: dict[str, Any], path: str | Path) -> None:
    """Plot chemistry and nuisance calibration curves with Brier scores."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, key, title, color in (
        (axes[0], "reliability", "chemistry", _COLORS["blue"]),
        (axes[1], "nuisance_reliability", "nuisance only", _COLORS["grey"]),
    ):
        curve = report.get(key, {})
        rows = [row for row in curve.get("bins", []) if row["count"]]
        predicted = np.asarray(
            [row["mean_predicted"] for row in rows], dtype=float
        )
        observed = np.asarray([row["observed"] for row in rows], dtype=float)
        counts = np.asarray([row["count"] for row in rows], dtype=float)
        ax.plot([0, 1], [0, 1], color=_COLORS["red"], ls="--", lw=1)
        if predicted.size:
            ax.plot(predicted, observed, "o-", color=color, lw=2)
            sizes = 18.0 + 90.0 * counts / counts.max()
            ax.scatter(predicted, observed, s=sizes, color=color, zorder=3)
        brier = curve.get("brier", float("nan"))
        ax.set_title(f"{title}: Brier={brier:.4f}")
        ax.set_xlabel("mean predicted probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("observed positive fraction")
    fig.tight_layout()
    _save(fig, path)


def plot_ceiling(report: dict[str, Any], path: str | Path) -> None:
    """Plot the noiseless classifier and regressor cross-validation."""
    from sklearn.metrics import roc_curve

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    y_true = np.asarray(report.get("y_true", []))
    probability = np.asarray(report.get("y_probability", []))
    if len(y_true) and len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, probability)
        axes[0].plot(fpr, tpr, color=_COLORS["blue"])
        axes[0].plot([0, 1], [0, 1], color=_COLORS["grey"], linestyle=":")
    else:
        axes[0].text(0.5, 0.5, "not enough classes", ha="center", va="center")
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    axes[0].set_title(f"detection ROC (AUC={report.get('auc', float('nan')):.3f})")
    axes[0].text(
        0.98,
        0.05,
        f"nuisance AUC={report.get('nuisance_auc', float('nan')):.3f}",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=_COLORS["grey"],
    )

    true = np.asarray(report.get("f_fep_true", []))
    prediction = np.asarray(report.get("f_fep_prediction", []))
    finite = np.isfinite(true) & np.isfinite(prediction)
    if finite.any():
        axes[1].hexbin(
            true[finite],
            prediction[finite],
            gridsize=35,
            extent=(0, 1, 0, 1),
            mincnt=1,
            cmap="viridis",
        )
    axes[1].plot([0, 1], [0, 1], color=_COLORS["grey"], linestyle=":")
    axes[1].set_xlabel(r"true $f_{\rm Fe}^{\rm P}$")
    axes[1].set_ylabel("prediction")
    axes[1].set_title(
        rf"$f_{{\rm Fe}}^{{\rm P}}$ R2={report.get('f_fep_r2', float('nan')):.3f}"
        rf"  MAE={report.get('f_fep_mae', float('nan')):.3f}"
    )

    if len(y_true) and len(probability):
        axes[2].hist(
            probability[y_true == 0],
            bins=20,
            range=(0, 1),
            alpha=0.7,
            label="y=0",
            color=_COLORS["blue"],
        )
        axes[2].hist(
            probability[y_true == 1],
            bins=20,
            range=(0, 1),
            alpha=0.7,
            label="y=1",
            color=_COLORS["orange"],
        )
    axes[2].set_xlabel("predicted probability")
    axes[2].set_ylabel("count")
    axes[2].set_title("class separation")
    axes[2].legend()

    rows = [
        row
        for row in report.get("auc_by_z_star", [])
        if row.get("auc") is not None
    ]
    if rows:
        centres = [0.5 * (row["low"] + row["high"]) for row in rows]
        axes[3].plot(
            centres,
            [row["auc"] for row in rows],
            marker="o",
            color=_COLORS["blue"],
            label="chemistry",
        )
        nuisance_rows = [
            row
            for row in report.get("nuisance_auc_by_z_star", [])
            if row.get("auc") is not None
        ]
        if nuisance_rows:
            nuisance_centres = [
                0.5 * (row["low"] + row["high"]) for row in nuisance_rows
            ]
            axes[3].plot(
                nuisance_centres,
                [row["auc"] for row in nuisance_rows],
                marker="s",
                color=_COLORS["grey"],
                label="nuisance only",
            )
        axes[3].axhline(
            report.get("auc", float("nan")),
            color=_COLORS["red"],
            ls="--",
            label="all",
        )
        axes[3].set_xlabel(r"$\log_{10} Z_\star$")
        axes[3].set_ylabel("AUC")
        axes[3].set_ylim(0.5, 1.0)
        axes[3].legend()
        axes[3].set_title("AUC within metallicity bins")
    else:
        axes[3].text(0.5, 0.5, "no stratified bins", ha="center", va="center")
        axes[3].set_axis_off()

    fig.suptitle(
        f"noiseless chemistry ceiling  n={report.get('n', 0)}"
        f"  folds={report.get('n_splits', 0)}"
    )
    _save(fig, path)


def plot_noise_sweep(report: dict[str, Any], path: str | Path) -> None:
    """Plot detection and regression scores against element-level noise."""
    sigmas = np.asarray(report.get("sigmas", []), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(
        sigmas,
        report.get("auc", []),
        marker="o",
        color=_COLORS["blue"],
    )
    axes[0].axhline(0.5, color=_COLORS["grey"], ls=":")
    axes[0].set_xlabel(r"$\sigma$ (dex per element)")
    axes[0].set_ylabel("AUC")
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_title("detection")
    axes[1].plot(
        sigmas,
        report.get("f_fep_r2", []),
        marker="o",
        label=r"$f_{\rm Fe}^{\rm P}$",
        color=_COLORS["orange"],
    )
    axes[1].plot(
        sigmas,
        report.get("f_zp_r2", []),
        marker="o",
        label=r"$f_Z^{\rm P}$",
        color=_COLORS["green"],
    )
    axes[1].axhline(0.0, color=_COLORS["grey"], ls=":")
    axes[1].set_xlabel(r"$\sigma$ (dex per element)")
    axes[1].set_ylabel(r"$R^2$")
    axes[1].set_ylim(-0.1, 1.0)
    axes[1].legend()
    axes[1].set_title("regression")
    fig.tight_layout()
    _save(fig, path)


def plot_ablation(summary: dict[str, dict[str, Any]], path: str | Path) -> None:
    """Plot acceptance, label rate, and cut fractions across ablations."""
    names = list(summary)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    width = 0.38
    positions = np.arange(len(names))
    axes[0].bar(positions - width / 2, [summary[name]["acceptance"] for name in names], width, label="accepted", color=_COLORS["green"])
    axes[0].bar(positions + width / 2, [summary[name]["y_rate"] for name in names], width, label="positive", color=_COLORS["orange"])
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(names, rotation=20)
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].set_title("acceptance and label rate")

    bottom = np.zeros(len(names))
    for key, color in (("r1", _COLORS["red"]), ("r2", _COLORS["orange"]), ("r3", _COLORS["purple"])):
        values = np.asarray([summary[name][key] for name in names])
        axes[1].bar(names, values, bottom=bottom, label=key.upper(), color=color)
        bottom += values
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=20)
    axes[1].set_ylabel("fraction of draws")
    axes[1].legend()
    axes[1].set_title("cut fractions")
    fig.tight_layout()
    _save(fig, path)


def plot_transfer(report: dict[str, Any], path: str | Path) -> None:
    """Plot train-on-base / test-on-variant AUC and Brier matrices."""
    chemistry = report["variants"]["chemistry"]
    nuisance = report["variants"]["nuisance"]
    names = list(chemistry)
    fig, axes = plt.subplots(1, 2, figsize=(max(10, 1.4 * len(names)), 4.5))
    for ax, metric, title, vmin, vmax, cmap in (
        (axes[0], "auc", "AUC", 0.5, 1.0, "viridis"),
        (axes[1], "brier", "Brier score", 0.0, 0.25, "viridis_r"),
    ):
        values = np.asarray(
            [
                [chemistry[name][metric] for name in names],
                [nuisance[name][metric] for name in names],
            ]
        )
        image = ax.imshow(values, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["chemistry", "nuisance"])
        for row in range(2):
            for column, name in enumerate(names):
                value = values[row, column]
                if np.isfinite(value):
                    ax.text(
                        column,
                        row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color="white" if metric == "auc" else "black",
                        fontsize=8,
                    )
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("train on base prior, test on each ablation prior", fontsize=9)
    fig.tight_layout()
    _save(fig, path)


def plot_all(
    arrays: dict[str, np.ndarray],
    meta: dict[str, Any],
    outdir: str | Path,
) -> dict[str, str]:
    """Write the full diagnostic plot set and the JSON report."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report = diagnose(arrays, meta)
    plots: list[tuple[str, Callable[[], None]]] = [
        ("ledger", lambda: plot_ledger(meta, outdir / "ledger.png")),
        ("labels", lambda: plot_labels(arrays, outdir / "labels.png")),
        ("class_balance", lambda: plot_class_balance(arrays, outdir / "class_balance.png")),
        ("features", lambda: plot_features(arrays, meta, outdir / "features.png")),
        ("mixture_cells", lambda: plot_mixture_cells(report, outdir / "mixture_cells.png")),
        ("pisn_mass", lambda: plot_pisn_mass(arrays, meta, outdir / "pisn_mass.png")),
        ("event_masses", lambda: plot_event_masses(arrays, meta, outdir / "event_masses.png")),
        ("clock", lambda: plot_clock(arrays, outdir / "clock.png")),
        ("dilution", lambda: plot_dilution(arrays, outdir / "dilution.png")),
        ("detection", lambda: plot_detection(arrays, outdir / "detection.png")),
        ("limit_sweep", lambda: plot_limit_sweep(arrays, meta, outdir / "limit_sweep.png")),
    ]
    paths: dict[str, str] = {}
    for name, function in plots:
        function()
        paths[name] = str(outdir / f"{name}.png")
    (outdir / "diagnostics.json").write_text(json.dumps(report, indent=2, default=str))
    return paths
