"""Mock-only figures that explain the potluck generator.

The figures use the noiseless mock arrays, never the Li+ observation layer.
They are meant for a talk: priors, clocks, yields, abundance trends, the
Fe-independent ratios, and the counterfactual label.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr

from ..chronology import popiii_lifetime
from ..constants import (
    DEFAULT_FEATURE_ELEMENTS,
    ELEMENT_INDEX,
    METALS,
    SOLAR_MASS_FRACTION,
)
from ..io import load_dataset
from ..yields.pisn import load_pisn_table
from ..yields.popii import PopIITable
from ..yields.popiii import PopIIITable

SOURCE_LABELS = ("PISN", "Pop III", "Pop II")
SOURCE_COLORS = ("tab:purple", "tab:green", "tab:blue")
DISPLAY_ELEMENTS = ("C", "Na", "Mg", "Si", "Ca", "Cr", "Mn", "Co", "Ni", "Zn")
TREND_ELEMENTS = ("Na", "Mg", "Si", "Ca", "Co", "Zn")
RATIO_PAIRS = (("Na", "Mg"), ("Co", "Ni"), ("Mn", "Cr"), ("Zn", "Mg"))
R4_WINDOW = (-3.8, -1.7)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.2,
            "axes.axisbelow": True,
            "font.size": 9,
        }
    )


def _load(dataset: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if isinstance(dataset, (str, Path)):
        return load_dataset(dataset)
    arrays, meta = dataset
    return arrays, meta


def _feature_names(meta: dict[str, Any]) -> tuple[str, ...]:
    names = meta.get("feature_names")
    if names:
        return tuple(names)
    return tuple(
        f"{element}/Fe"
        for element in DEFAULT_FEATURE_ELEMENTS
        if element != "Fe"
    )


def _feature_column(meta: dict[str, Any], element: str) -> int:
    return _feature_names(meta).index(f"{element}/Fe")


def _ratio(arrays: dict[str, np.ndarray], numerator: str, denominator: str) -> np.ndarray:
    loga = arrays["log_abundance"].astype(float)
    return loga[:, ELEMENT_INDEX[numerator]] - loga[:, ELEMENT_INDEX[denominator]]


def _xfe(vector: np.ndarray) -> np.ndarray:
    solar = np.asarray([SOLAR_MASS_FRACTION[element] for element in ELEMENT_INDEX])
    with np.errstate(divide="ignore", invalid="ignore"):
        loga = np.log10(np.asarray(vector, dtype=float) / solar)
    return loga - loga[ELEMENT_INDEX["Fe"]]


def _source_shares(arrays: dict[str, np.ndarray]) -> np.ndarray:
    indices = [ELEMENT_INDEX[element] for element in METALS]
    masses = arrays["matrix"][:, :3, :][:, :, indices].sum(axis=-1).astype(float)
    total = masses.sum(axis=1)
    shares = np.full_like(masses, np.nan)
    np.divide(masses, total[:, None], out=shares, where=total[:, None] > 0.0)
    return shares


def _finite_mask(*values: np.ndarray) -> np.ndarray:
    mask = np.ones(values[0].size, dtype=bool)
    for value in values:
        mask &= np.isfinite(value)
    return mask


def _binned_stats(
    x: np.ndarray,
    y: np.ndarray,
    bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return bin centres, median, 16th, 84th, and counts."""
    centers = 0.5 * (bins[:-1] + bins[1:])
    median = np.full(centers.size, np.nan)
    low = np.full(centers.size, np.nan)
    high = np.full(centers.size, np.nan)
    counts = np.zeros(centers.size, dtype=int)
    for index in range(centers.size):
        mask = (x >= bins[index]) & (x < bins[index + 1])
        values = y[mask]
        counts[index] = values.size
        if values.size:
            median[index] = np.median(values)
            low[index], high[index] = np.percentile(values, [16, 84])
    return centers, median, low, high, counts


@lru_cache(maxsize=1)
def _popiii_table() -> PopIIITable:
    return PopIIITable.from_raw()


@lru_cache(maxsize=1)
def _pisn_table():
    return load_pisn_table("HW2002")


@lru_cache(maxsize=1)
def _popii_table() -> PopIITable:
    return PopIITable(
        source_name="Limongi18",
        z_grid=(3.236e-5, 3.236e-4, 3.236e-3, 1.345e-2),
        z_grid_size=4,
        t_grid_size=16,
        t_min_myr=3.0,
        t_max_myr=40.0,
        cache_dir="data/potluck/cache",
    )


def plot_priors(arrays: dict[str, np.ndarray], path: Path) -> Path:
    """Show the input distributions and the resulting PISN fraction."""
    panels = (
        ("n_pisn", "number of PISN", False, np.arange(-0.5, 4.0, 1.0)),
        ("n_iii", "Pop III events", True, 40),
        ("mdil", r"$M_{\rm dil}$ [$M_\odot$]", True, 40),
        ("mstar_ii", r"$M_{\star,\rm II}$ [$M_\odot$]", True, 40),
        ("delta_myr", r"$\Delta$ [Myr]", True, 40),
        ("t_myr", r"$t$ [Myr]", True, 40),
        ("f_zp", r"$f_Z^P$", False, 40),
        ("z_star", r"$Z_\star$", True, 40),
    )
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    for axis, (key, label, log_x, bins) in zip(axes.ravel(), panels):
        values = arrays[key].astype(float)
        values = values[np.isfinite(values)]
        if log_x:
            values = values[values > 0.0]
            axis.hist(np.log10(values), bins=bins, color="tab:blue", alpha=0.8)
            axis.set_xlabel(rf"$\log_{{10}}$ {label}")
            median = np.median(values)
            low, high = np.percentile(values, [16, 84])
            axis.axvline(np.log10(median), color="black", ls="--", lw=1)
        else:
            axis.hist(values, bins=bins, color="tab:blue", alpha=0.8)
            axis.set_xlabel(label)
            median = np.median(values)
            low, high = np.percentile(values, [16, 84])
            axis.axvline(median, color="black", ls="--", lw=1)
        axis.set_title(
            f"{label}\nmedian {median:.2g}  [16-84%: {low:.2g}, {high:.2g}]",
            fontsize=9,
        )
        axis.set_ylabel("mocks")
    fig.suptitle("Potluck priors: what the generator samples", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _draw_clock(axis: plt.Axes) -> None:
    tau_iii = 6.46
    tau_pisn = 2.2
    delta = 20.0
    horizon = 35.0

    axis.set_xlim(-3.0, 112.0)
    axis.set_ylim(-1.9, 2.5)
    axis.axis("off")

    axis.add_patch(
        plt.Rectangle((0.0, -0.15), delta, 2.1, color="tab:blue", alpha=0.07, lw=0)
    )
    axis.add_patch(
        plt.Rectangle(
            (delta, -0.15), horizon - delta, 2.1, color="tab:orange", alpha=0.11, lw=0
        )
    )
    axis.plot([delta, delta], [-0.15, 1.95], color="0.45", ls=":", lw=1.0)
    axis.plot([horizon, horizon], [-0.15, 1.95], color="0.45", ls=":", lw=1.0)

    axis.text(
        delta / 2.0,
        2.1,
        r"pre-enriched gas: $\tau \leq \Delta$",
        color="tab:blue",
        ha="center",
        fontsize=9,
    )
    axis.text(
        horizon + 2.0,
        2.1,
        r"final cloud: $\Delta < \tau \leq \Delta + t$",
        color="tab:orange",
        ha="left",
        fontsize=9,
    )

    axis.annotate(
        "",
        xy=(104.0, -0.55),
        xytext=(0.0, -0.55),
        arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "0.35"},
    )
    axis.text(105.0, -0.55, "time", va="center", fontsize=9, color="0.35")

    axis.plot([0.0, tau_iii], [1.5, 1.5], color="tab:green", lw=2)
    axis.scatter([0.0], [1.5], marker="o", s=90, color="tab:green", zorder=5)
    axis.text(
        tau_iii + 1.5,
        1.5,
        r"Pop III star: $\tau_{\rm III}$",
        va="center",
        color="tab:green",
        fontsize=9,
    )

    axis.plot([0.0, tau_pisn], [0.95, 0.95], color="tab:purple", lw=2)
    axis.scatter([tau_pisn], [0.95], marker="*", s=190, color="tab:purple", zorder=5)
    axis.text(
        tau_pisn + 1.5,
        0.95,
        r"PISN: $\tau_{\rm PISN}$",
        va="center",
        color="tab:purple",
        fontsize=9,
    )

    axis.scatter([delta], [0.4], marker="s", s=90, color="tab:blue", zorder=5)
    axis.text(
        delta + 1.5,
        0.4,
        r"Pop II forms: $\Delta$",
        va="center",
        color="tab:blue",
        fontsize=9,
    )

    axis.scatter([horizon], [-0.05], marker="D", s=80, color="black", zorder=5)
    axis.text(
        horizon + 1.5,
        -0.05,
        r"observed star: $\Delta + t$",
        va="center",
        fontsize=9,
    )

    axis.text(
        52.0,
        -1.0,
        "A source reaches the pre-enrichment gas only if its lifetime is shorter than $\\Delta$;\n"
        "it reaches the final cloud if its lifetime is shorter than $\\Delta + t$.",
        ha="center",
        va="top",
        fontsize=9,
    )


def plot_timeline(arrays: dict[str, np.ndarray], path: Path) -> Path:
    """Explain the clocks, example histories, and the source metal budget."""
    fig = plt.figure(figsize=(14, 9))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.15))
    clock = fig.add_subplot(grid[0, :])
    _draw_clock(clock)

    axis = fig.add_subplot(grid[1, 0])
    n = arrays["n_pisn"].size
    order = np.lexsort((arrays["n_iii"], arrays["n_pisn"]))
    chosen = order[np.linspace(0, n - 1, min(8, n)).astype(int)]
    group_masses = _popiii_table().group_masses
    group_tau = np.asarray([float(popiii_lifetime(mass)) for mass in group_masses])
    for row, index in enumerate(chosen):
        delta = float(arrays["delta_myr"][index])
        horizon = delta + float(arrays["t_myr"][index])
        axis.plot([0.0, horizon], [row, row], color="0.75", lw=1.0, zorder=1)

        axis.scatter(
            [horizon],
            [row],
            marker="D",
            s=35,
            color="black",
            zorder=6,
        )

        counts = arrays["iii_group_counts"][index]
        reachable = np.any((counts > 0) & (group_tau <= horizon))
        axis.scatter(
            [0.0],
            [row],
            marker="o",
            s=30 + 6 * np.log1p(float(arrays["n_iii"][index])),
            facecolor="tab:green" if reachable else "white",
            edgecolor="tab:green",
            linewidth=1.0,
            zorder=5,
        )

        masses = arrays["pisn_masses"][index]
        for mass in masses[masses > 0.0]:
            tau = float(popiii_lifetime(mass))
            axis.scatter(
                [tau],
                [row],
                marker="*",
                s=70,
                facecolor="tab:purple" if tau <= horizon else "white",
                edgecolor="tab:purple",
                linewidth=0.9,
                zorder=5,
            )

        axis.scatter(
            [delta],
            [row],
            marker="s",
            s=35,
            facecolor="tab:blue" if arrays["popii_n"][index] > 0 else "white",
            edgecolor="tab:blue",
            linewidth=1.0,
            zorder=5,
        )
        axis.text(
            horizon + 1.0,
            row,
            f"$n_{{\\rm PISN}}$={int(arrays['n_pisn'][index])}, "
            f"$n_{{\\rm III}}$={int(arrays['n_iii'][index])}",
            va="center",
            fontsize=7,
        )

    axis.set_xlim(-2.0, 130.0)
    axis.set_ylim(-0.8, len(chosen) - 0.2)
    axis.set_yticks([])
    axis.set_xlabel("time since Pop III formation [Myr]")
    axis.set_title("Eight example enrichment histories")
    handles = [
        Line2D([], [], marker="o", color="tab:green", ls="", label="Pop III"),
        Line2D([], [], marker="*", color="tab:purple", ls="", label="PISN"),
        Line2D([], [], marker="s", color="tab:blue", ls="", label="Pop II"),
        Line2D([], [], marker="D", color="black", ls="", label="star"),
        Patch(facecolor="0.8", edgecolor="0.8", label="filled = contributes"),
    ]
    axis.legend(handles=handles, loc="lower right", fontsize=7, ncol=2)

    axis = fig.add_subplot(grid[1, 1])
    shares = _source_shares(arrays)
    groups = np.clip(arrays["n_pisn"], 0, 3)
    labels = ["0", "1", "2", "3+"]
    bottom = np.zeros(4)
    for source, color, label in zip(range(3), SOURCE_COLORS, SOURCE_LABELS):
        medians = np.asarray(
            [
                np.nanmean(shares[groups == group, source])
                if np.any(groups == group)
                else 0.0
                for group in range(4)
            ]
        )
        medians = np.nan_to_num(medians)
        axis.bar(labels, medians, bottom=bottom, color=color, label=label, alpha=0.85)
        bottom += medians
    counts = [int(np.sum(groups == group)) for group in range(4)]
    axis.set_xticks(range(4))
    axis.set_xticklabels([f"{label}\n(n={count:,})" for label, count in zip(labels, counts)])
    axis.set_xlabel(r"number of PISN events, $n_{\rm PISN}$")
    axis.set_ylabel("mean share of feature-metal mass")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("PISN dominates the budget only when it is present")
    axis.legend(fontsize=8, loc="upper right")

    fig.suptitle("Potluck clocks and event bookkeeping", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _terciles(values: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    low, high = np.quantile(values[mask], [1.0 / 3.0, 2.0 / 3.0])
    return [
        mask & (values <= low),
        mask & (values > low) & (values <= high),
        mask & (values > high),
    ]


def _curve_by_group(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    groups: list[np.ndarray],
    labels: tuple[str, ...],
    colors: tuple[str, ...],
    x_log: bool = True,
) -> None:
    finite = _finite_mask(x, y)
    if x_log:
        finite &= x > 0.0
    x_finite = np.log10(x[finite]) if x_log else x[finite]
    y_finite = y[finite]
    edges = np.quantile(x_finite, np.linspace(0.0, 1.0, 13))
    edges = np.unique(edges)
    if edges.size < 2:
        return
    for group, label, color in zip(groups, labels, colors):
        group_finite = group[finite]
        centers, median, low, high, counts = _binned_stats(
            x_finite[group_finite], y_finite[group_finite], edges
        )
        good = counts >= 5
        axis.plot(centers[good], median[good], color=color, lw=1.8, label=label)
        axis.fill_between(
            centers[good],
            low[good],
            high[good],
            color=color,
            alpha=0.12,
            lw=0,
        )


def plot_metallicity(arrays: dict[str, np.ndarray], path: Path) -> Path:
    """Show how dilution, burst mass, and pre-enrichment set [Fe/H]."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    feh = arrays["feh"].astype(float)
    mdil = arrays["mdil"].astype(float)
    mstar = arrays["mstar_ii"].astype(float)

    axis = axes[0, 0]
    finite = _finite_mask(feh, mdil, mstar) & (mdil > 0.0) & (mstar > 0.0)
    axis.hexbin(
        np.log10(mdil[finite]),
        feh[finite],
        gridsize=45,
        cmap="Blues",
        mincnt=1,
        bins="log",
        alpha=0.85,
    )
    _curve_by_group(
        axis,
        mdil,
        feh,
        _terciles(mstar, finite),
        ("low $M_{\\star,\\rm II}$", "mid", "high"),
        ("tab:orange", "tab:red", "tab:brown"),
    )
    axis.set_xlabel(r"$\log_{10} M_{\rm dil}$ [$M_\odot$]")
    axis.set_ylabel("[Fe/H]")
    axis.set_title("More dilution lowers [Fe/H]")
    axis.legend(fontsize=7, title="burst-mass tercile")

    axis = axes[0, 1]
    axis.hexbin(
        np.log10(mstar[finite]),
        feh[finite],
        gridsize=45,
        cmap="Greens",
        mincnt=1,
        bins="log",
        alpha=0.85,
    )
    _curve_by_group(
        axis,
        mstar,
        feh,
        _terciles(mdil, finite),
        ("low $M_{\\rm dil}$", "mid", "high"),
        ("tab:purple", "tab:red", "tab:brown"),
    )
    axis.set_xlabel(r"$\log_{10} M_{\star,\rm II}$ [$M_\odot$]")
    axis.set_ylabel("[Fe/H]")
    axis.set_title("A larger Pop II burst raises [Fe/H]")
    axis.legend(fontsize=7, title="dilution tercile")

    axis = axes[1, 0]
    n_pisn = np.clip(arrays["n_pisn"].astype(int), 0, 3)
    positions = [group for group in range(4) if np.any(n_pisn == group)]
    data = [feh[(n_pisn == group) & np.isfinite(feh)] for group in positions]
    boxes = axis.boxplot(
        data,
        positions=positions,
        widths=0.6,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "black"},
    )
    for patch in boxes["boxes"]:
        patch.set_facecolor("tab:blue")
        patch.set_alpha(0.45)
    for position, values in zip(positions, data):
        axis.text(
            position,
            np.median(values) + 0.06,
            f"{np.median(values):.2f}",
            ha="center",
            fontsize=8,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [f"{group}\n(n={values.size:,})" for group, values in zip(positions, data)]
    )
    axis.set_xlabel(r"number of PISN events, $n_{\rm PISN}$")
    axis.set_ylabel("[Fe/H]")
    axis.set_title("PISN events barely move [Fe/H]")
    axis.set_ylim(
        np.percentile(feh[np.isfinite(feh)], 1) - 0.2,
        np.percentile(feh[np.isfinite(feh)], 99) + 0.2,
    )

    axis = axes[1, 1]
    axis.hist(feh[np.isfinite(feh)], bins=45, color="0.65", alpha=0.9)
    axis.axvspan(*R4_WINDOW, color="tab:orange", alpha=0.15, label="R4 window")
    median = np.nanmedian(feh)
    axis.axvline(median, color="black", ls="--", lw=1.2, label=f"median {median:.2f}")
    axis.set_xlabel("[Fe/H]")
    axis.set_ylabel("mocks")
    axis.set_title("The adopted [Fe/H] window")
    axis.legend(fontsize=8)

    fig.suptitle("What sets the metallicity scale", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _fingerprint_lines(axis: plt.Axes, vectors: list[np.ndarray], labels: list[str]) -> None:
    positions = np.arange(len(DISPLAY_ELEMENTS))
    for vector, label in zip(vectors, labels):
        values = _xfe(vector)
        axis.plot(
            positions,
            values[[ELEMENT_INDEX[element] for element in DISPLAY_ELEMENTS]],
            marker="o",
            ms=4,
            lw=1.2,
            label=label,
        )
    axis.axhline(0.0, color="0.5", lw=0.8)
    axis.set_xticks(positions)
    axis.set_xticklabels(DISPLAY_ELEMENTS, rotation=45)
    axis.set_ylabel("[X/Fe]")
    axis.set_ylim(-3.6, 2.8)
    axis.legend(fontsize=7)


def plot_yields(path: Path, popii: PopIITable | None = None) -> Path:
    """Show the three yield sources and the PISN mass dependence."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axis = axes[0, 0]
    pisn = _pisn_table()
    pisn_masses = [220.0, 260.0]
    _fingerprint_lines(
        axis,
        [pisn.at_mass(mass) for mass in pisn_masses],
        [f"{mass:.0f} $M_\\odot$" for mass in pisn_masses],
    )
    axis.set_title("PISN: high Si/Ca/Cr, low Na/Co/Ni/Zn")

    axis = axes[0, 1]
    popiii = _popiii_table()
    wanted = ((13.0, 0.5), (25.0, 1.0), (40.0, 30.0), (100.0, 60.0))
    vectors = []
    labels = []
    for mass, energy in wanted:
        if (mass, energy) not in popiii.combo_keys:
            continue
        combo = popiii.combo_keys.index((mass, energy))
        vectors.append(popiii.vectors[popiii.combo_indices[combo]].mean(axis=0))
        labels.append(f"{mass:.0f} $M_\\odot$, E={energy:g}")
    _fingerprint_lines(axis, vectors, labels)
    axis.set_title("Pop III: comparatively flat fingerprints")

    axis = axes[1, 0]
    popii_table = popii if popii is not None else _popii_table()
    metallicities = [1.0e-3, 4.0e-3, 8.0e-3, 2.0e-2]
    _fingerprint_lines(
        axis,
        [popii_table.integrated(z, 10.0) for z in metallicities],
        [f"Z={z:.0e}" for z in metallicities],
    )
    axis.set_title("Pop II: alpha-enhanced, Na-rich")

    axis = axes[1, 1]
    mass_grid = np.linspace(140.0, 270.0, 131)
    vectors = pisn.at_masses(mass_grid)
    elements = ("Si", "Ca", "Fe", "Cr", "Mn", "Co", "Ni", "Zn")
    colors = (
        "tab:blue",
        "tab:orange",
        "black",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
    )
    for element, color in zip(elements, colors):
        values = np.log10(np.maximum(vectors[:, ELEMENT_INDEX[element]], 1.0e-16))
        axis.plot(
            mass_grid,
            values,
            lw=2.2 if element == "Fe" else 1.7,
            ls="-" if element != "Fe" else "--",
            color=color,
            label=element,
        )
    axis.axvline(148.0, color="0.5", ls=":", lw=1.0)
    axis.text(
        152.0,
        -15.4,
        "above $\\sim$148 $M_\\odot$:\nFe-peak ejecta turn on",
        fontsize=8,
        color="0.35",
        va="bottom",
    )
    axis.set_xlabel(r"PISN progenitor mass [$M_\odot$]")
    axis.set_ylabel(r"$\log_{10} M_X^{\rm ej}$ [$M_\odot$]")
    axis.set_ylim(-16.0, 2.0)
    axis.set_title("Heavy elements need massive PISN progenitors")
    axis.legend(fontsize=7, ncol=2, loc="lower right")

    fig.suptitle("Where the abundance pattern comes from", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_trends(
    arrays: dict[str, np.ndarray],
    meta: dict[str, Any],
    path: Path,
    seed: int = 0,
) -> Path:
    """Show [X/Fe] versus [Fe/H], coloured by the PISN mass fraction."""
    feh = arrays["feh"].astype(float)
    f_zp = arrays["f_zp"].astype(float)
    n = feh.size
    rng = np.random.default_rng(seed)
    sample = rng.choice(n, size=min(n, 20000), replace=False)
    fig, axes = plt.subplots(
        2, 3, figsize=(14, 8.5), sharex=True, layout="constrained"
    )
    for axis, element in zip(axes.ravel(), TREND_ELEMENTS):
        column = _feature_column(meta, element)
        values = arrays["features"][:, column].astype(float)
        mask = _finite_mask(feh, values, f_zp)
        points = sample[mask[sample]]
        points = points[np.argsort(f_zp[points])]
        scatter = axis.scatter(
            feh[points],
            values[points],
            c=f_zp[points],
            cmap="viridis",
            s=6,
            alpha=0.4,
            linewidths=0.0,
            rasterized=True,
            vmin=0.0,
            vmax=1.0,
        )
        bins = np.quantile(feh[mask], np.linspace(0.0, 1.0, 11))
        bins = np.unique(bins)
        for label, group_mask, color in (
            (r"$f_Z^P < 0.5$", f_zp < 0.5, "black"),
            (r"$f_Z^P \geq 0.5$", f_zp >= 0.5, "tab:red"),
        ):
            centers, median, low, high, counts = _binned_stats(
                feh[mask & group_mask], values[mask & group_mask], bins
            )
            good = counts >= 10
            axis.plot(centers[good], median[good], color=color, lw=2.0, label=label)
        axis.axhline(0.0, color="0.5", lw=0.8)
        axis.set_title(f"[{element}/Fe]")
        if axis in axes[:, 0]:
            axis.set_ylabel("[X/Fe]")
        if axis in axes[-1, :]:
            axis.set_xlabel("[Fe/H]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, loc="lower left", fontsize=8, framealpha=0.9)
    fig.colorbar(scatter, ax=axes, shrink=0.9, label=r"$f_Z^P$")
    fig.suptitle("PISN-dominated mocks are offset in the abundance plane", fontsize=13)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ratios(
    arrays: dict[str, np.ndarray],
    path: Path,
    seed: int = 1,
) -> Path:
    """Show which Fe-independent ratios carry PISN information."""
    f_zp = arrays["f_zp"].astype(float)
    n = f_zp.size
    rng = np.random.default_rng(seed)
    sample = rng.choice(n, size=min(n, 8000), replace=False)
    bins = np.linspace(0.0, 1.0, 11)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, (numerator, denominator) in zip(axes.ravel(), RATIO_PAIRS):
        values = _ratio(arrays, numerator, denominator)
        mask = _finite_mask(f_zp, values)
        axis.scatter(
            f_zp[sample][mask[sample]],
            values[sample][mask[sample]],
            s=3,
            color="0.65",
            alpha=0.2,
            rasterized=True,
        )
        centers, median, low, high, counts = _binned_stats(
            f_zp[mask], values[mask], bins
        )
        good = counts >= 30
        axis.plot(centers[good], median[good], color="tab:blue", lw=2.0)
        axis.fill_between(centers[good], low[good], high[good], color="tab:blue", alpha=0.18)
        axis.axvline(0.5, color="black", ls="--", lw=1.0)
        rho = spearmanr(f_zp[mask], values[mask]).statistic
        axis.text(
            0.03,
            0.95,
            f"Spearman $\\rho$ = {rho:+.2f}",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
        )
        axis.set_title(f"[{numerator}/{denominator}]")
        axis.set_xlabel(r"$f_Z^P$")
        axis.set_ylabel(f"[{numerator}/{denominator}]")
        y_low, y_high = np.nanpercentile(values[mask], [1, 99])
        axis.set_ylim(y_low - 0.2, y_high + 0.2)
    fig.suptitle(
        "Three Fe-independent ratios track the PISN fraction; Mn/Cr does not",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.012,
        "grey points: individual mocks; blue line: median; shaded band: 16-84%",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 1.0))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_label(
    arrays: dict[str, np.ndarray],
    path: Path,
    seed: int = 2,
) -> Path:
    """Show the label, its event driver, and the source budget by class."""
    f_zp = arrays["f_zp"].astype(float)
    n_pisn = arrays["n_pisn"].astype(int)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    axis = axes[0, 0]
    positive = f_zp >= 0.5
    axis.hist(f_zp[~positive], bins=30, color="tab:gray", alpha=0.8, label=r"$f_Z^P < 0.5$")
    axis.hist(f_zp[positive], bins=30, color="tab:red", alpha=0.8, label=r"$f_Z^P \geq 0.5$")
    axis.axvline(0.5, color="black", ls="--", lw=1.2)
    axis.set_xlabel(r"$f_Z^P$")
    axis.set_ylabel("mocks")
    axis.set_title(f"Label: {positive.mean():.1%} positive")
    axis.legend(fontsize=8)

    axis = axes[0, 1]
    groups = np.clip(n_pisn, 0, 3)
    categories = []
    data = []
    for group in range(4):
        values = f_zp[groups == group]
        if values.size:
            categories.append(f"{group}+" if group == 3 else str(group))
            data.append(values)
    axis.boxplot(data, tick_labels=categories, showfliers=False, patch_artist=True)
    axis.set_xlabel(r"$n_{\rm PISN}$")
    axis.set_ylabel(r"$f_Z^P$")
    axis.set_title("The label is driven by PISN events")

    axis = axes[1, 0]
    na_mg = _ratio(arrays, "Na", "Mg")
    co_ni = _ratio(arrays, "Co", "Ni")
    mask = _finite_mask(na_mg, co_ni, f_zp)
    rng = np.random.default_rng(seed + 2)
    sample = rng.choice(np.flatnonzero(mask), size=min(mask.sum(), 8000), replace=False)
    for values, color, label in (
        (~positive, "tab:gray", r"$f_Z^P < 0.5$"),
        (positive, "tab:red", r"$f_Z^P \geq 0.5$"),
    ):
        points = sample[values[sample]]
        axis.scatter(
            na_mg[points],
            co_ni[points],
            s=4,
            alpha=0.25,
            color=color,
            label=label,
            rasterized=True,
        )
        axis.scatter(
            [np.median(na_mg[points])],
            [np.median(co_ni[points])],
            marker="*",
            s=220,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            zorder=5,
        )
    axis.set_xlabel("[Na/Mg]")
    axis.set_ylabel("[Co/Ni]")
    axis.set_title("Two ratios separate the bulk, not every mock")
    axis.legend(fontsize=8)
    axis.set_xlim(np.nanpercentile(na_mg[mask], [1, 99]))
    axis.set_ylim(np.nanpercentile(co_ni[mask], [1, 99]))

    axis = axes[1, 1]
    shares = _source_shares(arrays)
    bottom = np.zeros(2)
    for source, color, label in zip(range(3), SOURCE_COLORS, SOURCE_LABELS):
        medians = [
            np.nanmean(shares[~positive, source]),
            np.nanmean(shares[positive, source]),
        ]
        medians = np.nan_to_num(medians)
        axis.bar([0, 1], medians, bottom=bottom, color=color, label=label, alpha=0.85)
        bottom += medians
    axis.set_xticks([0, 1])
    axis.set_xticklabels([r"$f_Z^P < 0.5$", r"$f_Z^P \geq 0.5$"])
    axis.set_ylabel("mean share of feature-metal mass")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Positive mocks are PISN-dominated")
    axis.legend(fontsize=8, loc="upper right")

    fig.suptitle("The counterfactual label and its physical driver", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


POPII_TABLE_ZMIN = 3.236e-5


def plot_zstar(arrays: dict[str, np.ndarray], path: Path) -> Path:
    """Compare the sampled metallicities with the Pop II yield grid."""
    z_star = arrays["z_star"].astype(float)
    z_final = arrays["z_final"].astype(float)
    mask = _finite_mask(z_star, z_final) & (z_star > 0.0) & (z_final > 0.0)
    log_z = np.log10(z_star[mask])
    log_zf = np.log10(z_final[mask])
    edge = np.log10(POPII_TABLE_ZMIN)
    floor = np.log10(1.0e-6)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axis = axes[0]
    axis.hist(log_z, bins=60, color="tab:blue", alpha=0.8)
    axis.axvline(
        floor, color="tab:red", ls="--", lw=1.2, label=r"$z_{\rm cr} = 10^{-6}$"
    )
    axis.axvline(
        edge, color="black", ls="--", lw=1.2, label="Limongi18 first node $3.24\\times10^{-5}$"
    )
    axis.set_xlabel(r"$\log_{10} z_\star$")
    axis.set_ylabel("mocks")
    axis.set_title("Pre-enrichment metallicity")
    axis.legend(fontsize=8)
    below = float(np.mean(z_star[mask] < POPII_TABLE_ZMIN))
    axis.text(
        0.03,
        0.95,
        f"{below:.2%} below $3.24\\times10^{{-5}}$",
        transform=axis.transAxes,
        va="top",
        fontsize=9,
    )

    axis = axes[1]
    axis.hexbin(log_z, log_zf, gridsize=45, cmap="Blues", mincnt=1, bins="log")
    axis.axvline(edge, color="black", ls="--", lw=1.2)
    axis.text(
        edge - 0.1,
        -5.5,
        "one Pop II table\nbelow $3.24\\times10^{-5}$",
        ha="right",
        fontsize=8,
        color="0.35",
    )
    axis.set_xlim(-6.2, -2.0)
    axis.set_xlabel(r"$\log_{10} z_\star$")
    axis.set_ylabel(r"$\log_{10} z_{\rm final}$")
    axis.set_title("Final metallicity after the Pop II burst")

    fig.suptitle("Metallicity sampling relative to the Pop II yield grid", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def run_explanation_plots(
    dataset: Any,
    outdir: str | Path = "src/potluck/tests/v7",
    seed: int = 0,
    popii: PopIITable | None = None,
) -> dict[str, Any]:
    """Write the mock-only explanation figures and a small summary."""
    _style()
    arrays, meta = _load(dataset)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "priors": plot_priors(arrays, outdir / "fig1_priors.png"),
        "timeline": plot_timeline(arrays, outdir / "fig2_timeline.png"),
        "metallicity": plot_metallicity(arrays, outdir / "fig3_metallicity.png"),
        "yields": plot_yields(outdir / "fig4_yields.png", popii=popii),
        "trends": plot_trends(arrays, meta, outdir / "fig5_trends.png", seed=seed),
        "ratios": plot_ratios(arrays, outdir / "fig6_ratios.png", seed=seed + 1),
        "label": plot_label(arrays, outdir / "fig7_label.png", seed=seed + 2),
        "zstar": plot_zstar(arrays, outdir / "fig8_zstar.png"),
    }

    f_zp = arrays["f_zp"].astype(float)
    shares = _source_shares(arrays)
    summary: dict[str, Any] = {
        "n_mocks": int(f_zp.size),
        "positive_fraction": float(np.mean(f_zp >= 0.5)),
        "feature_names": list(_feature_names(meta)),
        "figures": [path.name for path in paths.values()],
        "source_share_median": {
            label: float(np.nanmedian(shares[:, index]))
            for index, label in enumerate(SOURCE_LABELS)
        },
        "ratio_spearman": {
            f"{numerator}/{denominator}": float(
                spearmanr(
                    f_zp,
                    _ratio(arrays, numerator, denominator),
                    nan_policy="omit",
                ).statistic
            )
            for numerator, denominator in RATIO_PAIRS
        },
        "z_star_below_table_floor_fraction": float(
            np.mean(arrays["z_star"] < POPII_TABLE_ZMIN)
        ),
    }
    with (outdir / "explanation_plots.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main(argv: list[str] | None = None) -> int:
    """Run the explanation figures from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="dataset .npz with noiseless mock arrays")
    parser.add_argument("--outdir", default="src/potluck/tests/v7")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    summary = run_explanation_plots(args.dataset, outdir=args.outdir, seed=args.seed)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
