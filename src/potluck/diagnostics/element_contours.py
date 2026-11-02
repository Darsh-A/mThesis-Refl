"""Per-element [X/Fe] versus [Fe/H] contours coloured by the PISN share.

Every element gets its own figure. The noiseless mocks are binned in the
([Fe/H], [X/Fe]) plane and each bin is coloured by its median ``f_zp``, the
fraction of the metal mass that comes from PISN ejecta. The Li+ stars are
overlaid as white circles with their catalogue error bars; upper limits are
drawn as white triangles. Additional ratio panels (currently [Mg/Si]) use the
same layout with the ratio on the vertical axis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import binned_statistic_2d

from ..constants import LI_FEATURE_ELEMENTS
from ..io import load_dataset
from ..observation import DETECTED, LIMIT, load_li_records

MIN_COUNT = 12
GRID = (44, 24)
LEVELS = np.linspace(0.0, 1.0, 11)
DENSITY_FRACTIONS = (0.05, 0.2, 0.5)
RATIO_PAIRS = (("Mg", "Si"),)


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


def _range(values: np.ndarray, low: float = 1.0, high: float = 99.0) -> tuple[float, float]:
    lo, hi = np.percentile(values, [low, high])
    pad = 0.05 * (hi - lo)
    return float(lo - pad), float(hi + pad)


def _binned_median(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, grid: tuple[int, int], xlim, ylim
):
    stat, xe, ye, _ = binned_statistic_2d(
        x, y, z, statistic="median", bins=list(grid), range=[xlim, ylim]
    )
    count, _, _, _ = binned_statistic_2d(
        x, y, z, statistic="count", bins=list(grid), range=[xlim, ylim]
    )
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    return stat, count, xc, yc


def _plot_panel(
    label: str,
    ylabel: str,
    mock_feh: np.ndarray,
    mock_y: np.ndarray,
    mock_fzp: np.ndarray,
    obs: dict[str, Any] | None,
    path: Path,
    min_count: int,
    grid: tuple[int, int],
) -> dict[str, Any]:
    """Draw one contour panel and return its summary entry."""
    finite = np.isfinite(mock_feh) & np.isfinite(mock_y) & np.isfinite(mock_fzp)
    feh = mock_feh[finite]
    y_values = mock_y[finite]
    fzp = mock_fzp[finite]

    n_detected = int(obs["detected"].sum()) if obs is not None else 0
    n_limits = int(obs["limits"].sum()) if obs is not None else 0

    xlim = _range(feh)
    ylim = _range(y_values)
    obs_xlim = obs_ylim = None
    if n_detected:
        obs_xlim = _range(obs["feh"][obs["detected"]])
        obs_ylim = _range(obs["y"][obs["detected"]])
        xlim = (min(xlim[0], obs_xlim[0]), max(xlim[1], obs_xlim[1]))
        ylim = (min(ylim[0], obs_ylim[0]), max(ylim[1], obs_ylim[1]))

    stat, count, xc, yc = _binned_median(feh, y_values, fzp, grid, xlim, ylim)
    valid = count >= min_count
    if valid.any():
        dx, dy = xc[1] - xc[0], yc[1] - yc[0]
        xc_valid = xc[valid.any(axis=1)]
        yc_valid = yc[valid.any(axis=0)]
        xlim = (float(xc_valid.min() - dx), float(xc_valid.max() + dx))
        ylim = (float(yc_valid.min() - dy), float(yc_valid.max() + dy))
        if obs_xlim is not None:
            xlim = (min(xlim[0], obs_xlim[0]), max(xlim[1], obs_xlim[1]))
            ylim = (min(ylim[0], obs_ylim[0]), max(ylim[1], obs_ylim[1]))
        stat, count, xc, yc = _binned_median(feh, y_values, fzp, grid, xlim, ylim)
        valid = count >= min_count
    masked = np.ma.masked_where(~valid, stat)

    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    if valid.sum() >= 4:
        filled = ax.contourf(
            xc,
            yc,
            masked.T,
            levels=LEVELS,
            cmap="viridis",
            extend="both",
            antialiased=False,
        )
        peak = float(np.nanmax(count))
        levels = sorted({f * peak for f in DENSITY_FRACTIONS})
        ax.contour(
            xc,
            yc,
            np.ma.masked_where(~valid, count).T,
            levels=levels,
            colors="0.35",
            linewidths=0.8,
            alpha=0.85,
        )
    else:
        filled = ax.scatter(
            feh,
            y_values,
            c=fzp,
            s=12,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            edgecolors="none",
        )
    colorbar = fig.colorbar(filled, ax=ax, pad=0.02)
    colorbar.set_label(r"$f_Z^P$ (PISN metal-mass fraction)")

    if n_detected:
        ax.errorbar(
            obs["feh"][obs["detected"]],
            obs["y"][obs["detected"]],
            xerr=obs["xerr"][obs["detected"]],
            yerr=obs["yerr"][obs["detected"]],
            fmt="o",
            ms=5,
            mfc="white",
            mec="black",
            mew=0.8,
            ecolor="0.2",
            elinewidth=0.7,
            capsize=2,
            alpha=0.7,
            zorder=6,
        )
    if n_limits:
        ax.plot(
            obs["feh"][obs["limits"]],
            obs["y"][obs["limits"]],
            marker="v",
            ls="",
            ms=6,
            mfc="white",
            mec="black",
            mew=0.8,
            zorder=6,
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axhline(0.0, color="white", lw=0.8, alpha=0.6)
    ax.set_xlabel("[Fe/H]")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel.strip('[]')} versus [Fe/H]", pad=18)
    ax.text(
        0.0,
        1.01,
        f"{feh.size:,} mocks  ·  median $f_Z^P$ = {np.median(fzp):.2f}",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=8,
        color="0.35",
    )

    handles = []
    if n_detected:
        handles.append(
            Line2D([], [], marker="o", ls="", mfc="white", mec="black", mew=0.8,
                   ms=5, label=f"Li+ ({n_detected})")
        )
    if n_limits:
        handles.append(
            Line2D([], [], marker="v", ls="", mfc="white", mec="black", mew=0.8,
                   ms=6, label=f"upper limit ({n_limits})")
        )
    if handles:
        ax.legend(handles=handles, loc="lower right", framealpha=0.25, fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    ranks = np.argsort(np.argsort(feh)), np.argsort(np.argsort(y_values))
    return {
        "label": label,
        "n_mocks": int(feh.size),
        "n_bins": int(valid.sum()),
        "fzp_median": float(np.median(fzp)),
        "fzp_p16": float(np.percentile(fzp, 16)),
        "fzp_p84": float(np.percentile(fzp, 84)),
        "li_detected": n_detected,
        "li_limits": n_limits,
        "x_lim": list(xlim),
        "y_lim": list(ylim),
        "feh_xfe_spearman": float(np.corrcoef(ranks[0], ranks[1])[0, 1]),
    }


def _element_obs(element: str, records: Any) -> dict[str, Any] | None:
    """Return the Li+ overlay for one element, or None if it is unmeasured."""
    if element not in LI_FEATURE_ELEMENTS:
        return None
    column = LI_FEATURE_ELEMENTS.index(element)
    flags = records.flags[:, column]
    present = np.isfinite(records.feh) & np.isfinite(records.xfe[:, column])
    return {
        "feh": records.feh,
        "y": records.xfe[:, column],
        "xerr": records.e_feh,
        "yerr": records.sigma[:, column],
        "detected": present & (flags == DETECTED),
        "limits": present & (flags == LIMIT),
    }


def _ratio_obs(numerator: str, denominator: str, records: Any) -> dict[str, Any] | None:
    """Return the Li+ overlay for a ratio, requiring both elements to be measured."""
    if numerator not in LI_FEATURE_ELEMENTS or denominator not in LI_FEATURE_ELEMENTS:
        return None
    first = LI_FEATURE_ELEMENTS.index(numerator)
    second = LI_FEATURE_ELEMENTS.index(denominator)
    flags = records.flags[:, [first, second]]
    present = (
        np.isfinite(records.feh)
        & np.isfinite(records.xfe[:, first])
        & np.isfinite(records.xfe[:, second])
    )
    both = present & (flags == DETECTED).all(axis=1)
    limit_rows = present & ~both
    return {
        "feh": records.feh,
        "y": records.xfe[:, first] - records.xfe[:, second],
        "xerr": records.e_feh,
        "yerr": np.hypot(records.sigma[:, first], records.sigma[:, second]),
        "detected": both,
        "limits": limit_rows,
    }


def _plot_element(
    element: str,
    mock_feh: np.ndarray,
    mock_xfe: np.ndarray,
    mock_fzp: np.ndarray,
    records: Any,
    path: Path,
    min_count: int,
    grid: tuple[int, int],
) -> dict[str, Any]:
    return _plot_panel(
        element,
        f"[{element}/Fe]",
        mock_feh,
        mock_xfe,
        mock_fzp,
        _element_obs(element, records),
        path,
        min_count,
        grid,
    )


def _plot_ratio(
    numerator: str,
    denominator: str,
    mock_feh: np.ndarray,
    mock_y: np.ndarray,
    mock_fzp: np.ndarray,
    records: Any,
    path: Path,
    min_count: int,
    grid: tuple[int, int],
) -> dict[str, Any]:
    label = f"{numerator}/{denominator}"
    return _plot_panel(
        label,
        f"[{label}]",
        mock_feh,
        mock_y,
        mock_fzp,
        _ratio_obs(numerator, denominator, records),
        path,
        min_count,
        grid,
    )


def run_element_contours(
    dataset: Any,
    outdir: str | Path = "src/potluck/tests/v14",
    min_count: int = MIN_COUNT,
    grid: tuple[int, int] = GRID,
) -> dict[str, Any]:
    """Write one contour figure per element plus a JSON summary."""
    _style()
    arrays, meta = load_dataset(dataset) if isinstance(dataset, (str, Path)) else dataset
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    names = list(meta["feature_names"])
    elements = [name.split("/")[0] for name in names]
    records = load_li_records()
    feh = arrays["feh"].astype(float)
    fzp = arrays["f_zp"].astype(float)

    summary: dict[str, Any] = {"n_mocks": int(feh.size), "elements": {}, "ratios": {}}
    for element, name in zip(elements, names):
        path = outdir / f"{element}_Fe.png"
        summary["elements"][element] = _plot_element(
            element,
            feh,
            arrays["features"][:, names.index(name)].astype(float),
            fzp,
            records,
            path,
            min_count,
            grid,
        )
        summary["elements"][element]["figure"] = path.name

    for numerator, denominator in RATIO_PAIRS:
        if numerator not in elements or denominator not in elements:
            continue
        label = f"{numerator}/{denominator}"
        path = outdir / f"{numerator}_{denominator}.png"
        mock_y = (
            arrays["features"][:, names.index(f"{numerator}/Fe")].astype(float)
            - arrays["features"][:, names.index(f"{denominator}/Fe")].astype(float)
        )
        summary["ratios"][label] = _plot_ratio(
            numerator,
            denominator,
            feh,
            mock_y,
            fzp,
            records,
            path,
            min_count,
            grid,
        )
        summary["ratios"][label]["figure"] = path.name

    with (outdir / "element_contours.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main(argv: list[str] | None = None) -> int:
    """Run the per-element contour plots from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--outdir", default="src/potluck/tests/v14")
    parser.add_argument("--min-count", type=int, default=MIN_COUNT)
    args = parser.parse_args(argv)
    summary = run_element_contours(args.dataset, outdir=args.outdir, min_count=args.min_count)
    counts = {e: v["n_bins"] for e, v in summary["elements"].items()}
    counts.update({e: v["n_bins"] for e, v in summary["ratios"].items()})
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
