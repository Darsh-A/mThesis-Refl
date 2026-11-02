"""Per-element audit: mock ratios, yield fingerprints, and Li+ stars.

For every feature element the module compares the noiseless mock distribution
with the observed Li+ distribution (where available) and with the pure-source
yield fingerprints. It writes one figure and one JSON file.
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

from ..constants import (
    DEFAULT_FEATURE_ELEMENTS,
    ELEMENT_INDEX,
    LI_FEATURE_ELEMENTS,
    SOLAR_MASS_FRACTION,
)
from ..io import load_dataset
from ..observation import load_li_records
from ..yields.pisn import load_pisn_table
from ..yields.popii import PopIITable
from ..yields.popiii import PopIIITable

AUDIT_ELEMENTS = tuple(e for e in DEFAULT_FEATURE_ELEMENTS if e != "Fe")
OBS_ELEMENTS = tuple(e for e in LI_FEATURE_ELEMENTS if e != "Fe")
SOURCE_ORDER = ("PISN", "Pop III", "Pop II")
SOURCE_MARKERS = {"PISN": "*", "Pop III": "o", "Pop II": "s"}
SOURCE_COLORS = {"PISN": "tab:purple", "Pop III": "tab:green", "Pop II": "tab:blue"}
LIMONGI_Z_NODES = (3.236e-5, 3.236e-4, 3.236e-3, 1.345e-2)


def _xfe(vector: np.ndarray) -> np.ndarray:
    solar = np.asarray([SOLAR_MASS_FRACTION[e] for e in ELEMENT_INDEX])
    with np.errstate(divide="ignore", invalid="ignore"):
        loga = np.log10(np.asarray(vector, dtype=float) / solar)
    return loga - loga[ELEMENT_INDEX["Fe"]]


def _stats(values: np.ndarray) -> dict[str, float] | None:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    low, median, high = np.percentile(values, [16, 50, 84])
    return {
        "n": int(values.size),
        "median": float(median),
        "p16": float(low),
        "p84": float(high),
        "spread": float(high - low),
    }


def _mock_stats(arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> dict[str, dict]:
    names = list(meta["feature_names"])
    return {
        element: _stats(arrays["features"][:, names.index(f"{element}/Fe")])
        for element in AUDIT_ELEMENTS
    }


def _obs_stats() -> dict[str, dict | None]:
    records = load_li_records()
    out: dict[str, dict | None] = {}
    for element in AUDIT_ELEMENTS:
        if element not in OBS_ELEMENTS:
            out[element] = None
            continue
        column = LI_FEATURE_ELEMENTS.index(element)
        values = records.xfe[:, column]
        valid = records.flags[:, column].astype(bool)
        out[element] = _stats(values[valid])
    return out


def _yield_fingerprints() -> dict[str, dict[str, float]]:
    pisn = load_pisn_table("HW2002")
    popiii = PopIIITable.from_raw()
    popii = PopIITable(
        source_name="Limongi18",
        z_grid=LIMONGI_Z_NODES,
        z_grid_size=len(LIMONGI_Z_NODES),
        t_grid_size=8,
        t_min_myr=3.0,
        t_max_myr=40.0,
        cache_dir="data/potluck/cache",
    )
    vectors = {
        "PISN": _xfe(pisn.at_mass(220.0)),
        "Pop III": _xfe(popiii.vectors.mean(axis=0)),
        "Pop II": _xfe(popii.integrated(3.236e-3, 10.0)),
    }
    return {
        element: {
            source: float(vectors[source][ELEMENT_INDEX[element]])
            for source in SOURCE_ORDER
        }
        for element in AUDIT_ELEMENTS
    }


def _offsets(mock: dict[str, dict], obs: dict[str, dict | None]) -> dict[str, dict]:
    out = {}
    for element in AUDIT_ELEMENTS:
        observed = obs[element]
        if observed is None:
            out[element] = None
            continue
        out[element] = {
            "mock_minus_obs": mock[element]["median"] - observed["median"],
            "mock_spread": mock[element]["spread"],
            "obs_spread": observed["spread"],
        }
    return out


def _plot(
    mock: dict[str, dict],
    obs: dict[str, dict | None],
    yields: dict[str, dict[str, float]],
    path: Path,
) -> None:
    positions = np.arange(len(AUDIT_ELEMENTS))[::-1]
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(13.5, 0.33 * len(AUDIT_ELEMENTS) + 3.0), gridspec_kw={"width_ratios": (2.1, 1.0)}
    )

    for position, element in zip(positions, AUDIT_ELEMENTS):
        values = mock[element]
        left.plot(
            [values["p16"], values["p84"]],
            [position, position],
            color="0.55",
            lw=6,
            alpha=0.55,
            solid_capstyle="butt",
        )
        left.plot(
            [values["median"]],
            [position],
            marker="|",
            ms=14,
            mew=2.0,
            color="black",
        )
        if obs[element] is not None:
            observed = obs[element]
            left.plot(
                [observed["p16"], observed["p84"]],
                [position, position],
                color="tab:red",
                lw=2.0,
                alpha=0.85,
            )
            left.plot(
                [observed["median"]],
                [position],
                marker="D",
                ms=5,
                color="tab:red",
            )
        for source in SOURCE_ORDER:
            left.plot(
                [yields[element][source]],
                [position],
                marker=SOURCE_MARKERS[source],
                ms=7 if source != "PISN" else 11,
                color=SOURCE_COLORS[source],
                ls="",
                alpha=0.9,
            )

    left.set_yticks(positions)
    left.set_yticklabels(AUDIT_ELEMENTS)
    left.axvline(0.0, color="0.3", lw=0.8)
    left.set_xlabel("[X/Fe]")
    left.set_title("Mock (grey bar), Li+ (red), source yields (markers)")
    left.set_xlim(-3.4, 2.0)
    left.grid(axis="x", alpha=0.2)

    handles = [
        plt.Line2D([], [], color="0.55", lw=6, alpha=0.55, label="mock 16-84%"),
        plt.Line2D([], [], color="black", marker="|", ls="", ms=12, mew=2, label="mock median"),
        plt.Line2D([], [], color="tab:red", marker="D", ls="", ms=5, label="Li+ median"),
        plt.Line2D([], [], color="tab:red", lw=2, label="Li+ 16-84%"),
    ]
    handles += [
        plt.Line2D(
            [],
            [],
            color=SOURCE_COLORS[source],
            marker=SOURCE_MARKERS[source],
            ls="",
            ms=8,
            label=source,
        )
        for source in SOURCE_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    paired = [e for e in AUDIT_ELEMENTS if obs[e] is not None]
    pair_positions = np.arange(len(paired))[::-1]
    for position, element in zip(pair_positions, paired):
        delta = mock[element]["median"] - obs[element]["median"]
        right.barh(position, delta, color="tab:blue", alpha=0.75, height=0.6)
        right.text(
            delta + (0.04 if delta >= 0 else -0.04),
            position,
            f"{delta:+.2f}",
            va="center",
            ha="left" if delta >= 0 else "right",
            fontsize=8,
        )
    right.set_yticks(pair_positions)
    right.set_yticklabels(paired)
    right.axvline(0.0, color="0.3", lw=0.8)
    right.set_xlabel("median mock - median Li+  [dex]")
    right.set_title("Median offset, mock minus Li+")
    span = max(abs(v) for v in (mock[e]["median"] - obs[e]["median"] for e in paired))
    right.set_xlim(-span - 0.22, span + 0.22)
    right.grid(axis="x", alpha=0.2)

    fig.suptitle("Per-element abundance audit", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_element_audit(
    dataset: Any,
    outdir: str | Path = "src/potluck/tests/v10",
) -> dict[str, Any]:
    """Compare mock, yield, and Li+ abundance ratios element by element."""
    arrays, meta = load_dataset(dataset) if isinstance(dataset, (str, Path)) else dataset
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mock = _mock_stats(arrays, meta)
    obs = _obs_stats()
    yields = _yield_fingerprints()
    offsets = _offsets(mock, obs)

    _plot(mock, obs, yields, outdir / "element_audit.png")

    paired = {e: v["mock_minus_obs"] for e, v in offsets.items() if v is not None}
    largest = max(paired, key=lambda e: abs(paired[e]))
    summary = {
        "n_mocks": int(arrays["features"].shape[0]),
        "elements": {
            element: {
                "mock": mock[element],
                "obs": obs[element],
                "yields": yields[element],
                "offset": offsets[element],
            }
            for element in AUDIT_ELEMENTS
        },
        "offsets_mock_minus_obs": paired,
        "largest_offset_element": largest,
        "largest_offset_dex": float(paired[largest]),
        "mean_abs_offset_dex": float(np.mean(np.abs(list(paired.values())))),
        "observed_elements": list(paired),
    }
    with (outdir / "element_audit.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main(argv: list[str] | None = None) -> int:
    """Run the per-element audit from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--outdir", default="src/potluck/tests/v10")
    args = parser.parse_args(argv)
    print(json.dumps(run_element_audit(args.dataset, outdir=args.outdir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
