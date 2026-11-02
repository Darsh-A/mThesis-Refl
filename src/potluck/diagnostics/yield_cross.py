"""Matched-prior cross of Pop II and PISN yield tables."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..calibration import support_gate
from ..config import ModelConfig
from ..dataset import generate_dataset
from ..observation import load_li_records
from .support_decomposition import (
    _classifier_metrics,
    _ks_metrics,
    _real_arrays,
    _two_sample_auc,
)

POPII_TABLES = ("Nomoto13", "Limongi18", "WW95")
PISN_TABLES = ("HW2002", "Takahashi-MR", "Takahashi-NM", "Takahashi-NR")
RATIO_ELEMENTS = ("C", "Na", "Mg", "Si", "Ca", "Ti", "Cr", "Mn", "Co", "Ni", "Zn")


def yield_variants(config: ModelConfig) -> dict[str, ModelConfig]:
    """Return Pop II and PISN table variants at the same prior."""
    variants: dict[str, ModelConfig] = {}
    for table in POPII_TABLES:
        variants[f"popii_{table}"] = replace(
            config,
            yields=replace(config.yields, popii_table=table),
        )
    for table in PISN_TABLES:
        variants[f"pisn_{table}"] = replace(
            config,
            yields=replace(config.yields, pisn_table=table),
        )
    return variants


def _median_offsets(
    records: object,
    arrays: dict[str, np.ndarray],
) -> dict[str, dict[str, float | None]]:
    """Return real and mock medians for each Li+ ratio element."""
    real, _, _ = _real_arrays(records)
    mock = arrays["obs_features"]
    result: dict[str, dict[str, float | None]] = {}
    for column, element in enumerate(RATIO_ELEMENTS):
        real_values = real[:, column]
        mock_values = mock[:, column]
        real_values = real_values[np.isfinite(real_values)]
        mock_values = mock_values[np.isfinite(mock_values)]
        real_median = float(np.median(real_values)) if real_values.size else None
        mock_median = float(np.median(mock_values)) if mock_values.size else None
        offset = (
            real_median - mock_median
            if real_median is not None and mock_median is not None
            else None
        )
        result[element] = {
            "real_median": real_median,
            "mock_median": mock_median,
            "offset": offset,
        }
    return result


def _support_metrics(
    records: object,
    arrays: dict[str, np.ndarray],
    seed: int,
) -> dict[str, Any]:
    """Return support, distribution, and classifier metrics for one pool."""
    distance, threshold = support_gate(records, arrays, seed=seed)
    in_model = np.isfinite(distance) & (distance <= threshold)
    classifier = _classifier_metrics(arrays)
    return {
        "n": int(arrays["y_f_05"].size),
        "threshold": float(threshold),
        "in_model": int(in_model.sum()),
        "in_model_fraction": float(in_model.mean()),
        "median_distance": float(np.nanmedian(distance)),
        "p95_distance": float(np.nanpercentile(distance, 95)),
        "ks": _ks_metrics(records, arrays),
        "two_sample_auc": _two_sample_auc(records, arrays, seed=seed),
        "classifier_auc": classifier["auc"],
        "fpr_at_tpr_90": classifier["fpr_at_tpr_90"],
        "positive_rate": float(
            np.average(arrays["y_f_05"], weights=arrays["importance_weight"])
        ),
        "medians": _median_offsets(records, arrays),
    }


def run_yield_cross(
    config: ModelConfig | None = None,
    n_samples: int = 20000,
    seed: int = 0,
    outdir: str | Path = "src/potluck/tests/v4",
    verbose: bool = False,
) -> dict[str, Any]:
    """Generate one pool per yield table and compare Li+ support."""
    config = ModelConfig() if config is None else config
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    records = load_li_records()
    summary: dict[str, Any] = {}
    for name, variant in yield_variants(config).items():
        if verbose:
            print(f"generating {name} ({n_samples} samples)")
        arrays, meta = generate_dataset(
            variant,
            n_samples=n_samples,
            seed=seed,
            verbose=verbose,
        )
        metrics = _support_metrics(records, arrays, seed)
        metrics["popii_table"] = variant.yields.popii_table
        metrics["pisn_table"] = variant.yields.pisn_table
        metrics["ledger"] = meta["ledger"]
        summary[name] = metrics
    with (outdir / "yield_cross.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    _plot_cross(summary, outdir / "yield_cross.png")
    return summary


def _plot_cross(summary: dict[str, Any], path: Path) -> Path:
    """Plot support, Ti mismatch, and two-sample separation per table."""
    names = list(summary)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()

    axes[0].bar(names, [summary[name]["in_model_fraction"] for name in names], color="tab:blue")
    axes[0].set_ylabel("Li+ in-model fraction")
    axes[0].set_ylim(0.0, 1.0)

    axes[1].bar(
        names,
        [summary[name]["ks"].get("Ti", {}).get("statistic") or 0.0 for name in names],
        color="tab:red",
    )
    axes[1].set_ylabel("Ti KS distance")
    axes[1].set_ylim(0.0, 1.0)

    axes[2].bar(
        names,
        [summary[name]["medians"].get("Ti", {}).get("offset") or 0.0 for name in names],
        color="tab:purple",
    )
    axes[2].axhline(0.0, color="black", ls="--", lw=1)
    axes[2].set_ylabel("real - mock median [Ti/Fe] (dex)")

    axes[3].bar(
        names,
        [summary[name]["two_sample_auc"] or 0.0 for name in names],
        color="tab:orange",
    )
    axes[3].axhline(0.5, color="black", ls="--", lw=1)
    axes[3].set_ylabel("mock-vs-real AUC")
    axes[3].set_ylim(0.0, 1.0)

    for axis in axes:
        axis.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    """Run the yield-table cross from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20000, help="accepted mocks per table")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--outdir",
        default="src/potluck/tests/v4",
        help="directory for JSON and PNG output",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    summary = run_yield_cross(
        n_samples=args.n,
        seed=args.seed,
        outdir=args.outdir,
        verbose=args.verbose,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
