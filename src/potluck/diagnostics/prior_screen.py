"""Space-filling screen of the prior ranges against Li+ support."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..calibration import active_core_elements
from ..config import ModelConfig
from ..dataset import generate_dataset
from ..observation import load_li_records
from ..yields.popiii import PopIIITable
from .yield_cross import _support_metrics

FACTOR_LEVELS: dict[str, tuple[Any, ...]] = {
    "pisn_mass": (
        (140.0, 270.0),
        (140.0, 300.0),
        (160.0, 260.0),
        (180.0, 260.0),
        (200.0, 300.0),
    ),
    "mdil": (
        (2.0e5, 1.0e9),
        (1.0e6, 1.0e9),
        (2.0e5, 1.0e8),
        (5.0e6, 1.0e9),
        (2.0e5, 5.0e7),
    ),
    "z_cr": (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, None),
    "lambda_iii": (
        (0.1, 1000.0),
        (0.1, 100.0),
        (1.0, 1000.0),
        (0.01, 1000.0),
        (0.3, 300.0),
    ),
    "popiii": ("uniform", "low_mass", "high_mass", "high_energy", "row"),
}


def _level_label(value: Any) -> str:
    """Return a compact string label for one factor level."""
    if value is None:
        return "none"
    if isinstance(value, tuple):
        return "(" + ", ".join(_level_label(item) for item in value) + ")"
    if isinstance(value, float):
        return f"{value:.0e}" if abs(value) < 1.0e-2 else f"{value:g}"
    return str(value)


@lru_cache(maxsize=1)
def _popiii_weights() -> dict[str, tuple[str, tuple[float, ...] | None]]:
    """Return sampling mode and weights for each Pop III level."""
    table = PopIIITable.from_raw()
    masses = table.group_masses
    energies = np.asarray([key[1] for key in table.combo_keys], dtype=float)
    return {
        "uniform": ("combo", None),
        "low_mass": ("combo", tuple((1.0 / masses).tolist())),
        "high_mass": ("combo", tuple(masses.tolist())),
        "high_energy": ("combo", tuple(energies.tolist())),
        "row": ("row", None),
    }


def design_points(
    n_points: int = 25,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Return a discrete Latin-hypercube design plus a baseline point."""
    if n_points < 1:
        raise ValueError("n_points must be positive")
    rng = np.random.default_rng(seed)
    permutations = {
        factor: rng.permutation(n_points) for factor in FACTOR_LEVELS
    }
    points: list[dict[str, Any]] = []
    for index in range(n_points):
        levels: dict[str, Any] = {}
        for factor, values in FACTOR_LEVELS.items():
            position = int(permutations[factor][index]) % len(values)
            levels[factor] = values[position]
        points.append({"label": f"p{index:02d}", "levels": levels})
    points.append(
        {
            "label": "baseline",
            "levels": {
                factor: values[0] for factor, values in FACTOR_LEVELS.items()
            },
        }
    )
    return points


def _variant(config: ModelConfig, levels: dict[str, Any]) -> ModelConfig:
    """Return a configuration with the design levels applied."""
    sampling, weights = _popiii_weights()[levels["popiii"]]
    prior = replace(
        config.prior,
        pisn_mass_range=tuple(levels["pisn_mass"]),
        mdil_range=tuple(levels["mdil"]),
        z_cr=levels["z_cr"],
        lambda_iii_range=tuple(levels["lambda_iii"]),
        popiii_sampling=sampling,
        popiii_mass_energy_weights=weights,
    )
    runtime = replace(
        config.runtime,
        max_draw_factor=max(config.runtime.max_draw_factor, 20),
    )
    return replace(config, prior=prior, runtime=runtime)


def _main_effects(points: list[dict[str, Any]]) -> dict[str, Any]:
    """Return mean in-model fraction for each level of each factor."""
    effects: dict[str, Any] = {}
    for factor, values in FACTOR_LEVELS.items():
        table: dict[str, Any] = {}
        for level in values:
            selected = [
                point["metrics"]["in_model_fraction"]
                for point in points
                if point.get("metrics") is not None
                and point["levels"][factor] == level
            ]
            table[_level_label(level)] = {
                "n": len(selected),
                "mean_in_model": float(np.mean(selected)) if selected else None,
                "values": selected,
            }
        effects[factor] = table
    return effects


def _settings(levels: dict[str, Any]) -> dict[str, Any]:
    """Return a compact, JSON-friendly view of one design point."""
    return {
        "pisn_mass_range": list(levels["pisn_mass"]),
        "mdil_range": list(levels["mdil"]),
        "z_cr": levels["z_cr"],
        "lambda_iii_range": list(levels["lambda_iii"]),
        "popiii": levels["popiii"],
    }


def run_prior_screen(
    config: ModelConfig | None = None,
    n_samples: int = 3000,
    n_points: int = 25,
    seed: int = 0,
    outdir: str | Path = "src/potluck/tests/v6",
    verbose: bool = False,
) -> dict[str, Any]:
    """Generate one pool per design point and compare Li+ support."""
    config = ModelConfig() if config is None else config
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    records = load_li_records()
    points = design_points(n_points=n_points, seed=seed)
    rows: list[dict[str, Any]] = []
    for index, point in enumerate(points):
        if verbose:
            print(f"generating {point['label']} ({n_samples} samples)")
        variant = _variant(config, point["levels"])
        row: dict[str, Any] = {
            "label": point["label"],
            "levels": point["levels"],
            "settings": _settings(point["levels"]),
        }
        try:
            arrays, meta = generate_dataset(
                variant,
                n_samples=n_samples,
                seed=seed * 1000 + index,
                verbose=verbose,
            )
        except (RuntimeError, ValueError) as error:
            row["metrics"] = None
            row["error"] = str(error)
            rows.append(row)
            continue
        metrics = _support_metrics(records, arrays, seed=seed)
        metrics["ledger"] = meta["ledger"]
        metrics["core_elements"] = list(active_core_elements(arrays))
        row["metrics"] = metrics
        rows.append(row)

    valid = [row for row in rows if row.get("metrics") is not None]
    if not valid:
        raise RuntimeError("no design point completed")
    baseline = next(row for row in rows if row["label"] == "baseline")
    best = max(
        valid,
        key=lambda row: row["metrics"]["in_model_fraction"],
    )
    summary = {
        "n_samples": n_samples,
        "n_points": len(points),
        "n_failed": len(rows) - len(valid),
        "excluded_elements": list(config.observation.exclude_elements),
        "baseline": baseline,
        "best": best,
        "effects": _main_effects(rows),
        "points": rows,
    }
    with (outdir / "prior_screen.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    _plot_screen(summary, outdir / "prior_screen.png")
    return summary


def _confirm_metrics(
    config: ModelConfig,
    records: object,
    n_samples: int,
    seed: int,
    verbose: bool,
) -> dict[str, Any]:
    """Generate one confirmation pool and score it."""
    arrays, meta = generate_dataset(
        config,
        n_samples=n_samples,
        seed=seed,
        verbose=verbose,
    )
    metrics = _support_metrics(records, arrays, seed=seed)
    metrics["ledger"] = meta["ledger"]
    metrics["core_elements"] = list(active_core_elements(arrays))
    return metrics


def _aggregate_confirmation(
    rows: list[dict[str, Any]],
    labels: tuple[str, ...],
) -> dict[str, Any]:
    """Summarise repeated runs of each confirmed design point."""
    result: dict[str, Any] = {}
    for label in labels:
        runs = [
            row
            for row in rows
            if row["label"] == label and row.get("metrics") is not None
        ]
        if not runs:
            result[label] = {"n_runs": 0}
            continue
        values = [row["metrics"]["in_model_fraction"] for row in runs]

        def mean_of(name: str) -> float | None:
            values = [
                row["metrics"][name]
                for row in runs
                if row["metrics"].get(name) is not None
            ]
            return float(np.mean(values)) if values else None

        def mean_of_ks(element: str) -> float | None:
            values = [
                row["metrics"]["ks"][element]["statistic"]
                for row in runs
                if row["metrics"]["ks"][element]["statistic"] is not None
            ]
            return float(np.mean(values)) if values else None

        result[label] = {
            "n_runs": len(runs),
            "in_model_values": values,
            "in_model_mean": float(np.mean(values)),
            "in_model_min": float(np.min(values)),
            "in_model_max": float(np.max(values)),
            "threshold_mean": mean_of("threshold"),
            "classifier_auc_mean": mean_of("classifier_auc"),
            "fpr_at_tpr_90_mean": mean_of("fpr_at_tpr_90"),
            "two_sample_auc_mean": mean_of("two_sample_auc"),
            "cr_ks_mean": mean_of_ks("Cr"),
            "ni_ks_mean": mean_of_ks("Ni"),
            "positive_rate_mean": mean_of("positive_rate"),
        }
    return result


def run_confirmation(
    labels: tuple[str, ...] = ("baseline", "p11", "p15"),
    n_samples: int = 20000,
    seeds: tuple[int, ...] = (1, 2),
    seed: int = 0,
    outdir: str | Path = "src/potluck/tests/v6",
    verbose: bool = False,
) -> dict[str, Any]:
    """Re-run selected design points at larger n on independent seeds."""
    config = ModelConfig()
    points = {row["label"]: row for row in design_points(seed=seed)}
    missing = [label for label in labels if label not in points]
    if missing:
        raise ValueError(f"unknown design labels: {missing}")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    records = load_li_records()
    rows: list[dict[str, Any]] = []
    for label in labels:
        levels = points[label]["levels"]
        variant = _variant(config, levels)
        for run_seed in seeds:
            if verbose:
                print(f"confirming {label} seed {run_seed}", flush=True)
            row: dict[str, Any] = {
                "label": label,
                "seed": int(run_seed),
                "levels": levels,
                "settings": _settings(levels),
            }
            try:
                row["metrics"] = _confirm_metrics(
                    variant,
                    records,
                    n_samples,
                    run_seed,
                    verbose,
                )
            except (RuntimeError, ValueError) as error:
                row["metrics"] = None
                row["error"] = str(error)
            rows.append(row)
    summary = {
        "n_samples": n_samples,
        "seeds": list(seeds),
        "labels": list(labels),
        "excluded_elements": list(config.observation.exclude_elements),
        "aggregate": _aggregate_confirmation(rows, labels),
        "runs": rows,
    }
    with (outdir / "prior_screen_confirmation.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    _plot_confirmation(summary, outdir / "prior_screen_confirmation.png")
    return summary


def _plot_confirmation(summary: dict[str, Any], path: Path) -> Path:
    """Plot repeated-run support and classifier metrics per design point."""
    labels = summary["labels"]
    aggregate = summary["aggregate"]
    means = [
        aggregate[label].get("in_model_mean") or 0.0 for label in labels
    ]
    lows = [
        means[index] - (aggregate[label].get("in_model_min") or means[index])
        for index, label in enumerate(labels)
    ]
    highs = [
        (aggregate[label].get("in_model_max") or means[index]) - means[index]
        for index, label in enumerate(labels)
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    positions = np.arange(len(labels))

    axes[0].bar(positions, means, color="tab:blue", alpha=0.75)
    axes[0].errorbar(
        positions,
        means,
        yerr=np.vstack([lows, highs]),
        fmt="none",
        color="black",
        capsize=4,
    )
    for index, label in enumerate(labels):
        values = aggregate[label].get("in_model_values", [])
        axes[0].scatter(
            np.full(len(values), index),
            values,
            color="black",
            s=14,
            zorder=3,
        )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Li+ in-model fraction")
    axes[0].set_title("support (mean, min-max)")

    axes[1].bar(
        positions,
        [aggregate[label].get("cr_ks_mean") or 0.0 for label in labels],
        color="tab:red",
        alpha=0.75,
    )
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("Cr KS distance")
    axes[1].set_title("remaining Cr mismatch")

    axes[2].bar(
        positions,
        [aggregate[label].get("fpr_at_tpr_90_mean") or 0.0 for label in labels],
        color="tab:orange",
        alpha=0.75,
    )
    for index, label in enumerate(labels):
        auc = aggregate[label].get("classifier_auc_mean")
        if auc is not None:
            axes[2].text(
                index,
                0.02,
                f"AUC {auc:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )
    axes[2].set_xticks(positions)
    axes[2].set_xticklabels(labels)
    axes[2].set_ylim(0.0, max(0.3, axes[2].get_ylim()[1]))
    axes[2].set_ylabel("FPR at TPR 0.9")
    axes[2].set_title("classifier")

    fig.suptitle(
        f"prior confirmation ({summary['n_samples']} mocks, seeds "
        f"{summary['seeds']})"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_screen(summary: dict[str, Any], path: Path) -> Path:
    """Plot support fractions and main effects for every factor."""
    rows = summary["points"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.ravel()

    labels = [row["label"] for row in rows]
    values = [
        row["metrics"]["in_model_fraction"] if row["metrics"] else np.nan
        for row in rows
    ]
    order = np.argsort(np.asarray(values, dtype=float))[::-1]
    axes[0].bar(
        np.arange(len(rows)),
        [values[index] for index in order],
        color="tab:blue",
    )
    axes[0].set_xticks(np.arange(len(rows)))
    axes[0].set_xticklabels([labels[index] for index in order], rotation=90, fontsize=6)
    axes[0].set_ylabel("Li+ in-model fraction")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("design points")

    for axis, factor in zip(axes[1:], FACTOR_LEVELS):
        table = summary["effects"][factor]
        names = list(table)
        means = [table[name]["mean_in_model"] or 0.0 for name in names]
        axis.bar(np.arange(len(names)), means, color="tab:green", alpha=0.7)
        for column, name in enumerate(names):
            individual = table[name]["values"]
            axis.scatter(
                np.full(len(individual), column),
                individual,
                color="black",
                s=8,
                alpha=0.5,
            )
        axis.set_xticks(np.arange(len(names)))
        axis.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
        axis.set_ylim(0.0, 1.0)
        axis.set_title(factor)

    fig.suptitle(
        f"prior screen: baseline {summary['baseline']['metrics']['in_model_fraction']:.3f}, "
        f"best {summary['best']['metrics']['in_model_fraction']:.3f} "
        f"({summary['best']['label']})"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    """Run the prior screen from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3000, help="accepted mocks per point")
    parser.add_argument("--points", type=int, default=25, help="Latin-hypercube points")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--outdir",
        default="src/potluck/tests/v6",
        help="directory for JSON and PNG output",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="re-run selected design labels at larger n",
    )
    parser.add_argument(
        "--labels",
        default="baseline,p11,p15",
        help="comma-separated design labels for --confirm",
    )
    parser.add_argument(
        "--seeds",
        default="1,2",
        help="comma-separated generation seeds for --confirm",
    )
    args = parser.parse_args(argv)
    if args.confirm:
        labels = tuple(
            part.strip() for part in args.labels.split(",") if part.strip()
        )
        seeds = tuple(
            int(part) for part in args.seeds.split(",") if part.strip()
        )
        summary = run_confirmation(
            labels=labels,
            n_samples=args.n,
            seeds=seeds,
            seed=args.seed,
            outdir=args.outdir,
            verbose=args.verbose,
        )
        print(json.dumps(summary["aggregate"], indent=2))
        return 0
    summary = run_prior_screen(
        n_samples=args.n,
        n_points=args.points,
        seed=args.seed,
        outdir=args.outdir,
        verbose=args.verbose,
    )
    print(
        json.dumps(
            {
                "baseline": summary["baseline"]["metrics"]["in_model_fraction"],
                "best": summary["best"]["metrics"]["in_model_fraction"],
                "best_label": summary["best"]["label"],
                "failed": summary["n_failed"],
                "effects": {
                    factor: {
                        name: value["mean_in_model"]
                        for name, value in table.items()
                    }
                    for factor, table in summary["effects"].items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
