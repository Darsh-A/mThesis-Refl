"""Decompose real-star support distances and ablate the model gates."""

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
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split

from ..calibration import (
    CORE_SUPPORT_ELEMENTS,
    active_core_elements,
    support_gate,
)
from ..config import ModelConfig
from ..constants import ELEMENT_INDEX, LI_FEATURE_ELEMENTS, REFERENCE
from ..dataset import build_tables, generate_dataset
from ..io import load_dataset
from ..observation import GIANT_LOGG, LIRecords, load_li_records

FEH_BINS = (-4.0, -3.5, -3.0, -2.5, -2.0, -1.5)


def _ratio_elements() -> tuple[str, ...]:
    """Return the Li+ ratio elements in catalogue order."""
    return tuple(element for element in LI_FEATURE_ELEMENTS if element != REFERENCE)


def _real_arrays(records: LIRecords) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return real ratios, flags, and errors in ratio-element order."""
    elements = _ratio_elements()
    columns = [LI_FEATURE_ELEMENTS.index(element) for element in elements]
    return records.xfe[:, columns], records.flags[:, columns], records.sigma[:, columns]


def _mock_ratios(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Return noiseless mock ratios in ratio-element order."""
    reference = arrays["log_abundance"][:, ELEMENT_INDEX[REFERENCE]]
    return np.column_stack(
        [
            arrays["log_abundance"][:, ELEMENT_INDEX[element]] - reference
            for element in _ratio_elements()
        ]
    )


def _core_columns(arrays: dict[str, np.ndarray]) -> list[int]:
    """Return ratio-column indices for the populated core support elements."""
    elements = _ratio_elements()
    return [
        elements.index(element) for element in active_core_elements(arrays)
    ]


def _nearest_distances(
    query_ratios: np.ndarray,
    query_flags: np.ndarray,
    query_sigma: np.ndarray,
    reference_ratios: np.ndarray,
    reference_flags: np.ndarray,
    columns: list[int],
    min_core: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest-reference distances and indices."""
    distances = np.full(query_ratios.shape[0], np.nan)
    nearest = np.full(query_ratios.shape[0], -1, dtype=np.int64)
    for i in range(query_ratios.shape[0]):
        selected = [
            column
            for column in columns
            if query_flags[i, column] > 0
            and np.isfinite(query_ratios[i, column])
            and np.isfinite(query_sigma[i, column])
        ]
        if len(selected) < min_core:
            continue
        total = np.zeros(reference_ratios.shape[0])
        for column in selected:
            valid = (
                (reference_flags[:, column] > 0)
                & np.isfinite(reference_ratios[:, column])
            )
            difference = (
                reference_ratios[:, column] - query_ratios[i, column]
            ) / query_sigma[i, column]
            total += np.where(valid, np.square(difference), np.inf)
        index = int(np.argmin(total))
        if np.isfinite(total[index]):
            distances[i] = np.sqrt(total[index])
            nearest[i] = index
    return distances, nearest


def _threshold(
    query_ratios: np.ndarray,
    query_flags: np.ndarray,
    query_sigma: np.ndarray,
    reference_ratios: np.ndarray,
    reference_flags: np.ndarray,
    columns: list[int],
    min_core: int = 4,
    n_reference: int = 2000,
    seed: int = 0,
) -> tuple[float, np.ndarray]:
    """Return the 99th percentile and the reference distances."""
    rng = np.random.default_rng(seed)
    total = reference_ratios.shape[0]
    chosen = rng.choice(
        total,
        size=min(n_reference, max(1, total - 1)),
        replace=False,
    )
    keep = np.ones(total, dtype=bool)
    keep[chosen] = False
    distances, _ = _nearest_distances(
        query_ratios[chosen],
        query_flags[chosen],
        query_sigma[chosen],
        reference_ratios[keep],
        reference_flags[keep],
        columns,
        min_core=min_core,
    )
    return float(np.nanpercentile(distances, 99)), distances


def decompose_support(
    records: LIRecords,
    arrays: dict[str, np.ndarray],
    n_reference: int = 2000,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return per-star distances, drivers, and summary variants."""
    real_ratios, real_flags, real_sigma = _real_arrays(records)
    mock_ratios = _mock_ratios(arrays)
    mock_flags = np.isfinite(mock_ratios)
    observed_ratios = arrays["obs_features"]
    observed_flags = arrays["obs_flags"]
    observed_sigma = arrays["obs_sigma"]

    core_elements = active_core_elements(arrays)
    if not core_elements:
        raise ValueError("no core support elements are populated")
    columns = _core_columns(arrays)
    min_core = min(4, len(columns))
    available_core = min(3, len(columns))

    current_threshold, _ = _threshold(
        observed_ratios,
        observed_flags,
        observed_sigma,
        mock_ratios,
        mock_flags,
        columns,
        min_core=min_core,
        n_reference=n_reference,
        seed=seed,
    )
    distance, nearest = _nearest_distances(
        real_ratios,
        real_flags,
        real_sigma,
        mock_ratios,
        mock_flags,
        columns,
        min_core=min_core,
    )

    available_threshold, _ = _threshold(
        observed_ratios,
        observed_flags,
        observed_sigma,
        mock_ratios,
        mock_flags,
        columns,
        min_core=available_core,
        n_reference=n_reference,
        seed=seed,
    )
    available_distance, _ = _nearest_distances(
        real_ratios,
        real_flags,
        real_sigma,
        mock_ratios,
        mock_flags,
        columns,
        min_core=available_core,
    )

    observed_threshold, _ = _threshold(
        observed_ratios,
        observed_flags,
        observed_sigma,
        observed_ratios,
        observed_flags,
        columns,
        min_core=min_core,
        n_reference=n_reference,
        seed=seed,
    )
    observed_distance, _ = _nearest_distances(
        real_ratios,
        real_flags,
        real_sigma,
        observed_ratios,
        observed_flags,
        columns,
        min_core=min_core,
    )

    element_columns = {
        element: _ratio_elements().index(element) for element in core_elements
    }
    element_distances: dict[str, np.ndarray] = {}
    element_thresholds: dict[str, float] = {}
    for element, column in element_columns.items():
        threshold, _ = _threshold(
            observed_ratios,
            observed_flags,
            observed_sigma,
            mock_ratios,
            mock_flags,
            [column],
            min_core=1,
            n_reference=n_reference,
            seed=seed,
        )
        distances, _ = _nearest_distances(
            real_ratios,
            real_flags,
            real_sigma,
            mock_ratios,
            mock_flags,
            [column],
            min_core=1,
        )
        element_thresholds[element] = threshold
        element_distances[element] = distances

    rows = []
    for i in range(records.n):
        selected = [
            column
            for column in columns
            if real_flags[i, column] > 0
            and np.isfinite(real_ratios[i, column])
            and np.isfinite(real_sigma[i, column])
        ]
        driver = ""
        driver_residual = np.nan
        if nearest[i] >= 0 and selected:
            residuals = {
                _ratio_elements()[column]: abs(
                    (mock_ratios[nearest[i], column] - real_ratios[i, column])
                    / real_sigma[i, column]
                )
                for column in selected
            }
            driver = max(residuals, key=residuals.get)
            driver_residual = float(residuals[driver])

        row: dict[str, Any] = {
            "star": records.labels[i],
            "feh": float(records.feh[i]),
            "logg": float(records.logg[i]),
            "is_giant": bool(records.logg[i] < GIANT_LOGG),
            "n_core": len(selected),
            "distance": float(distance[i]),
            "threshold": current_threshold,
            "in_model": bool(np.isfinite(distance[i]) and distance[i] <= current_threshold),
            "distance_available": float(available_distance[i]),
            "threshold_available": available_threshold,
            "in_model_available": bool(
                np.isfinite(available_distance[i])
                and available_distance[i] <= available_threshold
            ),
            "distance_observed": float(observed_distance[i]),
            "threshold_observed": observed_threshold,
            "in_model_observed": bool(
                np.isfinite(observed_distance[i])
                and observed_distance[i] <= observed_threshold
            ),
            "driver": driver,
            "driver_residual": driver_residual,
        }
        for element in core_elements:
            row[f"d_{element}"] = float(element_distances[element][i])
        rows.append(row)

    table = pd.DataFrame(rows)
    table["feh_bin"] = pd.cut(table["feh"], bins=FEH_BINS)
    summary = _summary(
        table,
        current_threshold,
        available_threshold,
        observed_threshold,
        element_thresholds,
        core_elements,
    )
    return table, summary


def _summary(
    table: pd.DataFrame,
    threshold: float,
    available_threshold: float,
    observed_threshold: float,
    element_thresholds: dict[str, float],
    core_elements: tuple[str, ...],
) -> dict[str, Any]:
    """Summarize decomposition fractions and drivers."""
    def rate(frame: pd.DataFrame, column: str) -> dict[str, Any]:
        return {
            "n": int(len(frame)),
            "in_model": int(frame[column].sum()),
            "fraction": float(frame[column].mean()) if len(frame) else None,
        }

    by_n_core = {
        str(key): rate(group, "in_model")
        for key, group in table.groupby("n_core", observed=True)
    }
    by_giant = {
        str(key): rate(group, "in_model")
        for key, group in table.groupby("is_giant", observed=True)
    }
    by_feh = {
        str(key): rate(group, "in_model")
        for key, group in table.groupby("feh_bin", observed=True)
    }
    drivers = table.loc[table["driver"] != "", "driver"].value_counts().to_dict()
    elements = {}
    for element in core_elements:
        values = table[f"d_{element}"]
        available = values[np.isfinite(values)]
        elements[element] = {
            "threshold": element_thresholds[element],
            "n": int(available.size),
            "median": float(available.median()) if available.size else None,
            "outlier_fraction": float((available > element_thresholds[element]).mean())
            if available.size
            else None,
        }
    return {
        "n": int(len(table)),
        "threshold": threshold,
        "in_model": int(table["in_model"].sum()),
        "in_model_fraction": float(table["in_model"].mean()),
        "threshold_available": available_threshold,
        "in_model_available": int(table["in_model_available"].sum()),
        "in_model_available_fraction": float(table["in_model_available"].mean()),
        "threshold_observed": observed_threshold,
        "in_model_observed": int(table["in_model_observed"].sum()),
        "in_model_observed_fraction": float(table["in_model_observed"].mean()),
        "by_n_core": by_n_core,
        "by_giant": by_giant,
        "by_feh": by_feh,
        "drivers": drivers,
        "elements": elements,
    }


def plot_decomposition(
    table: pd.DataFrame,
    summary: dict[str, Any],
    path: str | Path,
) -> Path:
    """Plot distance, driver, and availability diagnostics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    finite = table.loc[np.isfinite(table["distance"]), "distance"]
    axes[0, 0].hist(finite, bins=30, color="tab:blue", alpha=0.75)
    axes[0, 0].axvline(summary["threshold"], color="tab:red", ls="--", label="99th percentile")
    axes[0, 0].set_xlabel("support distance")
    axes[0, 0].set_ylabel("stars")
    axes[0, 0].set_title("Li+ distances")
    axes[0, 0].legend()

    drivers = table.loc[table["driver"] != "", "driver"].value_counts()
    axes[0, 1].bar(drivers.index, drivers.values, color="tab:orange")
    axes[0, 1].set_ylabel("stars")
    axes[0, 1].set_title("largest residual element")

    by_n_core = table.groupby("n_core", observed=True)["in_model"].mean()
    axes[1, 0].bar(by_n_core.index.astype(str), by_n_core.values, color="tab:green")
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].set_xlabel("core elements available")
    axes[1, 0].set_ylabel("in-model fraction")
    axes[1, 0].set_title("support vs availability")

    labels = list(summary["elements"])
    fractions = [summary["elements"][element]["outlier_fraction"] or 0.0 for element in labels]
    axes[1, 1].bar(labels, fractions, color="tab:purple")
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].set_ylabel("outlier fraction")
    axes[1, 1].set_title("per-element mismatch")

    fig.suptitle(
        f"Li+ support: {summary['in_model']}/{summary['n']} in-model "
        f"(threshold {summary['threshold']:.2f})"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def ablation_variants(config: ModelConfig) -> dict[str, ModelConfig]:
    """Return gate and prior variants for the support ablation."""
    popiii_table = build_tables(config)[1]
    no_100 = tuple(
        0.0 if mass == 100 else 1.0 for mass, _ in popiii_table.combo_keys
    )
    return {
        "current": config,
        "no_r4": replace(
            config,
            prior=replace(config.prior, feh_range=None),
        ),
        "no_zcr": replace(
            config,
            prior=replace(config.prior, z_cr=None),
        ),
        "cap_0.3": replace(
            config,
            prior=replace(config.prior, mstar_total_max_ratio=0.3),
        ),
        "wide_lambda": replace(
            config,
            prior=replace(
                config.prior,
                lambda_iii_range=(0.1, 500.0),
                n_iii_max=4096,
            ),
        ),
        "no_100": replace(
            config,
            prior=replace(config.prior, popiii_mass_energy_weights=no_100),
        ),
    }


def _classifier_metrics(arrays: dict[str, np.ndarray]) -> dict[str, float | None]:
    """Return weighted OOF AUC and FPR at TPR 0.9."""
    from .ceiling import observed_matrix

    x = observed_matrix(arrays)
    y = arrays["y_f_05"].astype(int)
    weight = arrays["importance_weight"].astype(float)
    counts = np.bincount(y, minlength=2)
    if y.min() == y.max() or counts.min() < 3:
        return {"auc": None, "fpr_at_tpr_90": None}
    probability = np.full(y.size, np.nan)
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    for train, test in folds.split(x, y):
        model = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=0,
        )
        model.fit(x[train], y[train], sample_weight=weight[train])
        probability[test] = model.predict_proba(x[test])[:, 1]
    auc = float(roc_auc_score(y, probability, sample_weight=weight))
    fpr, tpr, _ = roc_curve(y, probability, sample_weight=weight)
    index = int(np.searchsorted(tpr, 0.9))
    return {
        "auc": auc,
        "fpr_at_tpr_90": float(fpr[min(index, fpr.size - 1)]),
    }


def _two_sample_auc(
    records: LIRecords,
    arrays: dict[str, np.ndarray],
    seed: int = 0,
) -> float | None:
    """Return mock-vs-real AUC in observed ratio space."""
    real_ratios, _, _ = _real_arrays(records)
    mock_ratios = arrays["obs_features"]
    available = [
        column
        for column in range(mock_ratios.shape[1])
        if np.any(np.isfinite(mock_ratios[:, column]))
    ]
    real_ratios = real_ratios[:, available]
    mock_ratios = mock_ratios[:, available]
    if real_ratios.shape[0] < 10 or mock_ratios.shape[0] < 10:
        return None
    rng = np.random.default_rng(seed)
    if mock_ratios.shape[0] > 5000:
        mock_ratios = mock_ratios[
            rng.choice(mock_ratios.shape[0], size=5000, replace=False)
        ]
    x = np.vstack([mock_ratios, real_ratios])
    y = np.concatenate(
        [np.zeros(mock_ratios.shape[0], dtype=int), np.ones(real_ratios.shape[0], dtype=int)]
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, stratify=y, random_state=seed
    )
    model = HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=3,
        learning_rate=0.05,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    return float(roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]))


def _ks_metrics(
    records: LIRecords,
    arrays: dict[str, np.ndarray],
) -> dict[str, dict[str, float | None]]:
    """Return per-element KS statistics between mocks and Li+."""
    real_ratios, _, _ = _real_arrays(records)
    mock_ratios = arrays["obs_features"]
    result = {}
    for element in CORE_SUPPORT_ELEMENTS:
        column = _ratio_elements().index(element)
        real = real_ratios[:, column]
        mock = mock_ratios[:, column]
        real = real[np.isfinite(real)]
        mock = mock[np.isfinite(mock)]
        if real.size == 0 or mock.size == 0:
            result[element] = {"statistic": None, "pvalue": None}
            continue
        test = ks_2samp(real, mock)
        result[element] = {
            "statistic": float(test.statistic),
            "pvalue": float(test.pvalue),
        }
    return result


def run_support_ablation(
    config: ModelConfig,
    n_samples: int = 20000,
    seed: int = 0,
    outdir: str | Path = "src/potluck/tests/v3",
    verbose: bool = False,
) -> dict[str, Any]:
    """Generate each ablation pool and measure Li+ support."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    records = load_li_records()
    summary: dict[str, Any] = {}
    for name, variant in ablation_variants(config).items():
        size = 10000 if name == "wide_lambda" else n_samples
        if verbose:
            print(f"generating {name}: {size}")
        arrays, meta = generate_dataset(
            variant,
            n_samples=size,
            seed=seed,
            verbose=verbose,
        )
        distance, threshold = support_gate(records, arrays, seed=seed)
        in_model = np.isfinite(distance) & (distance <= threshold)
        classifier = _classifier_metrics(arrays)
        summary[name] = {
            "n": int(arrays["y_f_05"].size),
            "threshold": threshold,
            "in_model": int(in_model.sum()),
            "in_model_fraction": float(in_model.mean()),
            "median_distance": float(np.nanmedian(distance)),
            "p95_distance": float(np.nanpercentile(distance, 95)),
            "ks": _ks_metrics(records, arrays),
            "two_sample_auc": _two_sample_auc(records, arrays, seed=seed),
            "classifier_auc": classifier["auc"],
            "fpr_at_tpr_90": classifier["fpr_at_tpr_90"],
            "positive_rate": float(np.average(arrays["y_f_05"], weights=arrays["importance_weight"])),
            "ledger": meta["ledger"],
        }
    with (outdir / "support_ablation.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    _plot_ablation(summary, outdir / "support_ablation.png")
    return summary


def _plot_ablation(summary: dict[str, Any], path: Path) -> Path:
    """Plot in-model fraction and two-sample AUC per pool."""
    names = list(summary)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(names, [summary[name]["in_model_fraction"] for name in names], color="tab:blue")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Li+ in-model fraction")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].bar(names, [summary[name]["two_sample_auc"] or 0.0 for name in names], color="tab:orange")
    axes[1].axhline(0.5, color="black", ls="--", lw=1)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("mock-vs-real AUC")
    axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    """Run decomposition or ablation from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    decompose = sub.add_parser("decompose", help="decompose Li+ support distances")
    decompose.add_argument("dataset")
    decompose.add_argument("--outdir", default="src/potluck/tests/v3")

    ablate = sub.add_parser("ablate", help="ablate gates and priors")
    ablate.add_argument("dataset")
    ablate.add_argument("--n", type=int, default=20000)
    ablate.add_argument("--seed", type=int, default=0)
    ablate.add_argument("--outdir", default="src/potluck/tests/v3")
    ablate.add_argument("--verbose", action="store_true")

    args = parser.parse_args(argv)
    arrays, meta = load_dataset(args.dataset)
    config = ModelConfig.from_dict(meta["config"])
    records = load_li_records()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.command == "decompose":
        table, summary = decompose_support(records, arrays)
        table.to_csv(outdir / "support_decomposition.csv", index=False)
        with (outdir / "support_decomposition.json").open("w") as handle:
            json.dump(summary, handle, indent=2)
        plot_decomposition(table, summary, outdir / "support_decomposition.png")
        print(json.dumps(summary, indent=2))
        return 0

    summary = run_support_ablation(
        config,
        n_samples=args.n,
        seed=args.seed,
        outdir=outdir,
        verbose=args.verbose,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
