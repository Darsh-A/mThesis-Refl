"""Calibrated PISN-fraction posteriors and real Li+ star scoring."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from .constants import ELEMENT_INDEX, LI_FEATURE_ELEMENTS, REFERENCE
from .diagnostics.ceiling import observation_mask, observed_matrix
from .observation import DETECTED, GIANT_LOGG, LIRecords, load_li_records

CORE_SUPPORT_ELEMENTS = ("Na", "Mg", "Ca", "Ti", "Cr", "Ni")
INDEPENDENT_PAIRS = (("Na", "Mg"), ("Co", "Ni"), ("Mn", "Cr"), ("Zn", "Mg"))
DEFAULT_PREVALENCE = 2.5e-3


def active_core_elements(arrays: dict[str, np.ndarray]) -> tuple[str, ...]:
    """Return the core support elements populated by the observation layer."""
    ratio_elements = tuple(
        element for element in LI_FEATURE_ELEMENTS if element != REFERENCE
    )
    return tuple(
        element
        for element in CORE_SUPPORT_ELEMENTS
        if np.any(arrays["obs_flags"][:, ratio_elements.index(element)] > 0)
    )


@dataclass(frozen=True)
class PhaseBModels:
    """Trained classifier, calibrator, and f_Z regressor."""

    classifier: HistGradientBoostingClassifier
    calibrator: IsotonicRegression
    regressor: HistGradientBoostingRegressor
    train_rate: float
    split: dict[str, np.ndarray]


def _weighted_rate(y: np.ndarray, weight: np.ndarray) -> float:
    return float(np.average(y, weights=weight))


def prior_shift(
    probability: np.ndarray,
    prevalence: float,
    training_rate: float,
) -> np.ndarray:
    """Shift a probability from the training prior to a target prevalence."""
    probability = np.clip(probability, 1e-12, 1.0 - 1e-12)
    odds = probability / (1.0 - probability)
    odds *= (prevalence / (1.0 - prevalence)) / (
        training_rate / (1.0 - training_rate)
    )
    return odds / (1.0 + odds)


def _split(
    y: np.ndarray, seed: int, fractions: tuple[float, float, float]
) -> dict[str, np.ndarray]:
    index = np.arange(y.size)
    train_fraction, cal_fraction, _ = fractions
    train, rest = train_test_split(
        index,
        test_size=1.0 - train_fraction,
        stratify=y,
        random_state=seed,
    )
    cal, test = train_test_split(
        rest,
        test_size=1.0 - cal_fraction / (1.0 - train_fraction),
        stratify=y[rest],
        random_state=seed,
    )
    return {"train": train, "cal": cal, "test": test}


def fit_phase_b_models(
    arrays: dict[str, np.ndarray],
    seed: int = 0,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> PhaseBModels:
    """Fit the classifier, isotonic calibrator, and f_Z regressor."""
    x = observed_matrix(arrays)
    y = arrays["y_f_05"].astype(int)
    weight = arrays["importance_weight"].astype(float)
    split = _split(y, seed, fractions)

    classifier = HistGradientBoostingClassifier(
        max_iter=150,
        max_depth=3,
        learning_rate=0.05,
        random_state=seed,
    )
    classifier.fit(
        x[split["train"]],
        y[split["train"]],
        sample_weight=weight[split["train"]],
    )

    cal_score = classifier.predict_proba(x[split["cal"]])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(
        cal_score,
        y[split["cal"]],
        sample_weight=weight[split["cal"]],
    )

    regressor = HistGradientBoostingRegressor(
        max_iter=150,
        max_depth=3,
        learning_rate=0.05,
        random_state=seed,
    )
    regressor.fit(
        x[split["train"]],
        arrays["f_zp"][split["train"]],
        sample_weight=weight[split["train"]],
    )

    train_rate = _weighted_rate(y[split["train"]], weight[split["train"]])
    return PhaseBModels(
        classifier=classifier,
        calibrator=calibrator,
        regressor=regressor,
        train_rate=train_rate,
        split=split,
    )


def _reliability(
    y: np.ndarray,
    probability: np.ndarray,
    weight: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, float]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(probability, edges) - 1, 0, n_bins - 1)
    rows = []
    for i in range(n_bins):
        selected = index == i
        total = float(weight[selected].sum())
        if total <= 0.0:
            continue
        rows.append(
            {
                "low": float(edges[i]),
                "high": float(edges[i + 1]),
                "n": int(selected.sum()),
                "predicted": float(np.average(probability[selected], weights=weight[selected])),
                "observed": float(np.average(y[selected], weights=weight[selected])),
            }
        )
    return rows


def evaluate_calibration(
    y: np.ndarray,
    probability: np.ndarray,
    weight: np.ndarray,
) -> dict[str, object]:
    """Return weighted discrimination, Brier, and reliability."""
    return {
        "auc": float(roc_auc_score(y, probability, sample_weight=weight)),
        "brier": float(brier_score_loss(y, probability, sample_weight=weight)),
        "reliability": _reliability(y, probability, weight),
        "n": int(y.size),
    }


def bootstrap_posterior(
    cal_score: np.ndarray,
    cal_y: np.ndarray,
    cal_weight: np.ndarray,
    target_score: np.ndarray,
    train_rate: float,
    prevalence: float,
    n_boot: int = 200,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap the isotonic calibration and return 5–95% bounds."""
    rng = np.random.default_rng(seed)
    draws = np.full((n_boot, target_score.size), np.nan)
    probability = cal_weight / cal_weight.sum()
    for i in range(n_boot):
        index = rng.choice(cal_score.size, size=cal_score.size, p=probability)
        if np.unique(cal_y[index]).size < 2:
            continue
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(cal_score[index], cal_y[index], sample_weight=cal_weight[index])
        draws[i] = prior_shift(
            model.predict(target_score), prevalence, train_rate
        )
    low = np.nanpercentile(draws, 5, axis=0)
    high = np.nanpercentile(draws, 95, axis=0)
    return low, high


def real_matrix(
    records: LIRecords,
    keep: np.ndarray | None = None,
) -> np.ndarray:
    """Build the classifier input for real Li+ stars.

    ``keep`` must be the observation mask from the mock pool so that the
    real matrix has the same columns as the training matrix.
    """
    ratio_elements = tuple(
        element for element in LI_FEATURE_ELEMENTS if element != REFERENCE
    )
    ratio_columns = [
        LI_FEATURE_ELEMENTS.index(element) for element in ratio_elements
    ]
    ratios = records.xfe[:, ratio_columns]
    flags = records.flags[:, ratio_columns]
    sigma = records.sigma[:, ratio_columns]
    log_sigma = np.where(
        (flags != 0) & np.isfinite(sigma) & (sigma > 0.0),
        np.log10(np.where(sigma > 0.0, sigma, 1.0)),
        np.nan,
    )
    giant = (records.logg < GIANT_LOGG).astype(np.float32)[:, None]

    independent = np.full((records.n, len(INDEPENDENT_PAIRS)), np.nan)
    independent_sigma = np.full_like(independent, np.nan)
    for column, (numerator, denominator) in enumerate(INDEPENDENT_PAIRS):
        num = LI_FEATURE_ELEMENTS.index(numerator)
        den = LI_FEATURE_ELEMENTS.index(denominator)
        available = (records.flags[:, num] > 0) & (records.flags[:, den] > 0)
        value = records.xfe[:, num] - records.xfe[:, den]
        error = np.sqrt(
            np.square(records.sigma[:, num]) + np.square(records.sigma[:, den])
        )
        independent[:, column] = np.where(available, value, np.nan)
        independent_sigma[:, column] = np.where(available, error, np.nan)

    matrix = np.hstack(
        [
            ratios,
            flags,
            log_sigma,
            giant,
            independent,
            independent_sigma,
        ]
    ).astype(np.float32)
    if keep is not None:
        matrix = matrix[:, keep]
    return matrix


def _support_distances(
    real_ratios: np.ndarray,
    real_flags: np.ndarray,
    real_sigma: np.ndarray,
    mock_ratios: np.ndarray,
    mock_flags: np.ndarray,
    mock_sigma: np.ndarray,
    columns: list[int],
    min_core: int = 4,
) -> np.ndarray:
    distances = np.full(real_ratios.shape[0], np.nan)
    for i in range(real_ratios.shape[0]):
        selected = [
            column
            for column in columns
            if real_flags[i, column] > 0
            and np.isfinite(real_ratios[i, column])
            and np.isfinite(real_sigma[i, column])
        ]
        if len(selected) < min_core:
            continue
        total = np.zeros(mock_ratios.shape[0])
        for column in selected:
            valid = (
                (mock_flags[:, column] > 0)
                & np.isfinite(mock_ratios[:, column])
                & np.isfinite(mock_sigma[:, column])
            )
            difference = (mock_ratios[:, column] - real_ratios[i, column]) / real_sigma[i, column]
            total += np.where(valid, np.square(difference), np.inf)
        distances[i] = np.sqrt(np.min(total))
    return distances


def support_gate(
    records: LIRecords,
    arrays: dict[str, np.ndarray],
    n_reference: int = 2000,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Return real-star support distances and the mock 99th percentile."""
    ratio_elements = tuple(
        element for element in LI_FEATURE_ELEMENTS if element != REFERENCE
    )
    core_elements = active_core_elements(arrays)
    if not core_elements:
        raise ValueError("no core support elements are populated")
    columns = [ratio_elements.index(element) for element in core_elements]
    min_core = min(4, len(columns))
    real_ratios = records.xfe[
        :, [LI_FEATURE_ELEMENTS.index(element) for element in ratio_elements]
    ]
    real_flags = records.flags[
        :, [LI_FEATURE_ELEMENTS.index(element) for element in ratio_elements]
    ]
    real_sigma = records.sigma[
        :, [LI_FEATURE_ELEMENTS.index(element) for element in ratio_elements]
    ]

    rng = np.random.default_rng(seed)
    total = arrays["obs_features"].shape[0]
    reference = rng.choice(
        total,
        size=min(n_reference, max(1, total - 1)),
        replace=False,
    )
    keep = np.ones(total, dtype=bool)
    keep[reference] = False

    mock_reference = np.column_stack(
        [
            arrays["log_abundance"][:, ELEMENT_INDEX[element]]
            - arrays["log_abundance"][:, ELEMENT_INDEX[REFERENCE]]
            for element in ratio_elements
        ]
    )
    mock_flags = np.isfinite(mock_reference)
    mock_sigma = np.ones_like(mock_reference)

    reference_distance = _support_distances(
        arrays["obs_features"][reference],
        arrays["obs_flags"][reference],
        arrays["obs_sigma"][reference],
        mock_reference[keep],
        mock_flags[keep],
        mock_sigma[keep],
        columns,
        min_core=min_core,
    )
    threshold = float(np.nanpercentile(reference_distance, 99))
    real_distance = _support_distances(
        real_ratios,
        real_flags,
        real_sigma,
        mock_reference,
        mock_flags,
        mock_sigma,
        columns,
        min_core=min_core,
    )
    return real_distance, threshold


def score_real_stars(
    arrays: dict[str, np.ndarray],
    models: PhaseBModels,
    prevalence: float = DEFAULT_PREVALENCE,
    seed: int = 0,
    n_boot: int = 200,
) -> pd.DataFrame:
    """Score real Li+ stars and return a candidate table."""
    records = load_li_records()
    x_real = real_matrix(records, keep=observation_mask(arrays))
    score = models.classifier.predict_proba(x_real)[:, 1]
    calibrated = models.calibrator.predict(score)
    posterior = prior_shift(calibrated, prevalence, models.train_rate)
    expected_fz = np.clip(models.regressor.predict(x_real), 0.0, 1.0)

    cal = models.split["cal"]
    x = observed_matrix(arrays)
    cal_score = models.classifier.predict_proba(x[cal])[:, 1]
    low, high = bootstrap_posterior(
        cal_score,
        arrays["y_f_05"][cal].astype(int),
        arrays["importance_weight"][cal].astype(float),
        score,
        models.train_rate,
        prevalence,
        n_boot=n_boot,
        seed=seed,
    )

    distance, threshold = support_gate(records, arrays, seed=seed)
    core_elements = active_core_elements(arrays)
    ratio_elements = tuple(
        element for element in LI_FEATURE_ELEMENTS if element != REFERENCE
    )
    available = []
    missing = []
    limits = []
    for i in range(records.n):
        flags = records.flags[i]
        available.append(
            ",".join(
                element
                for element in ratio_elements
                if flags[LI_FEATURE_ELEMENTS.index(element)] > 0
            )
        )
        missing.append(
            ",".join(
                element
                for element in ratio_elements
                if flags[LI_FEATURE_ELEMENTS.index(element)] == 0
            )
        )
        limits.append(
            ",".join(
                element
                for element in ratio_elements
                if flags[LI_FEATURE_ELEMENTS.index(element)] == 2
            )
        )

    table = pd.DataFrame(
        {
            "star": records.labels,
            "feh": records.feh,
            "n_core": [
                sum(
                    records.flags[i, LI_FEATURE_ELEMENTS.index(element)] > 0
                    for element in core_elements
                )
                for i in range(records.n)
            ],
            "score": score,
            "p_calibrated": calibrated,
            "p_posterior": posterior,
            "p_low": low,
            "p_high": high,
            "expected_fz": expected_fz,
            "support_distance": distance,
            "support_threshold": threshold,
            "out_of_model": ~np.isfinite(distance) | (distance > threshold),
            "near_threshold": (expected_fz >= 0.25) & (expected_fz < 0.5),
            "candidate": posterior >= 0.5,
            "is_giant": records.logg < GIANT_LOGG,
            "available": available,
            "missing": missing,
            "limits": limits,
        }
    )
    return table.sort_values("p_posterior", ascending=False).reset_index(drop=True)


def _save_reliability(
    test_y: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    weight: np.ndarray,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    for values, label, colour in (
        (before, "before", "tab:grey"),
        (after, "isotonic", "tab:blue"),
    ):
        rows = _reliability(test_y, values, weight)
        ax.plot(
            [row["predicted"] for row in rows],
            [row["observed"] for row in rows],
            "o-",
            color=colour,
            label=label,
        )
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed positive fraction")
    ax.set_title("Held-out calibration (Li+ weights)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_real(
    table: pd.DataFrame,
    prevalence: float,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    axes[0].hist(
        table["p_posterior"],
        bins=np.linspace(0.0, 1.0, 41),
        color="tab:blue",
        alpha=0.8,
    )
    axes[0].axvline(prevalence, color="tab:red", ls="--", lw=1)
    axes[0].set_xlabel(f"P(f_Z >= 0.5 | x), prior={prevalence:g}")
    axes[0].set_ylabel("Li+ stars")
    axes[0].set_yscale("log")
    axes[1].scatter(
        table["support_distance"],
        table["p_posterior"],
        c=table["expected_fz"],
        cmap="viridis",
        s=18,
    )
    axes[1].axvline(
        table["support_threshold"].iloc[0],
        color="tab:red",
        ls="--",
        lw=1,
        label="mock 99th percentile",
    )
    axes[1].set_xlabel("Support distance")
    axes[1].set_ylabel("Posterior")
    axes[1].legend(frameon=False)
    axes[1].set_yscale("symlog", linthresh=1e-4)
    fig.suptitle("Real Li+ candidates")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_phase_b(
    dataset: str | Path,
    outdir: str | Path,
    prevalence: float = DEFAULT_PREVALENCE,
    seed: int = 0,
    n_boot: int = 200,
) -> dict[str, object]:
    """Run calibration, support gating, and real-star scoring."""
    from .io import load_dataset

    arrays, meta = load_dataset(dataset)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    models = fit_phase_b_models(arrays, seed=seed)
    test = models.split["test"]
    x = observed_matrix(arrays)
    y_test = arrays["y_f_05"][test].astype(int)
    weight_test = arrays["importance_weight"][test].astype(float)
    before = models.classifier.predict_proba(x[test])[:, 1]
    after = models.calibrator.predict(before)

    table = score_real_stars(
        arrays, models, prevalence=prevalence, seed=seed, n_boot=n_boot
    )
    table.to_csv(outdir / "candidates.csv", index=False)
    _save_reliability(y_test, before, after, weight_test, outdir / "reliability.png")
    _save_real(table, prevalence, outdir / "real_stars.png")

    report = {
        "n": int(len(arrays["y_f_05"])),
        "train_rate": models.train_rate,
        "prevalence": prevalence,
        "before": evaluate_calibration(y_test, before, weight_test),
        "after": evaluate_calibration(y_test, after, weight_test),
        "support_threshold": float(table["support_threshold"].iloc[0]),
        "n_candidates": int(table["candidate"].sum()),
        "n_out_of_model": int(table["out_of_model"].sum()),
        "n_near_threshold": int(table["near_threshold"].sum()),
        "config_hash": meta.get("config_hash"),
    }
    (outdir / "calibration.json").write_text(json.dumps(report, indent=2))
    return report
