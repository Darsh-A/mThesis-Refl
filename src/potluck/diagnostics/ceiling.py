"""Noiseless chemistry ceiling from a small tree ensemble."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from ..constants import ELEMENT_INDEX


def _design_matrix(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Features plus explicit missingness flags."""
    parts = [
        arrays["features"].astype(np.float32),
        arrays["ratio_mask"].astype(np.float32),
    ]
    if "independent_features" in arrays:
        parts.extend(
            [
                arrays["independent_features"].astype(np.float32),
                arrays["independent_mask"].astype(np.float32),
            ]
        )
    return np.hstack(parts)


def _raw_observed_matrix(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Return the full observation matrix before column filtering."""
    sigma = arrays["obs_sigma"].astype(np.float32)
    log_sigma = np.where(sigma > 0.0, np.log10(sigma), np.nan)
    parts = [
        arrays["obs_features"].astype(np.float32),
        arrays["obs_flags"].astype(np.float32),
        log_sigma.astype(np.float32),
        arrays["is_giant"].astype(np.float32)[:, None],
        arrays["obs_independent_features"].astype(np.float32),
        arrays["obs_independent_sigma"].astype(np.float32),
    ]
    return np.hstack(parts)


def _nonconstant_columns(matrix: np.ndarray) -> np.ndarray:
    """Return columns with at least two finite values.

    Histogram gradient boosting cannot bin a constant or all-missing column.
    Excluded elements produce exactly such columns.
    """
    keep = np.zeros(matrix.shape[1], dtype=bool)
    for column in range(matrix.shape[1]):
        values = matrix[np.isfinite(matrix[:, column]), column]
        keep[column] = np.unique(values).size >= 2
    return keep


def observation_mask(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Return the observation-matrix columns used by the model."""
    return _nonconstant_columns(_raw_observed_matrix(arrays))


def observed_matrix(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Model input from the Li+ observation layer only."""
    matrix = _raw_observed_matrix(arrays)
    return matrix[:, _nonconstant_columns(matrix)]


def _nuisance_matrix(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Environment variables that are not abundance features."""
    columns = [
        np.log10(np.clip(arrays["z_star"], 1e-30, None)),
        np.log10(np.clip(arrays["t_myr"], 1e-30, None)),
        np.log10(np.clip(arrays["mdil"], 1e-30, None)),
        np.log10(np.clip(arrays["mstar_ii"], 1e-30, None)),
        arrays["delta_myr"],
    ]
    return np.column_stack(columns).astype(np.float32)


def _classifier_probability(
    x: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> np.ndarray:
    probability = np.zeros(len(y), dtype=np.float64)
    for train, test in splits:
        model = HistGradientBoostingClassifier(
            max_iter=150,
            max_depth=3,
            learning_rate=0.05,
            random_state=seed,
        )
        model.fit(x[train], y[train])
        probability[test] = model.predict_proba(x[test])[:, 1]
    return probability


def _regression_scores(
    x: np.ndarray,
    target: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> tuple[np.ndarray, float, float]:
    prediction = np.full(len(target), np.nan, dtype=np.float64)
    for train, test in splits:
        model = HistGradientBoostingRegressor(
            max_iter=150,
            max_depth=3,
            learning_rate=0.05,
            random_state=seed,
        )
        model.fit(x[train], target[train])
        prediction[test] = model.predict(x[test])
    finite = np.isfinite(target) & np.isfinite(prediction)
    if finite.sum() < 2:
        return prediction, float("nan"), float("nan")
    return (
        prediction,
        float(r2_score(target[finite], prediction[finite])),
        float(mean_absolute_error(target[finite], prediction[finite])),
    )


def _reliability_curve(
    y: np.ndarray, probability: np.ndarray, n_bins: int = 10
) -> dict[str, Any]:
    """Binned calibration curve and Brier score."""
    y = np.asarray(y, dtype=float)
    probability = np.asarray(probability, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(probability, edges) - 1, 0, n_bins - 1)
    rows = []
    for bin_index in range(n_bins):
        mask = index == bin_index
        count = int(mask.sum())
        rows.append(
            {
                "low": float(edges[bin_index]),
                "high": float(edges[bin_index + 1]),
                "count": count,
                "mean_predicted": (
                    float(probability[mask].mean()) if count else None
                ),
                "observed": float(y[mask].mean()) if count else None,
            }
        )
    brier = float(np.mean((probability - y) ** 2)) if len(y) else float("nan")
    return {
        "bins": rows,
        "brier": brier,
        "mean_predicted": float(probability.mean()) if len(y) else None,
        "observed": float(y.mean()) if len(y) else None,
    }


def _stratified_auc(
    y: np.ndarray,
    probability: np.ndarray,
    values: np.ndarray,
    n_bins: int = 5,
) -> list[dict[str, Any]]:
    """AUC within quantile bins of a nuisance variable."""
    log_values = np.log10(np.clip(values, 1e-30, None))
    edges = np.quantile(log_values, np.linspace(0.0, 1.0, n_bins + 1))
    rows: list[dict[str, Any]] = []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        if index == n_bins - 1:
            mask = (log_values >= low) & (log_values <= high)
        else:
            mask = (log_values >= low) & (log_values < high)
        n = int(mask.sum())
        if n < 10 or len(np.unique(y[mask])) < 2:
            auc = None
        else:
            auc = float(roc_auc_score(y[mask], probability[mask]))
        rows.append(
            {
                "low": float(low),
                "high": float(high),
                "n": n,
                "y_rate": float(y[mask].mean()) if n else None,
                "auc": auc,
            }
        )
    return rows


def noisy_arrays(
    arrays: dict[str, np.ndarray],
    meta: dict[str, Any],
    sigma: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Return a copy with independent noise added to each log abundance."""
    log_abundance = np.asarray(arrays["log_abundance"], dtype=np.float64)
    noisy = log_abundance + rng.normal(0.0, sigma, log_abundance.shape)
    feature_elements = meta["feature_elements"]
    reference = meta["reference"]
    feature_indices = [
        ELEMENT_INDEX[element]
        for element in feature_elements
        if element != reference
    ]
    reference_index = ELEMENT_INDEX[reference]
    ratios = noisy[:, feature_indices] - noisy[:, reference_index][:, None]
    ratio_mask = arrays["ratio_mask"] & np.isfinite(ratios)

    result = dict(arrays)
    result["log_abundance"] = noisy.astype(np.float32)
    result["features"] = ratios.astype(np.float32)
    result["ratio_mask"] = ratio_mask
    return result


def noise_sweep(
    arrays: dict[str, np.ndarray],
    meta: dict[str, Any],
    sigmas: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5),
    cv: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """Repeat the chemistry ceiling over a grid of element-level noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for sigma in sigmas:
        perturbed = noisy_arrays(arrays, meta, sigma, rng)
        report = ceiling_report(perturbed, meta, cv=cv, seed=seed)
        rows.append(
            {
                "sigma": float(sigma),
                "auc": report["auc"],
                "f_fep_r2": report["f_fep_r2"],
                "f_zp_r2": report["f_zp_r2"],
            }
        )
    return {
        "sigmas": [row["sigma"] for row in rows],
        "auc": [row["auc"] for row in rows],
        "f_fep_r2": [row["f_fep_r2"] for row in rows],
        "f_zp_r2": [row["f_zp_r2"] for row in rows],
    }


def _binned_scores(
    y: np.ndarray,
    probability: np.ndarray,
    f_fep_true: np.ndarray,
    f_fep_prediction: np.ndarray,
    f_zp_true: np.ndarray,
    f_zp_prediction: np.ndarray,
    values: np.ndarray,
    edges: np.ndarray,
) -> list[dict[str, Any]]:
    """Classification and regression scores in fixed-value bins."""
    rows: list[dict[str, Any]] = []
    finite = np.isfinite(values)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = finite & (values >= low) & (values < high)
        n = int(mask.sum())
        row: dict[str, Any] = {
            "low": float(low),
            "high": float(high),
            "center": (
                float(np.median(values[mask])) if n else float((low + high) / 2)
            ),
            "n": n,
            "y_rate": float(y[mask].mean()) if n else None,
            "auc": None,
            "brier": None,
            "f_fep_r2": None,
            "f_zp_r2": None,
            "f_fep_mae": None,
            "f_zp_mae": None,
            "f_fep_rmse": None,
            "f_zp_rmse": None,
        }
        if n:
            row["brier"] = float(brier_score_loss(y[mask], probability[mask]))
        if n >= 10 and len(np.unique(y[mask])) == 2:
            row["auc"] = float(roc_auc_score(y[mask], probability[mask]))
        for key, target, prediction in (
            ("f_fep", f_fep_true, f_fep_prediction),
            ("f_zp", f_zp_true, f_zp_prediction),
        ):
            good = mask & np.isfinite(target) & np.isfinite(prediction)
            if good.sum() >= 2:
                row[f"{key}_mae"] = float(
                    mean_absolute_error(target[good], prediction[good])
                )
                row[f"{key}_rmse"] = float(
                    np.sqrt(np.mean((target[good] - prediction[good]) ** 2))
                )
                if np.std(target[good]) > 0.02:
                    row[f"{key}_r2"] = float(
                        r2_score(target[good], prediction[good])
                    )
        rows.append(row)
    return rows


def ceiling_report(
    arrays: dict[str, np.ndarray],
    meta: dict[str, Any],
    cv: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """Cross-validated AUC and regression scores on noiseless features."""
    del meta
    x = _design_matrix(arrays)
    y = arrays["y"].astype(np.int8)
    counts = np.bincount(y, minlength=2)
    n_splits = int(min(cv, counts[counts > 0].min()))
    if n_splits < 2:
        return {
            "n": int(len(y)),
            "n_splits": n_splits,
            "auc": float("nan"),
            "y_rate": float(y.mean()) if len(y) else float("nan"),
            "auc_by_z_star": [],
            "auc_by_mstar_ii": [],
            "nuisance_auc_by_z_star": [],
            "nuisance_auc_by_mstar_ii": [],
            "by_fzp": [],
            "by_pisn_mass": [],
            "by_feh": [],
        }

    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    splits = list(splitter.split(x, y))
    probability = _classifier_probability(x, y, splits, seed)
    nuisance = _nuisance_matrix(arrays)
    nuisance_probability = _classifier_probability(nuisance, y, splits, seed)

    f_fep_pred, f_fep_r2, f_fep_mae = _regression_scores(
        x, arrays["f_fep"].astype(np.float64), splits, seed
    )
    f_zp_pred, f_zp_r2, f_zp_mae = _regression_scores(
        x, arrays["f_zp"].astype(np.float64), splits, seed
    )

    return {
        "n": int(len(y)),
        "n_splits": n_splits,
        "y_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, probability)),
        "nuisance_auc": float(roc_auc_score(y, nuisance_probability)),
        "nuisance_features": [
            "log10_z_star",
            "log10_t",
            "log10_mdil",
            "log10_mstar_ii",
            "delta_myr",
        ],
        "f_fep_r2": f_fep_r2,
        "f_fep_mae": f_fep_mae,
        "f_zp_r2": f_zp_r2,
        "f_zp_mae": f_zp_mae,
        "auc_by_z_star": _stratified_auc(y, probability, arrays["z_star"]),
        "auc_by_mstar_ii": _stratified_auc(
            y, probability, arrays["mstar_ii"]
        ),
        "nuisance_auc_by_z_star": _stratified_auc(
            y, nuisance_probability, arrays["z_star"]
        ),
        "nuisance_auc_by_mstar_ii": _stratified_auc(
            y, nuisance_probability, arrays["mstar_ii"]
        ),
        "by_fzp": _binned_scores(
            y,
            probability,
            arrays["f_fep"].astype(np.float64),
            f_fep_pred,
            arrays["f_zp"].astype(np.float64),
            f_zp_pred,
            arrays["f_zp"].astype(np.float64),
            np.asarray([-0.01, 0.01, 0.25, 0.50, 0.75, 0.99, 1.01]),
        ),
        "by_pisn_mass": _binned_scores(
            y,
            probability,
            arrays["f_fep"].astype(np.float64),
            f_fep_pred,
            arrays["f_zp"].astype(np.float64),
            f_zp_pred,
            np.where(
                arrays["n_pisn"] > 0,
                arrays["pisn_masses"][:, 0].astype(np.float64),
                np.nan,
            ),
            np.asarray([140.0, 160.0, 180.0, 200.0, 220.0, 240.0, 270.1]),
        ),
        "by_feh": _binned_scores(
            y,
            probability,
            arrays["f_fep"].astype(np.float64),
            f_fep_pred,
            arrays["f_zp"].astype(np.float64),
            f_zp_pred,
            arrays["feh"].astype(np.float64),
            np.asarray([-5.0, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, 0.0, 1.0]),
        ),
        "y_true": y,
        "y_probability": probability,
        "nuisance_probability": nuisance_probability,
        "reliability": _reliability_curve(y, probability),
        "nuisance_reliability": _reliability_curve(y, nuisance_probability),
        "f_fep_true": arrays["f_fep"].astype(np.float64),
        "f_fep_prediction": f_fep_pred,
        "f_zp_true": arrays["f_zp"].astype(np.float64),
        "f_zp_prediction": f_zp_pred,
    }
