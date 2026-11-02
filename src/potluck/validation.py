"""Validation checks and prior-ablation scaffolding."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from .diagnostics.ceiling import _design_matrix, _nuisance_matrix

from src.formulas import raiteri_mass_from_lifetime
from src.load_data import load_hw2002, load_takahashi
from src.salvadori_funcs import (
    salvadori_Y_X_II,
    salvadori_Y_Z_II,
    salvadori_combined_abundratio,
    salvadori_yields,
)
from src.yield_sources import get_source

from .config import ModelConfig
from .constants import ELEMENT_INDEX, METALS, SOLAR_MASS_FRACTION
from .dataset import generate_dataset
from .diagnostics import diagnose
from .io import save_dataset
from .yields.base import element_vector


def _select_entry(table: str, mass: float) -> dict:
    """Return the PISN table entry closest to ``mass``."""
    if table == "HW2002":
        entries = load_hw2002()
    elif table.startswith("Takahashi-"):
        entries = load_takahashi()[table.split("-", 1)[1]]
    else:
        raise ValueError(f"unsupported PISN table for Eq. 13: {table}")
    return min(entries, key=lambda entry: abs(entry["params"]["mass"] - mass))


def eq13_check(
    config: ModelConfig,
    pisn_mass: float = 200.0,
    f_pisn: float = 0.5,
    f_ratio: float = 1.0e-3,
    tpop2_myr: float = 10.0,
    elements: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Compare this model's bookkeeping with Salvadori et al. (2019) Eq. 13.

    The check uses the legacy implementation as the reference.  It validates
    the PISN normalisation, the dilution mass, and the Pop II mass tied to
    ``f_pisn``, not the stochastic sampler.
    """
    entry = _select_entry(config.yields.pisn_table, pisn_mass)
    m_pisn = float(entry["params"]["mass"])
    normalized = salvadori_yields([entry])[0]
    yields_pisn = element_vector(normalized)
    masses_pisn = element_vector(entry)
    metal_indices = [ELEMENT_INDEX[element] for element in METALS]
    y_z_pisn = float(yields_pisn[metal_indices].sum())
    m_z_pisn = float(masses_pisn[metal_indices].sum())

    z_star = f_ratio * y_z_pisn / f_pisn
    source = get_source(config.yields.popii_table)
    popii_data = source.load_for_metallicity(z_star)
    m_popii = raiteri_mass_from_lifetime(tpop2_myr * 1.0e6, z_star)
    if m_popii is None:
        raise ValueError("turnoff mass is undefined for the requested t_popII")

    y_z_ii = salvadori_Y_Z_II(
        popii_data,
        m_popii,
        model=config.yields.popii_model,
        sn_input=source,
        feh=None,
    )
    beta = (1.0 - f_pisn) / f_pisn
    m_star_ii = beta * m_z_pisn / y_z_ii if y_z_ii > 0.0 else 0.0

    if elements is None:
        elements = tuple(e for e in config.feature_elements if e != config.reference)

    differences: dict[str, float] = {}
    for element in elements:
        y_x_ii = salvadori_Y_X_II(
            popii_data,
            element,
            m_popii,
            model=config.yields.popii_model,
            sn_input=source,
            feh=None,
        )
        y_fe_ii = salvadori_Y_X_II(
            popii_data,
            config.reference,
            m_popii,
            model=config.yields.popii_model,
            sn_input=source,
            feh=None,
        )
        total = masses_pisn[ELEMENT_INDEX[element]] + m_star_ii * y_x_ii
        total_fe = masses_pisn[ELEMENT_INDEX[config.reference]] + m_star_ii * y_fe_ii
        if total <= 0.0 or total_fe <= 0.0:
            continue
        ours = np.log10(
            (total / SOLAR_MASS_FRACTION[element])
            / (total_fe / SOLAR_MASS_FRACTION[config.reference])
        )
        reference = salvadori_combined_abundratio(
            element,
            element,
            config.reference,
            config.reference,
            normalized,
            popii_data,
            popii_data,
            False,
            False,
            source,
            f_pisn,
            f_ratio,
            tpop2_myr * 1.0e6,
            feh=None,
        )
        if np.isfinite(reference):
            differences[element] = float(ours - reference)

    finite = np.asarray([abs(value) for value in differences.values()])
    return {
        "pisn_mass": m_pisn,
        "f_pisn": f_pisn,
        "f_ratio": f_ratio,
        "tpop2_myr": tpop2_myr,
        "z_star": float(z_star),
        "m_star_ii": float(m_star_ii),
        "max_abs_difference": float(finite.max()) if finite.size else None,
        "mean_abs_difference": float(finite.mean()) if finite.size else None,
        "differences": differences,
    }


def validate_dataset(
    arrays: dict[str, np.ndarray],
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Check internal consistency of a stored dataset."""
    checks: dict[str, Any] = {}
    n = int(len(arrays["y"]))
    metal_indices = [ELEMENT_INDEX[element] for element in METALS]

    def record(name: str, error: float, tolerance: float) -> None:
        checks[name] = {
            "ok": bool(np.isfinite(error) and error <= tolerance),
            "max_error": float(error),
            "tolerance": float(tolerance),
        }

    def range_error(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return 0.0
        return float(np.max(np.maximum(finite - 1.0, -finite)))

    matrix = arrays["matrix"].astype(float)
    m_tot = arrays["M_tot"].astype(float)
    mass_tolerance = 5e-2  # float32 rounding reaches ~2e-2 at 2e5 M_sun
    record("matrix_sum", np.max(np.abs(matrix.sum(axis=1) - m_tot)), mass_tolerance)
    record(
        "metal_sum",
        np.max(np.abs(m_tot[:, metal_indices].sum(axis=1) - arrays["M_z_tot"])),
        mass_tolerance,
    )
    record(
        "pre_le_total",
        np.max(np.maximum(arrays["M_z_pre"] - arrays["M_z_tot"], 0.0)),
        mass_tolerance,
    )

    z_star = arrays["z_star"].astype(float)
    z_final = arrays["z_final"].astype(float)
    record(
        "z_final_ge_z_star",
        np.max(np.maximum(z_star - z_final, 0.0)),
        1e-6,
    )

    f_zp = arrays["f_zp"].astype(float)
    f_fep = arrays["f_fep"].astype(float)
    record("f_zp_range", range_error(f_zp), 1e-5)
    record("f_fep_range", range_error(f_fep), 1e-5)
    record("f_xp_range", range_error(arrays["f_xp"].astype(float)), 1e-5)

    mask = arrays["ratio_mask"].astype(bool)
    n_detected = arrays["n_detected"].astype(int)
    reason = arrays["detect_reason"]
    used = reason != 3
    if "obs_flags" in arrays:
        expected_detected = (arrays["obs_flags"] == 1).sum(axis=1)
    else:
        expected_detected = mask.sum(axis=1)
    checks["n_detected"] = {
        "ok": bool(np.array_equal(n_detected[used], expected_detected[used])),
        "max_error": float(
            np.max(
                np.abs(n_detected[used] - expected_detected[used]), initial=0.0
            )
        ),
        "tolerance": 0.0,
    }

    d = arrays["D"].astype(float)
    y = arrays["y"].astype(bool)
    normal = reason == 0
    checks["y_matches_threshold"] = {
        "ok": bool(np.array_equal(y[normal], d[normal] > meta["config"]["detect"]["k"])),
        "max_error": float(
            np.count_nonzero(y[normal] != (d[normal] > meta["config"]["detect"]["k"]))
        ),
        "tolerance": 0.0,
    }
    checks["no_pisn_label"] = {
        "ok": bool(np.all(~y[reason == 3])),
        "max_error": float(np.count_nonzero(y[reason == 3])),
        "tolerance": 0.0,
    }
    checks["insufficient_label"] = {
        "ok": bool(np.all(~y[reason == 2])),
        "max_error": float(np.count_nonzero(y[reason == 2])),
        "tolerance": 0.0,
    }
    checks["baseline_zero_label"] = {
        "ok": bool(np.all(y[reason == 1])),
        "max_error": float(np.count_nonzero(~y[reason == 1])),
        "tolerance": 0.0,
    }

    ledger = meta["ledger"]
    accepted_total = ledger["accepted"] + ledger.get("discarded", 0)
    checks["ledger"] = {
        "ok": bool(
            ledger["r1"]
            + ledger["r2"]
            + ledger["r3"]
            + ledger.get("r4", 0)
            + accepted_total
            == ledger["drawn"]
        ),
        "max_error": float(
            abs(
                ledger["r1"]
                + ledger["r2"]
                + ledger["r3"]
                + ledger.get("r4", 0)
                + accepted_total
                - ledger["drawn"]
            )
        ),
        "tolerance": 0.0,
    }

    # Feature masks must be a subset of the validity mask for their elements.
    feature_positions = {
        element: index for index, element in enumerate(meta["config"]["feature_elements"])
    }
    ref_position = feature_positions[meta["reference"]]
    valid = arrays["valid_features"].astype(bool)
    violations = 0.0
    for index, name in enumerate(meta["feature_names"]):
        element = name.split("/")[0]
        position = feature_positions[element]
        violations += float(
            np.count_nonzero(mask[:, index] & ~valid[:, position])
        )
        violations += float(
            np.count_nonzero(mask[:, index] & ~valid[:, ref_position])
        )
    checks["feature_mask_subset"] = {
        "ok": bool(violations == 0.0),
        "max_error": violations,
        "tolerance": 0.0,
    }

    if "obs_flags" in arrays:
        flags = arrays["obs_flags"]
        observation_mask = arrays["obs_mask"].astype(bool)
        checks["obs_flags"] = {
            "ok": bool(np.all((flags >= 0) & (flags <= 2))),
            "max_error": float(np.max(np.maximum(flags - 2, -flags), initial=0.0)),
            "tolerance": 0.0,
        }
        checks["obs_mask"] = {
            "ok": bool(np.array_equal(observation_mask, flags > 0)),
            "max_error": float(np.count_nonzero(observation_mask != (flags > 0))),
            "tolerance": 0.0,
        }
        bounds = meta["config"]["prior"]["feh_range"]
        if bounds is not None:
            feh_obs = arrays["feh_obs"].astype(float)
            inside = (
                np.isfinite(feh_obs)
                & (feh_obs >= bounds[0])
                & (feh_obs <= bounds[1])
            )
            checks["feh_obs_range"] = {
                "ok": bool(np.all(inside)),
                "max_error": float(np.count_nonzero(~inside)),
                "tolerance": 0.0,
            }
        weights = arrays["importance_weight"].astype(float)
        checks["importance_weight"] = {
            "ok": bool(np.all(np.isfinite(weights)) and np.all(weights > 0.0)),
            "max_error": float(np.count_nonzero(~(np.isfinite(weights) & (weights > 0.0)))),
            "tolerance": 0.0,
        }

    return {
        "ok": all(check["ok"] for check in checks.values()),
        "checks": checks,
    }


def _transfer_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Return AUC, Brier score, and accuracy."""
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    auc = (
        float(roc_auc_score(y, probability))
        if len(np.unique(y)) == 2
        else float("nan")
    )
    return {
        "auc": auc,
        "brier": float(brier_score_loss(y, probability)),
        "accuracy": float(np.mean((probability >= 0.5) == y)),
        "y_rate": float(y.mean()) if len(y) else float("nan"),
        "n": int(len(y)),
    }


def transfer_report(
    base: dict[str, np.ndarray],
    variants: dict[str, dict[str, np.ndarray]],
    seed: int = 0,
    test_size: float = 0.3,
) -> dict[str, Any]:
    """Train on the base prior and evaluate each ablation prior."""
    y = np.asarray(base["y"], dtype=int)
    train, test = train_test_split(
        np.arange(len(y)),
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    out: dict[str, Any] = {
        "train_n": int(len(train)),
        "base_test_n": int(len(test)),
        "variants": {},
    }
    for model_name, matrix_function in (
        ("chemistry", _design_matrix),
        ("nuisance", _nuisance_matrix),
    ):
        model = HistGradientBoostingClassifier(
            max_iter=150,
            max_depth=3,
            learning_rate=0.05,
            random_state=seed,
        )
        model.fit(matrix_function(base)[train], y[train])
        rows: dict[str, Any] = {
            "base": _transfer_metrics(
                y[test], model.predict_proba(matrix_function(base)[test])[:, 1]
            )
        }
        for name, arrays in variants.items():
            probability = model.predict_proba(matrix_function(arrays))[:, 1]
            rows[name] = _transfer_metrics(arrays["y"], probability)
        out["variants"][model_name] = rows
    return out


def ablation_configs(config: ModelConfig) -> dict[str, ModelConfig]:
    """Return prior-structure variants for a small ablation study."""
    return {
        "base": config,
        "no_pisn": replace(
            config,
            prior=replace(config.prior, n_pisn_probs=(1.0, 0.0, 0.0, 0.0)),
        ),
        "no_iii": replace(
            config,
            prior=replace(config.prior, lambda_iii_range=(1.0e-12, 1.0e-9)),
        ),
        "wide_lambda": replace(
            config,
            prior=replace(
                config.prior,
                n_pisn_probs=(0.25, 0.25, 0.25, 0.25),
                lambda_iii_range=(0.1, 5000.0),
                n_iii_max=8192,
            ),
        ),
        "no_r4": replace(
            config,
            prior=replace(config.prior, feh_range=None),
        ),
        "low_zcr": replace(
            config,
            prior=replace(config.prior, z_cr=1.0e-4),
        ),
        "zcr_1e-6": replace(
            config,
            prior=replace(config.prior, z_cr=1.0e-6),
        ),
        "cap_0.05": replace(
            config,
            prior=replace(config.prior, mstar_total_max_ratio=0.05),
        ),
    }


def run_ablation(
    config: ModelConfig,
    n_samples: int = 1000,
    output_dir: str | Path | None = None,
    verbose: bool = False,
) -> dict[str, dict[str, Any]]:
    """Generate and diagnose every ablation variant."""
    summary: dict[str, dict[str, Any]] = {}
    for name, variant in ablation_configs(config).items():
        variant = replace(
            variant,
            runtime=replace(
                variant.runtime, chunk_size=max(200, min(n_samples, 1000))
            ),
        )
        arrays, meta = generate_dataset(
            variant, n_samples=n_samples, verbose=verbose
        )
        report = diagnose(arrays, meta)
        if output_dir is not None:
            save_dataset(Path(output_dir) / f"{name}.npz", arrays, meta)
        summary[name] = {
            "y_rate": report["y_rate"],
            "acceptance": report["ledger"]["accepted_total"],
            "r1": report["ledger"]["r1"],
            "r2": report["ledger"]["r2"],
            "r3": report["ledger"]["r3"],
            "feature_coverage": report["feature_coverage"],
        }
    return summary
